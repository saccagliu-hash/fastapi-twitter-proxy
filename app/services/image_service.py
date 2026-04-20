import io
import logging
import os
import textwrap
from typing import Optional

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720

_GRADIENT_THEMES = [
    ((8, 8, 40), (50, 15, 110)),
    ((8, 35, 8), (15, 90, 50)),
    ((45, 8, 8), (110, 35, 15)),
    ((8, 25, 55), (15, 60, 120)),
    ((35, 8, 45), (80, 15, 85)),
    ((45, 35, 8), (110, 80, 15)),
    ((8, 35, 35), (15, 90, 90)),
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    return ImageFont.load_default()


def _make_gradient(theme_idx: int) -> Image.Image:
    top, bottom = _GRADIENT_THEMES[theme_idx % len(_GRADIENT_THEMES)]
    img = Image.new("RGB", (WIDTH, HEIGHT))
    draw = ImageDraw.Draw(img)
    for y in range(HEIGHT):
        ratio = y / HEIGHT
        color = tuple(int(top[c] + (bottom[c] - top[c]) * ratio) for c in range(3))
        draw.line([(0, y), (WIDTH, y)], fill=color)
    return img


def _crop_to_16_9(img: Image.Image) -> Image.Image:
    w, h = img.size
    target = 16 / 9
    if w / h > target:
        new_w = int(h * target)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))
    return img


def _fetch_pexels(query: str, api_key: str, output_path: str, offset: int) -> bool:
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": api_key},
            params={"query": query, "per_page": 15, "page": (offset // 15) + 1},
            timeout=15,
        )
        logging.info("Pexels [%s] status=%s", query, resp.status_code)
        if resp.status_code != 200:
            logging.warning("Pexels error body: %s", resp.text[:200])
            return False
        photos = resp.json().get("photos", [])
        if not photos:
            logging.warning("Pexels: nessuna foto per query '%s'", query)
            return False
        photo = photos[offset % len(photos)]
        img_url = photo["src"].get("large") or photo["src"].get("original")
        img_resp = requests.get(img_url, timeout=20)
        img_resp.raise_for_status()
        with Image.open(io.BytesIO(img_resp.content)) as raw:
            raw = raw.convert("RGB")
            raw = _crop_to_16_9(raw)
            raw = raw.resize((WIDTH, HEIGHT), Image.LANCZOS)
            raw.save(output_path, "JPEG", quality=92)
        return True
    except Exception as exc:
        logging.warning("Pexels fetch fallito per '%s': %s", query, exc)
        return False


def test_pexels(api_key: str) -> dict:
    try:
        resp = requests.get(
            "https://api.pexels.com/v1/search",
            headers={"Authorization": api_key},
            params={"query": "nature", "per_page": 1},
            timeout=10,
        )
        return {"status_code": resp.status_code, "ok": resp.status_code == 200, "body": resp.json()}
    except Exception as exc:
        return {"status_code": None, "ok": False, "error": str(exc)}


def fetch_background(
    theme_idx: int,
    output_path: str,
    pexels_api_key: Optional[str] = None,
    search_query: Optional[str] = None,
) -> str:
    """Crea o scarica l'immagine di sfondo SENZA testo (il testo viene aggiunto come overlay)."""
    fetched = False
    if pexels_api_key and search_query:
        fetched = _fetch_pexels(search_query, pexels_api_key, output_path, theme_idx)

    if not fetched:
        img = _make_gradient(theme_idx)
        img.save(output_path, "JPEG", quality=92)

    return output_path


def make_subtitle_overlay(text: str) -> np.ndarray:
    """
    Crea un overlay RGBA (WIDTH x HEIGHT) con i sottotitoli stile YouTube.
    Testo giallo, outline nero spesso, barra scura semitrasparente in basso.
    """
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 140
    draw.rectangle([(0, HEIGHT - bar_h), (WIDTH, HEIGHT)], fill=(0, 0, 0, 170))

    font = _get_font(52)
    wrapped = textwrap.fill(text[:160], width=58)
    lines = wrapped.split("\n")[:2]
    line_h = 62
    total_h = len(lines) * line_h
    start_y = HEIGHT - bar_h + (bar_h - total_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = start_y + i * line_h
        # Thick black outline (8 directions)
        for dx, dy in [(-3,0),(3,0),(0,-3),(0,3),(-2,-2),(2,-2),(-2,2),(2,2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        # Yellow text
        draw.text((x, y), line, font=font, fill=(255, 220, 0, 255))

    return np.array(overlay)


def make_intro_card(topic: str) -> np.ndarray:
    """Card di apertura con il titolo del video."""
    img = _make_gradient(0)
    draw = ImageDraw.Draw(img)

    # Linea decorativa
    draw.rectangle([(WIDTH // 2 - 200, HEIGHT // 2 - 90), (WIDTH // 2 + 200, HEIGHT // 2 - 85)],
                   fill=(255, 220, 0))

    title_font = _get_font(72)
    sub_font = _get_font(36)

    wrapped = textwrap.fill(topic.upper(), width=28)
    lines = wrapped.split("\n")
    line_h = 82
    total_h = len(lines) * line_h
    start_y = HEIGHT // 2 - total_h // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = start_y + i * line_h
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(0, 0, 0))
        draw.text((x, y), line, font=title_font, fill=(255, 220, 0))

    # Linea decorativa in basso
    draw.rectangle([(WIDTH // 2 - 200, HEIGHT // 2 + total_h // 2 + 10),
                    (WIDTH // 2 + 200, HEIGHT // 2 + total_h // 2 + 15)],
                   fill=(255, 220, 0))

    return np.array(img)


def bake_subtitle(bg_path: str, text: str, output_path: str) -> str:
    """Applica i sottotitoli stile YouTube su un'immagine di sfondo e salva."""
    with Image.open(bg_path) as img:
        img = img.convert("RGBA")
        overlay = Image.fromarray(make_subtitle_overlay(text))
        img = Image.alpha_composite(img, overlay).convert("RGB")
        img.save(output_path, "JPEG", quality=92)
    return output_path


def create_slide(
    text: str,
    theme_idx: int,
    output_path: str,
    pexels_api_key: Optional[str] = None,
    search_query: Optional[str] = None,
) -> str:
    fetch_background(theme_idx, output_path, pexels_api_key, search_query)
    return bake_subtitle(output_path, text, output_path)
