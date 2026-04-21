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

try:
    _RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE = Image.BILINEAR  # type: ignore[attr-defined]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    explicit = [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]
    for path in explicit:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass

    import glob
    for pattern in ["/usr/share/fonts/**/*Bold*.ttf", "/usr/share/fonts/**/*bold*.ttf",
                    "/usr/share/fonts/**/*.ttf"]:
        matches = glob.glob(pattern, recursive=True)
        for path in matches:
            try:
                font = ImageFont.truetype(path, size)
                logging.info("Font trovato: %s", path)
                return font
            except Exception:
                pass

    logging.warning("Nessun font TTF trovato, uso bitmap default")
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
    fetched = False
    if pexels_api_key and search_query:
        fetched = _fetch_pexels(search_query, pexels_api_key, output_path, theme_idx)
    if not fetched:
        img = _make_gradient(theme_idx)
        img.save(output_path, "JPEG", quality=92)
    return output_path


def make_karaoke_overlay(text: str, spoken_words: int) -> np.ndarray:
    """Sottotitoli karaoke: spoken=bianco, parola corrente=giallo, future=grigio."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 200
    draw.rectangle([(0, HEIGHT - bar_h), (WIDTH, HEIGHT)], fill=(0, 0, 0, 185))

    font = _get_font(48)
    wrapped = textwrap.fill(text[:250], width=36)
    lines = wrapped.split("\n")[:3]
    line_h = 60
    total_h = len(lines) * line_h
    start_y = HEIGHT - bar_h + (bar_h - total_h) // 2

    # Calcola larghezza spazio per il font corrente
    try:
        sw = (draw.textbbox((0, 0), "a b", font=font)[2]
              - draw.textbbox((0, 0), "ab", font=font)[2])
        space_w = max(6, sw)
    except Exception:
        space_w = 13

    word_counter = 0
    for i, line in enumerate(lines):
        line_words = line.split()
        if not line_words:
            continue
        y = start_y + i * line_h

        word_widths = [draw.textbbox((0, 0), w, font=font)[2]
                       - draw.textbbox((0, 0), w, font=font)[0]
                       for w in line_words]
        total_w = sum(word_widths) + space_w * (len(line_words) - 1)
        x = max(30, (WIDTH - total_w) // 2)

        for j, (word, ww) in enumerate(zip(line_words, word_widths)):
            gidx = word_counter + j
            if gidx < spoken_words:
                color = (255, 255, 255, 255)   # già detto: bianco
            elif gidx == spoken_words:
                color = (255, 220, 0, 255)     # corrente: giallo
            else:
                color = (150, 150, 150, 210)   # futuro: grigio

            for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                draw.text((x + dx, y + dy), word, font=font, fill=(0, 0, 0, 255))
            draw.text((x, y), word, font=font, fill=color)
            x += ww + space_w

        word_counter += len(line_words)

    return np.array(overlay)


def make_intro_card(topic: str) -> np.ndarray:
    """Card di apertura con il titolo del video."""
    img = _make_gradient(0)
    draw = ImageDraw.Draw(img)

    draw.rectangle([(WIDTH // 2 - 200, HEIGHT // 2 - 90), (WIDTH // 2 + 200, HEIGHT // 2 - 85)],
                   fill=(255, 220, 0))

    title_font = _get_font(72)

    wrapped = textwrap.fill(topic.upper(), width=20)
    lines = wrapped.split("\n")[:4]
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

    draw.rectangle([(WIDTH // 2 - 200, HEIGHT // 2 + total_h // 2 + 10),
                    (WIDTH // 2 + 200, HEIGHT // 2 + total_h // 2 + 15)],
                   fill=(255, 220, 0))

    return np.array(img)
