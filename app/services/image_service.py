import io
import logging
import os
import textwrap
from typing import Optional

import requests
from PIL import Image, ImageDraw, ImageFont

WIDTH, HEIGHT = 1280, 720

_GRADIENT_THEMES = [
    ((10, 10, 50), (60, 20, 120)),
    ((10, 40, 10), (20, 100, 60)),
    ((50, 10, 10), (120, 40, 20)),
    ((10, 30, 60), (20, 70, 130)),
    ((40, 10, 50), (90, 20, 90)),
    ((50, 40, 10), (120, 90, 20)),
    ((10, 40, 40), (20, 100, 100)),
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


def _draw_centered_text(draw: ImageDraw.ImageDraw, text: str, y_center: int, font: ImageFont.FreeTypeFont):
    wrapped = textwrap.fill(text, width=48)
    lines = wrapped.split("\n")
    line_h = font.size + 12
    total_h = len(lines) * line_h
    start_y = y_center - total_h // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = start_y + i * line_h
        draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))


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
    """Testa la connessione Pexels — usato dall'endpoint /api/v1/test-pexels."""
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


def _add_subtitle_bar(img: Image.Image, text: str) -> Image.Image:
    font = _get_font(46)
    bar_h = 160
    overlay = Image.new("RGBA", (WIDTH, bar_h), (0, 0, 0, 170))
    base = img.convert("RGBA")
    base.paste(overlay, (0, HEIGHT - bar_h), overlay)
    img = base.convert("RGB")
    draw = ImageDraw.Draw(img)

    wrapped = textwrap.fill(text[:180], width=72)
    lines = wrapped.split("\n")[:3]
    line_h = font.size + 8
    total_h = len(lines) * line_h
    start_y = HEIGHT - bar_h + (bar_h - total_h) // 2

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = start_y + i * line_h
        draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))

    return img


def create_slide(
    text: str,
    theme_idx: int,
    output_path: str,
    pexels_api_key: Optional[str] = None,
    search_query: Optional[str] = None,
) -> str:
    fetched = False
    if pexels_api_key and search_query:
        fetched = _fetch_pexels(search_query, pexels_api_key, output_path, theme_idx)

    if fetched:
        with Image.open(output_path) as img:
            img = _add_subtitle_bar(img, text)
            img.save(output_path, "JPEG", quality=92)
    else:
        img = _make_gradient(theme_idx)
        draw = ImageDraw.Draw(img)
        font = _get_font(58)
        _draw_centered_text(draw, text, HEIGHT // 2, font)
        img.save(output_path, "JPEG", quality=92)

    return output_path
