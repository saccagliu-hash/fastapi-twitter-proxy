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
        # Montserrat Bold — inclusa nel repo, priorità massima
        os.path.join(os.path.dirname(__file__), "..", "fonts", "Montserrat-Bold.ttf"),
        # Fallback di sistema
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


def _fetch_unsplash(query: str, api_key: str, output_path: str, offset: int) -> bool:
    try:
        resp = requests.get(
            "https://api.unsplash.com/search/photos",
            headers={"Authorization": f"Client-ID {api_key}"},
            params={"query": query, "per_page": 20, "page": (offset // 20) + 1, "orientation": "landscape"},
            timeout=15,
        )
        logging.info("Unsplash [%s] status=%s", query, resp.status_code)
        if resp.status_code != 200:
            logging.warning("Unsplash error body: %s", resp.text[:200])
            return False
        results = resp.json().get("results", [])
        if not results:
            logging.warning("Unsplash: nessuna foto per query '%s'", query)
            return False
        photo = results[offset % len(results)]
        img_url = photo["urls"].get("regular") or photo["urls"].get("full")
        img_resp = requests.get(img_url, timeout=20)
        img_resp.raise_for_status()
        with Image.open(io.BytesIO(img_resp.content)) as raw:
            raw = raw.convert("RGB")
            raw = _crop_to_16_9(raw)
            raw = raw.resize((WIDTH, HEIGHT), Image.LANCZOS)
            raw.save(output_path, "JPEG", quality=92)
        return True
    except Exception as exc:
        logging.warning("Unsplash fetch fallito per '%s': %s", query, exc)
        return False


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


def _fetch_google(query: str, api_key: str, cx: str, output_path: str, offset: int) -> bool:
    try:
        start = (offset % 9) + 1  # Google CSE: start 1-91 (max 100 results)
        resp = requests.get(
            "https://www.googleapis.com/customsearch/v1",
            params={
                "key": api_key,
                "cx": cx,
                "q": query,
                "searchType": "image",
                "num": 10,
                "start": start,
                "imgSize": "large",
                "imgType": "photo",
                "safe": "active",
                "gl": "it",
            },
            timeout=15,
        )
        logging.info("Google CSE [%s] status=%s", query, resp.status_code)
        if resp.status_code != 200:
            logging.warning("Google CSE error: %s", resp.text[:200])
            return False
        items = resp.json().get("items", [])
        if not items:
            logging.warning("Google CSE: nessuna foto per query '%s'", query)
            return False
        # Prova ogni risultato finché uno è scaricabile
        for item in items:
            img_url = item.get("link")
            if not img_url:
                continue
            try:
                img_resp = requests.get(img_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
                img_resp.raise_for_status()
                with Image.open(io.BytesIO(img_resp.content)) as raw:
                    raw = raw.convert("RGB")
                    raw = _crop_to_16_9(raw)
                    raw = raw.resize((WIDTH, HEIGHT), Image.LANCZOS)
                    raw.save(output_path, "JPEG", quality=92)
                return True
            except Exception:
                continue
        logging.warning("Google CSE: nessuna immagine scaricabile per '%s'", query)
        return False
    except Exception as exc:
        logging.warning("Google CSE fetch fallito per '%s': %s", query, exc)
        return False


def _fetch_dalle(query: str, context: str, api_key: str, output_path: str) -> bool:
    try:
        prompt = (
            f"Professional photorealistic image, cinematic lighting, ultra high quality. "
            f"Subject: {query}. "
            f"No text, no watermarks, no people unless essential, wide angle shot."
        )
        resp = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": "dall-e-3",
                "prompt": prompt,
                "size": "1792x1024",
                "quality": "standard",
                "n": 1,
            },
            timeout=90,
        )
        logging.info("DALL-E status=%s", resp.status_code)
        if resp.status_code != 200:
            logging.warning("DALL-E error: %s", resp.text[:300])
            return False
        img_url = resp.json()["data"][0]["url"]
        img_resp = requests.get(img_url, timeout=30)
        img_resp.raise_for_status()
        with Image.open(io.BytesIO(img_resp.content)) as raw:
            raw = raw.convert("RGB")
            raw = _crop_to_16_9(raw)
            raw = raw.resize((WIDTH, HEIGHT), Image.LANCZOS)
            raw.save(output_path, "JPEG", quality=92)
        return True
    except Exception as exc:
        logging.warning("DALL-E fetch fallito: %s", exc)
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
    unsplash_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_cx: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    dalle_context: str = "",
) -> str:
    fetched = False
    if openai_api_key and search_query:
        fetched = _fetch_dalle(search_query, dalle_context, openai_api_key, output_path)
    if not fetched and google_api_key and google_cx and search_query:
        fetched = _fetch_google(search_query, google_api_key, google_cx, output_path, theme_idx)
    if not fetched and unsplash_api_key and search_query:
        fetched = _fetch_unsplash(search_query, unsplash_api_key, output_path, theme_idx)
    if not fetched and pexels_api_key and search_query:
        fetched = _fetch_pexels(search_query, pexels_api_key, output_path, theme_idx)
    if not fetched:
        img = _make_gradient(theme_idx)
        img.save(output_path, "JPEG", quality=92)
    return output_path


def make_karaoke_overlay(text: str, spoken_words: int) -> np.ndarray:
    """Karaoke: spoken=bianco dimmed, corrente=giallo brillante con box, future=grigio."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 210
    draw.rectangle([(0, HEIGHT - bar_h), (WIDTH, HEIGHT)], fill=(0, 0, 0, 200))

    font = _get_font(52)
    wrapped = textwrap.fill(text[:250], width=32)
    lines = wrapped.split("\n")[:3]
    line_h = 68
    total_h = len(lines) * line_h
    start_y = HEIGHT - bar_h + (bar_h - total_h) // 2 + 4

    try:
        sw = (draw.textbbox((0, 0), "a b", font=font)[2]
              - draw.textbbox((0, 0), "ab", font=font)[2])
        space_w = max(8, sw)
    except Exception:
        space_w = 15

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
                color = (210, 210, 210, 220)   # già detto: bianco dimmed
            elif gidx == spoken_words:
                # parola corrente: box giallo + testo nero per massimo contrasto
                pad = 6
                draw.rounded_rectangle(
                    [(x - pad, y - 2), (x + ww + pad, y + line_h - 10)],
                    radius=6, fill=(255, 220, 0, 240),
                )
                color = (20, 20, 20, 255)      # testo scuro sul box giallo
            else:
                color = (120, 120, 120, 180)   # futuro: grigio scuro

            if gidx != spoken_words:
                for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
                    draw.text((x + dx, y + dy), word, font=font, fill=(0, 0, 0, 200))
            draw.text((x, y), word, font=font, fill=color)
            x += ww + space_w

        word_counter += len(line_words)

    return np.array(overlay)


def bake_intro_card(bg_path: str, topic: str, output_path: str) -> str:
    """Intro card su foto Pexels: vignette scura + box + titolo giallo centrato."""
    with Image.open(bg_path) as bg:
        bg = bg.convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Vignette: bordi scuri per leggibilità
    for y in range(HEIGHT):
        top_alpha = max(0, int(200 * (1 - y / (HEIGHT * 0.45))))
        bot_alpha = max(0, int(220 * ((y - HEIGHT * 0.55) / (HEIGHT * 0.45))))
        alpha = min(255, top_alpha + bot_alpha)
        if alpha > 0:
            draw.line([(0, y), (WIDTH, y)], fill=(0, 0, 0, alpha))

    # Box semi-trasparente centrato per il titolo
    title_font = _get_font(72)
    wrapped = textwrap.fill(topic.upper(), width=18)
    lines = wrapped.split("\n")[:4]
    line_h = 86
    total_h = len(lines) * line_h
    mid_y = HEIGHT // 2
    pad = 28
    box_top = mid_y - total_h // 2 - pad
    box_bot = mid_y + total_h // 2 + pad
    draw.rectangle([(60, box_top), (WIDTH - 60, box_bot)], fill=(0, 0, 0, 160))

    # Linee decorative gialle
    draw.rectangle([(WIDTH // 2 - 180, box_top + 8), (WIDTH // 2 + 180, box_top + 14)],
                   fill=(255, 220, 0))
    draw.rectangle([(WIDTH // 2 - 180, box_bot - 14), (WIDTH // 2 + 180, box_bot - 8)],
                   fill=(255, 220, 0))

    # Testo titolo
    start_y = mid_y - total_h // 2
    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=title_font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        y = start_y + i * line_h
        for dx, dy in [(-3, 0), (3, 0), (0, -3), (0, 3)]:
            draw.text((x + dx, y + dy), line, font=title_font, fill=(0, 0, 0, 255))
        draw.text((x, y), line, font=title_font, fill=(255, 220, 0, 255))

    result = Image.alpha_composite(bg, overlay).convert("RGB")
    result.save(output_path, "JPEG", quality=92)
    return output_path


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


def bake_subtitle(bg_path: str, text: str, output_path: str) -> str:
    """Sottotitoli gialli su barra scura — font auto-ridotto se il testo non ci sta in 3 righe."""
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    bar_h = 210
    draw.rectangle([(0, HEIGHT - bar_h), (WIDTH, HEIGHT)], fill=(0, 0, 0, 200))

    # Auto-scale font: prova 52px, poi 44px, poi 38px finché il testo sta in 3 righe
    for font_size, wrap_width in [(52, 32), (44, 38), (38, 46)]:
        font = _get_font(font_size)
        wrapped = textwrap.fill(text[:300], width=wrap_width)
        lines = wrapped.split("\n")
        if len(lines) <= 3:
            break
    lines = lines[:3]
    line_h = int(font_size * 1.3)
    total_h = len(lines) * line_h
    start_y = HEIGHT - bar_h + (bar_h - total_h) // 2 + 4

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = max(30, (WIDTH - tw) // 2)
        y = start_y + i * line_h
        for dx, dy in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 220))
        draw.text((x, y), line, font=font, fill=(255, 220, 0, 255))

    with Image.open(bg_path) as img:
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, overlay).convert("RGB")
        img.save(output_path, "JPEG", quality=92)
    return output_path
