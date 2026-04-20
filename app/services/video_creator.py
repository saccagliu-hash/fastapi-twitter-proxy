import asyncio
import os
import re
import shutil
import uuid
from typing import Optional

import numpy as np
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image as PILImage

from .image_service import (
    WIDTH,
    HEIGHT,
    fetch_background,
    make_intro_card,
    make_subtitle_overlay,
)
from .tts import generate_tts

MIN_SLIDE_DURATION = 3.5
WORDS_PER_SEGMENT = 22
FADE_DURATION = 0.4
INTRO_DURATION = 3.0

_STOP_WORDS = {
    "il", "la", "lo", "le", "i", "gli", "un", "una", "uno", "e", "è",
    "in", "a", "di", "da", "con", "su", "per", "tra", "fra", "che",
    "the", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
}


def _split_script(script: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    segments: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if count + len(words) > WORDS_PER_SEGMENT and current:
            segments.append(" ".join(current))
            current = [sentence]
            count = len(words)
        else:
            current.append(sentence)
            count += len(words)
    if current:
        segments.append(" ".join(current))
    return [s for s in segments if s.strip()]


def _keywords(text: str, topic: str) -> str:
    words = [w.lower().strip(".,!?;:") for w in text.split()]
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 3]
    return f"{topic} {' '.join(keywords[:4])}"


def _ken_burns_clip(img_path: str, duration: float, zoom_in: bool) -> VideoClip:
    """Effetto Ken Burns: lento zoom in o out sull'immagine."""
    pil_img = PILImage.open(img_path).convert("RGB")
    w, h = pil_img.size
    zoom = 0.08

    def make_frame(t):
        progress = min(t / max(duration, 0.001), 1.0)
        scale = (1.0 + zoom * progress) if zoom_in else (1.0 + zoom * (1.0 - progress))
        new_w, new_h = int(w * scale), int(h * scale)
        resized = pil_img.resize((new_w, new_h), PILImage.BILINEAR)
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        return np.array(resized.crop((left, top, left + w, top + h)))

    return VideoClip(make_frame, duration=duration).set_fps(24)


def _apply_subtitle(clip: VideoClip, subtitle_rgba: np.ndarray) -> VideoClip:
    """Applica l'overlay dei sottotitoli su ogni frame del clip."""
    alpha = subtitle_rgba[:, :, 3:4].astype(np.float32) / 255.0
    rgb = subtitle_rgba[:, :, :3].astype(np.float32)

    def add_sub(frame):
        return (frame.astype(np.float32) * (1.0 - alpha) + rgb * alpha).astype(np.uint8)

    return clip.fl_image(add_sub)


def _make_intro_clip(topic: str) -> VideoClip:
    """Clip di apertura con il titolo."""
    intro_arr = make_intro_card(topic)
    clip = ImageClip(intro_arr).set_duration(INTRO_DURATION).set_fps(24)
    return clip.crossfadeout(FADE_DURATION)


async def create_faceless_video(
    topic: str,
    script: str,
    voice: str,
    output_dir: str,
    pexels_api_key: Optional[str] = None,
    min_duration: float = 50.0,
    video_id: Optional[str] = None,
) -> dict:
    if video_id is None:
        video_id = str(uuid.uuid4())

    work_dir = os.path.join(output_dir, "tmp", video_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        segments = _split_script(script)
        clips = [_make_intro_clip(topic)]

        for i, segment in enumerate(segments):
            tts_path = os.path.join(work_dir, f"audio_{i}.mp3")
            tts_duration = await asyncio.to_thread(generate_tts, segment, voice, tts_path)
            clip_duration = max(tts_duration, MIN_SLIDE_DURATION)

            img_path = os.path.join(work_dir, f"bg_{i}.jpg")
            await asyncio.to_thread(
                fetch_background,
                i,
                img_path,
                pexels_api_key,
                _keywords(segment, topic) if pexels_api_key else None,
            )

            # Ken Burns: alterna zoom in/out
            kb = _ken_burns_clip(img_path, clip_duration, zoom_in=(i % 2 == 0))

            # Sottotitoli stile YouTube
            subtitle_rgba = make_subtitle_overlay(segment)
            kb = _apply_subtitle(kb, subtitle_rgba)

            # Audio TTS
            audio = AudioFileClip(tts_path)
            kb = kb.set_audio(audio)

            # Fade in/out per transizioni fluide
            kb = kb.crossfadein(FADE_DURATION).crossfadeout(FADE_DURATION)

            clips.append(kb)

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        final = concatenate_videoclips(clips, method="compose", padding=-FADE_DURATION)

        if final.duration < min_duration:
            content_clips = clips[1:]  # escludi l'intro dalla ripetizione
            repeats = int(min_duration / max(final.duration - INTRO_DURATION, 1)) + 1
            repeated = clips + content_clips * repeats
            final = concatenate_videoclips(repeated, method="compose", padding=-FADE_DURATION)
            final = final.subclip(0, min_duration)

        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        await asyncio.to_thread(
            final.write_videofile,
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
            preset="ultrafast",
            ffmpeg_params=["-crf", "28"],
            temp_audiofile=os.path.join(work_dir, "temp_audio.m4a"),
            remove_temp=True,
            logger=None,
        )
        duration = final.duration
        final.close()

        return {"video_id": video_id, "status": "completed", "output_path": output_path, "duration": duration}

    except Exception as exc:
        return {"video_id": video_id, "status": "failed", "error": str(exc)}

    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
