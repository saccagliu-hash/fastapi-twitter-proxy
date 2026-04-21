import asyncio
import functools
import os
import re
import shutil
import uuid
from typing import Optional

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    AudioFileClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from PIL import Image as PILImage

from .image_service import fetch_background, make_intro_card, make_karaoke_overlay
from .tts import generate_tts

MIN_SLIDE_DURATION = 2.0   # floor minimo per slide molto corte
WORDS_PER_SEGMENT = 18
MIN_SEGMENT_WORDS = 5

INTRO_DURATION = 3.0
N_KB_STEPS = 6    # frame karaoke pre-renderizzati per slide
KB_ZOOM = 0.07    # Ken Burns: zoom massimo 7%

try:
    _RESAMPLE = PILImage.Resampling.BILINEAR
except AttributeError:
    _RESAMPLE = PILImage.BILINEAR  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Script splitting
# ---------------------------------------------------------------------------

def _split_script(script: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", script.strip())
    segments: list[str] = []
    current: list[str] = []
    count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        words = sentence.split()

        if len(words) > WORDS_PER_SEGMENT:
            if current:
                segments.append(" ".join(current))
                current, count = [], 0
            sub_chunks = re.split(r"(?<=[,;:])\s+", sentence)
            sub_buf: list[str] = []
            sub_count = 0
            for sc in sub_chunks:
                sc = sc.strip()
                if not sc:
                    continue
                sc_words = sc.split()
                if len(sc_words) > WORDS_PER_SEGMENT:
                    if sub_buf:
                        segments.append(" ".join(sub_buf))
                        sub_buf, sub_count = [], 0
                    for j in range(0, len(sc_words), WORDS_PER_SEGMENT):
                        segments.append(" ".join(sc_words[j:j + WORDS_PER_SEGMENT]))
                elif sub_count + len(sc_words) > WORDS_PER_SEGMENT and sub_buf:
                    segments.append(" ".join(sub_buf))
                    sub_buf, sub_count = [sc], len(sc_words)
                else:
                    sub_buf.append(sc)
                    sub_count += len(sc_words)
            if sub_buf:
                segments.append(" ".join(sub_buf))
        elif count + len(words) > WORDS_PER_SEGMENT and current:
            segments.append(" ".join(current))
            current, count = [sentence], len(words)
        else:
            current.append(sentence)
            count += len(words)

    if current:
        segments.append(" ".join(current))

    result: list[str] = []
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
        if result and len(seg.split()) < MIN_SEGMENT_WORDS:
            result[-1] += " " + seg
        else:
            result.append(seg)

    return result


def _pexels_query(base_keywords: str, slide_idx: int) -> str:
    words = [w for w in base_keywords.split() if w.isalpha()][:3]
    return " ".join(words) if words else "business"


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _silence(duration: float, fps: int = 44100) -> AudioArrayClip:
    samples = np.zeros((max(1, int(fps * duration)), 2), dtype=np.float32)
    return AudioArrayClip(samples, fps=fps)


# ---------------------------------------------------------------------------
# Ken Burns + Karaoke
# ---------------------------------------------------------------------------

def _karaoke_frames(bg_path: str, text: str, n_steps: int,
                    work_dir: str, slide_idx: int) -> list[str]:
    """Pre-genera n_steps JPEG con karaoke progressivo."""
    words = text.split()
    n_words = len(words)
    paths: list[str] = []
    for step in range(n_steps):
        spoken = round(n_words * step / max(n_steps - 1, 1))
        spoken = min(spoken, n_words)

        with PILImage.open(bg_path) as bg:
            bg = bg.convert("RGBA")
        ov = PILImage.fromarray(make_karaoke_overlay(text, spoken))
        frame = PILImage.alpha_composite(bg, ov).convert("RGB")

        out = os.path.join(work_dir, f"kb_{slide_idx}_{step}.jpg")
        frame.save(out, "JPEG", quality=88)
        paths.append(out)

    return paths


def _kb_make_frame(paths: list[str], n_steps: int, dur: float,
                   z0: float, z1: float, t: float) -> np.ndarray:
    """make_frame per VideoClip: karaoke lookup + Ken Burns zoom per ogni frame."""
    step = min(int(t / max(dur, 0.001) * n_steps), n_steps - 1)
    zoom = z0 + (z1 - z0) * (t / max(dur, 0.001))

    with PILImage.open(paths[step]) as img:
        img = img.convert("RGB")
        w, h = img.size
        zw, zh = int(w * zoom), int(h * zoom)
        resized = img.resize((zw, zh), _RESAMPLE)
        x, y = (zw - w) // 2, (zh - h) // 2
        return np.array(resized.crop((x, y, x + w, y + h)))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def create_faceless_video(
    topic: str,
    script: str,
    voice: str,
    output_dir: str,
    pexels_api_key: Optional[str] = None,
    image_keywords: Optional[str] = None,
    elevenlabs_api_key: Optional[str] = None,
    elevenlabs_voice_id: str = "EXAVITQu4vr4xnSDxMaL",
    min_duration: float = 50.0,
    video_id: Optional[str] = None,
) -> dict:
    if video_id is None:
        video_id = str(uuid.uuid4())

    work_dir = os.path.join(output_dir, "tmp", video_id)
    os.makedirs(work_dir, exist_ok=True)

    try:
        segments = _split_script(script)

        # --- Step 1: UNA SOLA chiamata TTS per tutto lo script ---
        # Voce completamente naturale: nessuna discontinuità tra le slide
        narration_path = os.path.join(work_dir, "narration.mp3")
        narration_dur = await asyncio.to_thread(
            generate_tts, script, voice, narration_path,
            elevenlabs_api_key, elevenlabs_voice_id,
        )

        # Durata slide proporzionale al numero di parole (proxy del tempo parlato)
        seg_words = [len(seg.split()) for seg in segments]
        total_words = max(sum(seg_words), 1)
        clip_durations = [
            max(narration_dur * w / total_words, MIN_SLIDE_DURATION)
            for w in seg_words
        ]

        # Traccia voce: silenzio durante intro + narrazione + eventuale padding
        content_dur = sum(clip_durations)
        voice_parts: list = [_silence(INTRO_DURATION), AudioFileClip(narration_path)]
        gap = content_dur - narration_dur
        if gap > 0.1:
            voice_parts.append(_silence(gap))
        voice_audio = concatenate_audioclips(voice_parts)

        # --- Step 2: intro card con Ken Burns ---
        intro_path = os.path.join(work_dir, "intro.jpg")
        PILImage.fromarray(make_intro_card(topic)).save(intro_path, "JPEG", quality=92)

        intro_fn = functools.partial(
            _kb_make_frame, [intro_path], 1, INTRO_DURATION, 1.0, 1.0 + KB_ZOOM * 0.5
        )
        intro_clip = VideoClip(intro_fn, duration=INTRO_DURATION).set_fps(24)
        clips = [intro_clip]

        # --- Step 3: slide Ken Burns + Karaoke ---
        for i, segment in enumerate(segments):
            bg_path = os.path.join(work_dir, f"bg_{i}.jpg")
            base_kw = image_keywords or topic
            await asyncio.to_thread(
                fetch_background, i, bg_path, pexels_api_key,
                _pexels_query(base_kw, i) if pexels_api_key else None,
            )

            karaoke_paths = await asyncio.to_thread(
                _karaoke_frames, bg_path, segment, N_KB_STEPS, work_dir, i
            )

            z0, z1 = (1.0, 1.0 + KB_ZOOM) if i % 2 == 0 else (1.0 + KB_ZOOM, 1.0)
            dur = clip_durations[i]
            make_fn = functools.partial(_kb_make_frame, karaoke_paths, N_KB_STEPS, dur, z0, z1)
            clips.append(VideoClip(make_fn, duration=dur).set_fps(24))

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        # --- Step 4: concatena + attacca audio ---
        final = concatenate_videoclips(clips, method="compose")
        final = final.set_audio(voice_audio.set_duration(final.duration))

        # --- Step 5: loop se troppo corto ---
        if final.duration < min_duration:
            content = clips[1:]
            repeats = int(min_duration / max(final.duration - INTRO_DURATION, 1)) + 1
            narration_clip = AudioFileClip(narration_path)
            ext_voice = concatenate_audioclips(
                [_silence(INTRO_DURATION)] + [narration_clip] * (repeats + 1)
            )
            final = concatenate_videoclips(clips + content * repeats, method="compose")
            final = final.subclip(0, min_duration)
            final = final.set_audio(ext_voice.set_duration(min_duration))

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
