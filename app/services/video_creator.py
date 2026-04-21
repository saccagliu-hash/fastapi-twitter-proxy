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
    CompositeAudioClip,
    VideoClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from PIL import Image as PILImage

from .image_service import fetch_background, make_intro_card, make_karaoke_overlay
from .tts import generate_tts

MIN_SLIDE_DURATION = 5.5
WORDS_PER_SEGMENT = 18
MIN_SEGMENT_WORDS = 5

INTRO_DURATION = 3.0
N_KB_STEPS = 6    # frame karaoke pre-renderizzati per slide
KB_ZOOM = 0.07    # Ken Burns: zoom massimo 7%
MUSIC_VOL = 0.13  # volume musica ambient (~-18 dB)

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


def _ambient_music(duration: float, seed: int = 42, fps: int = 44100) -> AudioArrayClip:
    """Musica ambient procedurale: scala pentatonica minore con ampiezza respirante."""
    rng = np.random.default_rng(seed)
    n = max(1, int(fps * duration))
    t = np.linspace(0, duration, n, endpoint=False)

    # A minor pentatonic: A2–A4
    freqs = [110.0, 130.8, 146.8, 164.8, 196.0, 220.0, 261.6, 293.7, 329.6, 392.0]
    music = np.zeros(n, dtype=np.float64)
    for freq in freqs:
        amp = 0.07 + rng.random() * 0.06
        phase = rng.random() * 2 * np.pi
        # Leggero chorus: modulazione lentissima della frequenza
        mod = 1.0 + 0.0008 * np.sin(2 * np.pi * rng.uniform(0.05, 0.18) * t + phase)
        music += amp * np.sin(2 * np.pi * freq * mod * t + phase)

    # Respiro lento (8–12 s)
    period = rng.uniform(8.0, 12.0)
    music *= 0.55 + 0.45 * np.sin(2 * np.pi * t / period)

    # Shimmer armonico
    music += 0.025 * np.sin(2 * np.pi * 880.0 * t)

    # Fade in/out (2 s)
    fade = min(int(fps * 2), n // 4)
    if fade > 0:
        music[:fade] *= np.linspace(0, 1, fade) ** 2
        music[-fade:] *= np.linspace(1, 0, fade) ** 2

    music = (music / (np.max(np.abs(music)) + 1e-9) * MUSIC_VOL).astype(np.float32)

    # Stereo con leggera differenza L/R
    right = music * 0.97 + (0.015 * np.sin(2 * np.pi * 110.0 * t)).astype(np.float32)
    stereo = np.stack([music, right], axis=1)
    return AudioArrayClip(stereo, fps=fps)


# ---------------------------------------------------------------------------
# Ken Burns + Karaoke
# ---------------------------------------------------------------------------

def _karaoke_frames(bg_path: str, text: str, n_steps: int,
                    work_dir: str, slide_idx: int) -> list[str]:
    """Pre-genera n_steps JPEG con karaoke progressivo (nessun zoom: gestito da VideoClip)."""
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

        # --- Step 1: genera tutto l'audio TTS ---
        tts_paths: list[str] = []
        clip_durations: list[float] = []
        for i, segment in enumerate(segments):
            tts_path = os.path.join(work_dir, f"audio_{i}.mp3")
            tts_duration = await asyncio.to_thread(
                generate_tts, segment, voice, tts_path,
                elevenlabs_api_key, elevenlabs_voice_id,
            )
            clip_durations.append(max(tts_duration, MIN_SLIDE_DURATION))
            tts_paths.append(tts_path)

        # --- Step 2: traccia voce continua paddato per-slide ---
        padded_audio: list = [_silence(INTRO_DURATION)]
        for i, tts_path in enumerate(tts_paths):
            tts_clip = AudioFileClip(tts_path)
            pad_dur = clip_durations[i] - tts_clip.duration
            if pad_dur > 0.05:
                padded_audio.append(concatenate_audioclips([tts_clip, _silence(pad_dur)]))
            else:
                padded_audio.append(tts_clip)
        voice_audio = concatenate_audioclips(padded_audio)

        # --- Step 3: intro card con Ken Burns ---
        intro_path = os.path.join(work_dir, "intro.jpg")
        intro_arr = make_intro_card(topic)
        PILImage.fromarray(intro_arr).save(intro_path, "JPEG", quality=92)

        intro_make_fn = functools.partial(
            _kb_make_frame, [intro_path], 1, INTRO_DURATION, 1.0, 1.0 + KB_ZOOM * 0.5
        )
        intro_clip = VideoClip(intro_make_fn, duration=INTRO_DURATION).set_fps(24)
        clips = [intro_clip]

        # --- Step 4: slide con Ken Burns + Karaoke ---
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

            # Alterna zoom in / zoom out per varietà visiva
            z0, z1 = (1.0, 1.0 + KB_ZOOM) if i % 2 == 0 else (1.0 + KB_ZOOM, 1.0)
            dur = clip_durations[i]
            make_fn = functools.partial(_kb_make_frame, karaoke_paths, N_KB_STEPS, dur, z0, z1)
            clip = VideoClip(make_fn, duration=dur).set_fps(24)
            clips.append(clip)

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        # --- Step 5: concatena video ---
        final = concatenate_videoclips(clips, method="compose")

        # --- Step 6: mix voce + musica ambient ---
        video_dur = final.duration
        ambient = _ambient_music(video_dur, seed=abs(hash(topic)) % (2 ** 31))
        mixed_audio = CompositeAudioClip([
            voice_audio.set_duration(video_dur),
            ambient.set_duration(video_dur),
        ])
        final = final.set_audio(mixed_audio)

        # --- Step 7: loop se troppo corto ---
        if final.duration < min_duration:
            content = clips[1:]
            repeats = int(min_duration / max(final.duration - INTRO_DURATION, 1)) + 1
            ext_padded = padded_audio + padded_audio[1:] * repeats
            ext_audio = concatenate_audioclips(ext_padded)
            ambient_ext = _ambient_music(min_duration, seed=abs(hash(topic)) % (2 ** 31))
            final = concatenate_videoclips(clips + content * repeats, method="compose")
            final = final.subclip(0, min_duration)
            mixed_ext = CompositeAudioClip([
                ext_audio.set_duration(min_duration),
                ambient_ext.set_duration(min_duration),
            ])
            final = final.set_audio(mixed_ext)

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
