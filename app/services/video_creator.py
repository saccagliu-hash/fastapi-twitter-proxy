import asyncio
import os
import re
import shutil
import uuid
from typing import Optional

import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import AudioFileClip, ImageClip, concatenate_audioclips, concatenate_videoclips

from .image_service import fetch_background, make_intro_card, bake_subtitle
from .tts import generate_tts

MIN_SLIDE_DURATION = 3.5
WORDS_PER_SEGMENT = 18
MIN_SEGMENT_WORDS = 5  # Segmenti più corti vengono fusi nel precedente

INTRO_DURATION = 3.0
CROSSFADE = 0.4


def _split_script(script: str) -> list[str]:
    # Primo livello: spezza sulle frasi (. ! ?)
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
            # Frase lunga: svuota buffer corrente, poi spezza su sotto-clausole (,;:)
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

    # Post-processing: fonde segmenti orfani troppo corti nel precedente
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


def _silence(duration: float, fps: int = 44100) -> AudioArrayClip:
    samples = np.zeros((max(1, int(fps * duration)), 2), dtype=np.float32)
    return AudioArrayClip(samples, fps=fps)


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

        # --- Step 2: traccia audio unica continua (silenzio intro + TTS concatenati) ---
        # Una sola traccia elimina completamente i glitch ai tagli tra slide
        tts_audio_clips = [AudioFileClip(p) for p in tts_paths]
        full_audio = concatenate_audioclips([_silence(INTRO_DURATION)] + tts_audio_clips)

        # --- Step 3: slide video (senza audio individuale) ---
        intro_path = os.path.join(work_dir, "intro.jpg")
        intro_arr = make_intro_card(topic)
        from PIL import Image as PILImage
        PILImage.fromarray(intro_arr).save(intro_path, "JPEG", quality=92)
        intro_clip = ImageClip(intro_path).set_duration(INTRO_DURATION).crossfadein(0.5)
        clips = [intro_clip]

        for i, segment in enumerate(segments):
            bg_path = os.path.join(work_dir, f"bg_{i}.jpg")
            base_kw = image_keywords or topic
            await asyncio.to_thread(
                fetch_background, i, bg_path, pexels_api_key,
                _pexels_query(base_kw, i) if pexels_api_key else None,
            )
            slide_path = os.path.join(work_dir, f"slide_{i}.jpg")
            await asyncio.to_thread(bake_subtitle, bg_path, segment, slide_path)

            clip = ImageClip(slide_path).set_duration(clip_durations[i]).crossfadein(CROSSFADE)
            clips.append(clip)

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        # --- Step 4: concatena con crossfade visuale, attacca audio continuo ---
        final = concatenate_videoclips(clips, method="compose", padding=-CROSSFADE)
        final = final.set_audio(full_audio.set_duration(final.duration))

        if final.duration < min_duration:
            content = clips[1:]
            repeats = int(min_duration / max(final.duration - INTRO_DURATION, 1)) + 1
            extended_audio = concatenate_audioclips(
                [_silence(INTRO_DURATION)] + tts_audio_clips * (repeats + 1)
            )
            final = concatenate_videoclips(clips + content * repeats, method="compose", padding=-CROSSFADE)
            final = final.subclip(0, min_duration)
            final = final.set_audio(extended_audio.set_duration(min_duration))

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
