import asyncio
import os
import re
import shutil
import uuid
from typing import Optional

from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

from .image_service import fetch_background, make_intro_card, bake_subtitle
from .tts import generate_tts

MIN_SLIDE_DURATION = 3.5
WORDS_PER_SEGMENT = 18  # Slide più brevi = testo sempre leggibile per intero

_STOP_WORDS = {
    "il", "la", "lo", "le", "i", "gli", "un", "una", "uno", "e", "è",
    "in", "a", "di", "da", "con", "su", "per", "tra", "fra", "che",
    "the", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that",
}


def _split_script(script: str) -> list[str]:
    # Divide su . ! ? , ; : — così frasi lunghe con virgole creano slide separate
    chunks = re.split(r"(?<=[.!?,;:])\s+", script.strip())

    segments: list[str] = []
    current: list[str] = []
    count = 0

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        words = chunk.split()

        # Chunk singolo già troppo lungo: spezzalo a forza per parola
        if len(words) > WORDS_PER_SEGMENT:
            if current:
                segments.append(" ".join(current))
                current, count = [], 0
            for j in range(0, len(words), WORDS_PER_SEGMENT):
                segments.append(" ".join(words[j:j + WORDS_PER_SEGMENT]))
        elif count + len(words) > WORDS_PER_SEGMENT and current:
            segments.append(" ".join(current))
            current, count = [chunk], len(words)
        else:
            current.append(chunk)
            count += len(words)

    if current:
        segments.append(" ".join(current))

    return [s for s in segments if s.strip()]


def _pexels_query(base_keywords: str, slide_idx: int) -> str:
    words = [w for w in base_keywords.split() if w.isalpha()][:3]
    return " ".join(words) if words else "business"


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
        clips = []

        # Intro card (3 secondi, nessun audio)
        intro_path = os.path.join(work_dir, "intro.jpg")
        intro_arr = make_intro_card(topic)
        from PIL import Image as PILImage
        PILImage.fromarray(intro_arr).save(intro_path, "JPEG", quality=92)
        intro_clip = ImageClip(intro_path).set_duration(3.0).fadein(0.5).fadeout(0.5)
        clips.append(intro_clip)

        for i, segment in enumerate(segments):
            tts_path = os.path.join(work_dir, f"audio_{i}.mp3")
            tts_duration = await asyncio.to_thread(
                generate_tts, segment, voice, tts_path,
                elevenlabs_api_key, elevenlabs_voice_id,
            )
            clip_duration = max(tts_duration, MIN_SLIDE_DURATION)

            bg_path = os.path.join(work_dir, f"bg_{i}.jpg")
            base_kw = image_keywords or topic
            await asyncio.to_thread(
                fetch_background, i, bg_path, pexels_api_key,
                _pexels_query(base_kw, i) if pexels_api_key else None,
            )

            slide_path = os.path.join(work_dir, f"slide_{i}.jpg")
            await asyncio.to_thread(bake_subtitle, bg_path, segment, slide_path)

            audio = AudioFileClip(tts_path)
            clip = (
                ImageClip(slide_path)
                .set_duration(clip_duration)
                .set_audio(audio)
                .fadein(0.4)
                .fadeout(0.4)
            )
            clips.append(clip)

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        final = concatenate_videoclips(clips, method="compose")

        if final.duration < min_duration:
            content = clips[1:]
            repeats = int(min_duration / max(final.duration - 3.0, 1)) + 1
            final = concatenate_videoclips(clips + content * repeats, method="compose")
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
