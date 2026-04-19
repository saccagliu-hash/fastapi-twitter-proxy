import os
import re
import shutil
import uuid
from typing import Optional

from moviepy.editor import AudioFileClip, ImageClip, concatenate_videoclips

from .image_service import create_slide
from .tts import generate_tts

MIN_SLIDE_DURATION = 3.5
WORDS_PER_SEGMENT = 22

_STOP_WORDS = {
    "il", "la", "lo", "le", "i", "gli", "un", "una", "uno", "e", "è",
    "in", "a", "di", "da", "con", "su", "per", "tra", "fra", "che",
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
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
        clips = []

        for i, segment in enumerate(segments):
            tts_path = os.path.join(work_dir, f"audio_{i}.mp3")
            tts_duration = await generate_tts(segment, voice, tts_path)
            clip_duration = max(tts_duration, MIN_SLIDE_DURATION)

            img_path = os.path.join(work_dir, f"slide_{i}.jpg")
            create_slide(
                text=segment,
                theme_idx=i,
                output_path=img_path,
                pexels_api_key=pexels_api_key,
                search_query=_keywords(segment, topic) if pexels_api_key else None,
            )

            audio = AudioFileClip(tts_path)
            clip = ImageClip(img_path).set_duration(clip_duration).set_audio(audio)
            clips.append(clip)

        if not clips:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        final = concatenate_videoclips(clips, method="compose")

        if final.duration < min_duration:
            repeats = int(min_duration / final.duration) + 1
            final = concatenate_videoclips(clips * repeats, method="compose")
            final = final.subclip(0, min_duration)

        output_path = os.path.join(output_dir, f"{video_id}.mp4")
        final.write_videofile(
            output_path,
            fps=24,
            codec="libx264",
            audio_codec="aac",
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
