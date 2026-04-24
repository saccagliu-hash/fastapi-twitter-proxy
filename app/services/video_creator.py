import asyncio
import functools
import os
import re
import shutil
import uuid
from typing import Optional

import PIL.Image
# moviepy 1.0.3 usa ANTIALIAS internamente, rimosso in Pillow 10+
if not hasattr(PIL.Image, 'ANTIALIAS'):
    PIL.Image.ANTIALIAS = PIL.Image.LANCZOS

import requests as _requests
import numpy as np
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    VideoFileClip,
    concatenate_audioclips,
    concatenate_videoclips,
)
from PIL import Image as PILImage

from .image_service import (
    fetch_background, make_intro_card, bake_intro_card, bake_subtitle,
    make_subtitle_overlay, _fetch_pexels_video,
)
from .tts import generate_tts

MIN_SLIDE_DURATION = 3.0
WORDS_PER_SEGMENT = 18
MIN_SEGMENT_WORDS = 5

INTRO_DURATION = 1.5
SLIDE_FADE = 0.4


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


_STOP_WORDS = {
    'il', 'la', 'lo', 'i', 'gli', 'le', 'un', 'una', 'uno', 'di', 'da', 'in',
    'con', 'su', 'per', 'tra', 'fra', 'che', 'non', 'e', 'a', 'o', 'ma', 'si',
    'del', 'della', 'dei', 'degli', 'delle', 'al', 'alla', 'ai', 'agli', 'alle',
    'nel', 'nella', 'nei', 'negli', 'nelle', 'sul', 'sulla', 'sui', 'sugli',
    'dal', 'dalla', 'dai', 'dagli', 'dalle', 'sono', 'siamo', 'questo', 'questa',
    'questi', 'queste', 'anche', 'come', 'più', 'molto', 'tutti', 'tutto', 'ogni',
    'quando', 'dove', 'chi', 'cosa', 'quale', 'quali', 'può', 'deve', 'hanno',
    'viene', 'essere', 'avere', 'fare', 'the', 'and', 'or', 'for', 'with', 'from',
    'that', 'this', 'are', 'was', 'were', 'will', 'have', 'has', 'been',
}


def _parse_keyword_sets(image_keywords: str) -> list[str]:
    """Split comma-separated keyword sets, e.g. 'vending machine, coffee office, snack drink'."""
    sets = [k.strip() for k in image_keywords.split(",") if k.strip()]
    return sets if sets else [image_keywords]


def _slide_query(base_keywords: str, segment: str, slide_idx: int) -> str:
    """Pick a keyword set (cycling through the list) then add a relevant word from the segment."""
    keyword_sets = _parse_keyword_sets(base_keywords)
    base_str = keyword_sets[slide_idx % len(keyword_sets)]
    base = [w for w in base_str.split() if w.isalpha()][:4]
    base_lower = {w.lower() for w in base}
    seg_words = [
        w.strip(".,;:!?").lower() for w in segment.split()
        if w.isalpha() and w.lower() not in _STOP_WORDS and len(w) > 4
    ]
    for w in seg_words:
        if w not in base_lower:
            base.append(w)
            break
    return " ".join(base) if base else "vending machine"


def _silence(duration: float, fps: int = 44100) -> AudioArrayClip:
    samples = np.zeros((max(1, int(fps * duration)), 2), dtype=np.float32)
    return AudioArrayClip(samples, fps=fps)


def _load_frame(path: str) -> np.ndarray:
    with PILImage.open(path) as img:
        return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def create_faceless_video(
    topic: str,
    script: str,
    voice: str,
    output_dir: str,
    pexels_api_key: Optional[str] = None,
    unsplash_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    google_cx: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    image_keywords: Optional[str] = None,
    use_video_clips: bool = False,
    background_music_url: Optional[str] = None,
    music_volume: float = 0.07,
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

        # --- Step 1: TTS unico per tutto lo script → voce fluente e naturale ---
        narration_path = os.path.join(work_dir, "narration.mp3")
        narration_dur = await asyncio.to_thread(
            generate_tts, script, voice, narration_path,
            elevenlabs_api_key, elevenlabs_voice_id,
        )

        # Durate slide proporzionali ai caratteri (più accurato per testi con numeri)
        seg_chars = [len(seg) for seg in segments]
        total_chars = max(sum(seg_chars), 1)
        clip_durations = [
            max(narration_dur * c / total_chars, MIN_SLIDE_DURATION)
            for c in seg_chars
        ]

        # Audio: silenzio intro + narrazione + eventuale padding
        content_dur = sum(clip_durations)
        voice_parts: list = [_silence(INTRO_DURATION), AudioFileClip(narration_path)]
        gap = content_dur - narration_dur
        if gap > 0.1:
            voice_parts.append(_silence(gap))
        voice_audio = concatenate_audioclips(voice_parts)

        # --- Step 2: intro card ---
        intro_path = os.path.join(work_dir, "intro.jpg")
        intro_bg_path = os.path.join(work_dir, "intro_bg.jpg")
        base_kw = image_keywords or topic
        fetched_intro = False
        has_any_img_key = openai_api_key or google_api_key or unsplash_api_key or pexels_api_key
        if has_any_img_key:
            await asyncio.to_thread(
                fetch_background, 99, intro_bg_path, pexels_api_key,
                base_kw, unsplash_api_key, google_api_key, google_cx,
                openai_api_key, topic,
            )
            if os.path.exists(intro_bg_path):
                await asyncio.to_thread(bake_intro_card, intro_bg_path, topic, intro_path)
                fetched_intro = True
        if not fetched_intro:
            PILImage.fromarray(make_intro_card(topic)).save(intro_path, "JPEG", quality=92)

        intro_fn = functools.partial(_load_frame, intro_path)
        intro_clip = VideoClip(lambda t: intro_fn(), duration=INTRO_DURATION).set_fps(24)
        clips = [intro_clip]

        # --- Step 3: slide con sottotitoli gialli statici ---
        for i, segment in enumerate(segments):
            base_kw = image_keywords or topic
            slide_q = _slide_query(base_kw, segment, i) if has_any_img_key else None
            dur = clip_durations[i]

            # Prova video clip Pexels se richiesto
            used_video = False
            if use_video_clips and pexels_api_key and slide_q:
                video_path = os.path.join(work_dir, f"clip_{i}.mp4")
                used_video = await asyncio.to_thread(
                    _fetch_pexels_video, slide_q, pexels_api_key, video_path, i
                )

            if used_video:
                base_clip = VideoFileClip(video_path).resize((1280, 720)).set_fps(24)
                if base_clip.duration < dur:
                    loops = int(dur / base_clip.duration) + 2
                    base_clip = concatenate_videoclips([base_clip] * loops).subclip(0, dur)
                else:
                    base_clip = base_clip.subclip(0, dur)
                rgb_arr, mask_arr = await asyncio.to_thread(make_subtitle_overlay, segment)
                rgb_clip = ImageClip(rgb_arr).set_duration(dur)
                mask_clip = ImageClip(mask_arr, ismask=True).set_duration(dur)
                sub_clip = rgb_clip.set_mask(mask_clip)
                slide_clip = CompositeVideoClip([base_clip, sub_clip]).set_fps(24).fadein(SLIDE_FADE)
            else:
                bg_path = os.path.join(work_dir, f"bg_{i}.jpg")
                await asyncio.to_thread(
                    fetch_background, i, bg_path, pexels_api_key,
                    slide_q, unsplash_api_key, google_api_key, google_cx,
                    openai_api_key, segment,
                )
                slide_path = os.path.join(work_dir, f"slide_{i}.jpg")
                await asyncio.to_thread(bake_subtitle, bg_path, segment, slide_path)
                frame_fn = functools.partial(_load_frame, slide_path)
                slide_clip = (
                    VideoClip(lambda t, f=frame_fn: f(), duration=dur)
                    .set_fps(24)
                    .fadein(SLIDE_FADE)
                )
            clips.append(slide_clip)

        if len(clips) <= 1:
            return {"video_id": video_id, "status": "failed", "error": "Nessun contenuto generato"}

        # --- Step 4: concatena + audio ---
        final = concatenate_videoclips(clips, method="compose")
        final = final.set_audio(voice_audio.set_duration(final.duration))

        # --- Step 5: loop se troppo corto ---
        if final.duration < min_duration:
            content = clips[1:]
            repeats = int(min_duration / max(final.duration - INTRO_DURATION, 1)) + 1
            ext_narration = AudioFileClip(narration_path)
            ext_voice = concatenate_audioclips(
                [_silence(INTRO_DURATION)] + [ext_narration] * (repeats + 1)
            )
            final = concatenate_videoclips(clips + content * repeats, method="compose")
            final = final.subclip(0, min_duration)
            final = final.set_audio(ext_voice.set_duration(min_duration))

        # --- Step 6: musica di sottofondo ---
        if background_music_url:
            try:
                music_path = os.path.join(work_dir, "music.mp3")
                music_resp = await asyncio.to_thread(
                    lambda: _requests.get(background_music_url, timeout=30)
                )
                music_resp.raise_for_status()
                with open(music_path, "wb") as f:
                    f.write(music_resp.content)
                music = AudioFileClip(music_path)
                loops = int(final.duration / music.duration) + 2
                music_loop = concatenate_audioclips([music] * loops).subclip(0, final.duration)
                music_quiet = music_loop.volumex(music_volume)
                from moviepy.editor import CompositeAudioClip
                final = final.set_audio(CompositeAudioClip([final.audio, music_quiet]))
            except Exception as exc:
                pass  # musica opzionale, non blocca la generazione

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
