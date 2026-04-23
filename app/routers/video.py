import asyncio
import json
import os
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    VideoRequest,
    VideoResponse,
    VideoStatus,
    VoiceInfo,
    VoicesResponse,
)
from app.services.image_service import test_pexels, _fetch_google
from app.services.tts import RECOMMENDED_VOICES, list_voices
from app.services.video_creator import create_faceless_video

router = APIRouter(prefix="/api/v1", tags=["videos"])

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_STATE_FILE = os.path.join(OUTPUT_DIR, "_jobs.json")
_jobs: dict[str, dict] = {}


def _load_state():
    if os.path.exists(_STATE_FILE):
        try:
            with open(_STATE_FILE) as f:
                data = json.load(f)
            # Job in processing/pending al riavvio = interrotti, segnalali come failed
            for job in data.values():
                if job.get("status") in (VideoStatus.processing, VideoStatus.pending,
                                         "processing", "pending"):
                    job["status"] = VideoStatus.failed
                    job["error"] = "Interrotto dal riavvio del server — riprova"
            _jobs.update(data)
        except Exception:
            pass


def _save_state():
    try:
        with open(_STATE_FILE, "w") as f:
            json.dump(_jobs, f)
    except Exception:
        pass


def _job_from_disk(video_id: str) -> dict | None:
    """Ricostruisce il job dal file MP4 su disco se mancante in memoria."""
    path = os.path.join(OUTPUT_DIR, f"{video_id}.mp4")
    if os.path.exists(path):
        return {
            "status": VideoStatus.completed,
            "output_path": path,
            "download_url": f"/api/v1/videos/{video_id}/download",
        }
    return None


_load_state()


@router.post("/videos/generate", response_model=VideoResponse, status_code=202)
async def generate_video(request: VideoRequest, background_tasks: BackgroundTasks):
    """
    Avvia la generazione di un video faceless YouTube.

    - Invia lo script e l'argomento del video
    - Ricevi un `video_id` per monitorare lo stato
    - Scarica il video completato tramite `/api/v1/videos/{video_id}/download`
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": VideoStatus.pending}
    _save_state()
    background_tasks.add_task(_run_generation, job_id, request)
    return VideoResponse(video_id=job_id, status=VideoStatus.pending)


async def _run_generation(job_id: str, request: VideoRequest):
    try:
        _jobs[job_id]["status"] = VideoStatus.processing
        _save_state()
        result = await create_faceless_video(
            topic=request.topic,
            script=request.script,
            voice=request.voice,
            output_dir=OUTPUT_DIR,
            pexels_api_key=request.pexels_api_key,
            unsplash_api_key=request.unsplash_api_key,
            google_api_key=request.google_api_key,
            google_cx=request.google_cx,
            image_keywords=request.image_keywords,
            elevenlabs_api_key=request.elevenlabs_api_key,
            elevenlabs_voice_id=request.elevenlabs_voice_id,
            min_duration=request.min_duration,
            video_id=job_id,
        )
        if result["status"] == "completed":
            _jobs[job_id].update({
                "status": VideoStatus.completed,
                "output_path": result["output_path"],
                "duration": result["duration"],
                "download_url": f"/api/v1/videos/{job_id}/download",
            })
        else:
            _jobs[job_id].update({"status": VideoStatus.failed, "error": result.get("error")})
    except Exception as exc:
        _jobs[job_id].update({"status": VideoStatus.failed, "error": str(exc)})
    finally:
        _save_state()


@router.get("/videos/{video_id}/status", response_model=VideoResponse)
async def get_status(video_id: str):
    """Controlla lo stato di generazione del video."""
    job = _jobs.get(video_id) or _job_from_disk(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video non trovato")
    return VideoResponse(
        video_id=video_id,
        status=job["status"],
        download_url=job.get("download_url"),
        duration=job.get("duration"),
        error=job.get("error"),
    )


@router.get("/videos/{video_id}/download")
async def download_video(video_id: str):
    """Scarica il video MP4 completato."""
    job = _jobs.get(video_id) or _job_from_disk(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video non trovato")
    if job["status"] != VideoStatus.completed:
        status_val = job["status"].value if hasattr(job["status"], "value") else job["status"]
        raise HTTPException(status_code=400, detail=f"Video in stato: {status_val}")
    path = job.get("output_path", "")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File video non trovato")
    return FileResponse(path, media_type="video/mp4", filename=f"video_{video_id}.mp4")


@router.get("/test-pexels")
async def test_pexels_connection(api_key: str):
    """Testa se la API key Pexels funziona correttamente."""
    result = await asyncio.to_thread(test_pexels, api_key)
    return result


@router.get("/test-google")
async def test_google_connection(api_key: str, cx: str, query: str = "distributori automatici italia"):
    """Testa Google Custom Search API scaricando una foto di prova."""
    import tempfile, os as _os

    def _check():
        tmp = tempfile.mktemp(suffix=".jpg")
        try:
            ok = _fetch_google(query, api_key, cx, tmp, 0)
            size = _os.path.getsize(tmp) if ok and _os.path.exists(tmp) else 0
            return {"ok": ok, "file_size_bytes": size, "query": query}
        finally:
            if _os.path.exists(tmp):
                _os.remove(tmp)

    return await asyncio.to_thread(_check)


@router.get("/test-elevenlabs")
async def test_elevenlabs_connection(api_key: str, voice_id: str = "EXAVITQu4vr4xnSDxMaL"):
    """Testa direttamente il TTS di ElevenLabs."""
    import requests as _req, tempfile, os as _os

    def _check():
        tmp = tempfile.mktemp(suffix=".mp3")
        try:
            resp = _req.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                headers={"xi-api-key": api_key, "Content-Type": "application/json"},
                json={"text": "Ciao, questo e un test.", "model_id": "eleven_multilingual_v2",
                      "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
                timeout=20,
            )
            if resp.ok:
                with open(tmp, "wb") as f:
                    f.write(resp.content)
                from pydub import AudioSegment
                dur = len(AudioSegment.from_mp3(tmp)) / 1000.0
                return {"ok": True, "duration_seconds": round(dur, 2)}
            else:
                return {"ok": False, "status": resp.status_code, "error": resp.text[:400]}
        finally:
            if _os.path.exists(tmp):
                _os.remove(tmp)

    return await asyncio.to_thread(_check)


@router.get("/voices", response_model=VoicesResponse)
async def get_voices():
    """Lista tutte le voci TTS disponibili con le consigliate per lingua."""
    all_voices = await list_voices()
    voices = [
        VoiceInfo(name=v["Name"], language=v["Locale"], gender=v["Gender"])
        for v in all_voices
    ]
    recommended = {lang: voices_dict["female"] for lang, voices_dict in RECOMMENDED_VOICES.items()}
    return VoicesResponse(voices=voices, recommended=recommended)
