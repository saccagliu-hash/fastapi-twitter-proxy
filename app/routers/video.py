import asyncio
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.models.schemas import (
    VideoRequest,
    VideoResponse,
    VideoStatus,
    VoiceInfo,
    VoicesResponse,
)
from app.services.tts import RECOMMENDED_VOICES, list_voices
from app.services.video_creator import create_faceless_video

router = APIRouter(prefix="/api/v1", tags=["videos"])

_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=2)
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    background_tasks.add_task(_run_generation, job_id, request)
    return VideoResponse(video_id=job_id, status=VideoStatus.pending)


async def _run_generation(job_id: str, request: VideoRequest):
    _jobs[job_id]["status"] = VideoStatus.processing
    result = await create_faceless_video(
        topic=request.topic,
        script=request.script,
        voice=request.voice,
        output_dir=OUTPUT_DIR,
        pexels_api_key=request.pexels_api_key,
        min_duration=request.min_duration,
        video_id=job_id,
    )
    if result["status"] == "completed":
        _jobs[job_id].update(
            {
                "status": VideoStatus.completed,
                "output_path": result["output_path"],
                "duration": result["duration"],
                "download_url": f"/api/v1/videos/{job_id}/download",
            }
        )
    else:
        _jobs[job_id].update({"status": VideoStatus.failed, "error": result.get("error")})


@router.get("/videos/{video_id}/status", response_model=VideoResponse)
async def get_status(video_id: str):
    """Controlla lo stato di generazione del video."""
    job = _jobs.get(video_id)
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
    job = _jobs.get(video_id)
    if not job:
        raise HTTPException(status_code=404, detail="Video non trovato")
    if job["status"] != VideoStatus.completed:
        raise HTTPException(status_code=400, detail=f"Video in stato: {job['status']}")
    path = job.get("output_path", "")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File video non trovato")
    return FileResponse(path, media_type="video/mp4", filename=f"video_{video_id}.mp4")


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
