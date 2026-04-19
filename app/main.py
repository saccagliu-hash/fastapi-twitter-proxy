import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routers.video import OUTPUT_DIR, router

load_dotenv()

app = FastAPI(
    title="Faceless YouTube Video Generator",
    description=(
        "API per creare video faceless per YouTube con voce sintetica (TTS) e slideshow di immagini. "
        "Supporta oltre 300 voci in decine di lingue tramite Microsoft Edge TTS (gratuito). "
        "Opzionalmente usa immagini di stock da Pexels."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")
app.include_router(router)


@app.get("/", tags=["root"])
async def root():
    return {
        "service": "Faceless YouTube Video Generator",
        "docs": "/docs",
        "endpoints": {
            "generate_video": "POST /api/v1/videos/generate",
            "check_status": "GET /api/v1/videos/{video_id}/status",
            "download": "GET /api/v1/videos/{video_id}/download",
            "voices": "GET /api/v1/voices",
        },
    }
