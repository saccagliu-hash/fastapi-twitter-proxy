from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class VideoStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class VideoRequest(BaseModel):
    topic: str = Field(..., description="Titolo/argomento del video")
    script: str = Field(
        ...,
        description="Testo da narrare (min ~125 parole per 50+ secondi)",
        min_length=50,
    )
    voice: str = Field(
        default="it",
        description="Lingua della voce: 'it', 'en', 'es', 'fr', 'de' ... (vedi GET /api/v1/voices)",
    )
    pexels_api_key: Optional[str] = Field(
        default=None,
        description="API key Pexels per immagini di stock (opzionale)",
    )
    image_keywords: Optional[str] = Field(
        default=None,
        description="Parole chiave IN INGLESE per cercare le immagini su Pexels (es: 'vending machine business'). Se non specificato usa il topic.",
    )
    min_duration: float = Field(
        default=50.0,
        ge=10.0,
        le=600.0,
        description="Durata minima video in secondi",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "topic": "Intelligenza Artificiale",
                "script": (
                    "L'intelligenza artificiale sta rivoluzionando il mondo moderno. "
                    "Ogni giorno nuovi strumenti vengono sviluppati per semplificare la vita "
                    "delle persone. Dalle diagnosi mediche alla guida autonoma, l'IA è "
                    "ovunque. Le aziende tecnologiche investono miliardi di dollari nella "
                    "ricerca. I modelli linguistici come GPT e Claude possono generare testi "
                    "in modo quasi umano. Questo apre possibilità straordinarie ma anche "
                    "sfide etiche importanti. Come garantire che l'IA sia usata in modo "
                    "responsabile? Come proteggere i posti di lavoro? Queste sono le domande "
                    "che la società deve affrontare oggi."
                ),
                "voice": "it",
                "pexels_api_key": None,
                "min_duration": 50.0,
            }
        }
    }


class VideoResponse(BaseModel):
    video_id: str
    status: VideoStatus
    download_url: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None


class VoiceInfo(BaseModel):
    name: str
    language: str
    gender: str


class VoicesResponse(BaseModel):
    voices: list[VoiceInfo]
    recommended: dict[str, str]
