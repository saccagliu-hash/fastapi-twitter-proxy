import logging
from typing import Optional

import requests
from gtts import gTTS
from pydub import AudioSegment

SUPPORTED_LANGUAGES: dict[str, str] = {
    "it": "Italiano",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "pt": "Português",
    "nl": "Nederlands",
    "pl": "Polski",
}

RECOMMENDED_VOICES: dict[str, dict[str, str]] = {
    lang: {"female": lang, "male": lang} for lang in SUPPORTED_LANGUAGES
}

ELEVENLABS_DEFAULT_VOICE = "EXAVITQu4vr4xnSDxMaL"  # Bella — multilingual, ottima in italiano


def _duration_mp3(path: str) -> float:
    audio = AudioSegment.from_mp3(path)
    return len(audio) / 1000.0


def _gtts(text: str, voice: str, output_path: str) -> float:
    lang = voice.split("-")[0].lower()
    gTTS(text=text, lang=lang, slow=False).save(output_path)
    return _duration_mp3(output_path)


def _elevenlabs(text: str, api_key: str, voice_id: str, output_path: str) -> float:
    # Chiavi sk_* usano Bearer token, le vecchie chiavi usano xi-api-key
    if api_key.startswith("sk_"):
        auth_headers = {"Authorization": f"Bearer {api_key}"}
    else:
        auth_headers = {"xi-api-key": api_key}

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        headers={**auth_headers, "Content-Type": "application/json"},
        json={
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {"stability": 0.45, "similarity_boost": 0.80},
        },
        timeout=30,
    )
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)
    return _duration_mp3(output_path)


def generate_tts(
    text: str,
    voice: str,
    output_path: str,
    elevenlabs_api_key: Optional[str] = None,
    elevenlabs_voice_id: str = ELEVENLABS_DEFAULT_VOICE,
) -> float:
    if elevenlabs_api_key:
        try:
            return _elevenlabs(text, elevenlabs_api_key, elevenlabs_voice_id, output_path)
        except Exception as exc:
            logging.warning("ElevenLabs fallito (%s), uso gTTS come fallback", exc)
    return _gtts(text, voice, output_path)


async def list_voices() -> list[dict]:
    return [
        {"Name": lang, "Locale": lang, "Gender": "Neutral", "DisplayName": name}
        for lang, name in SUPPORTED_LANGUAGES.items()
    ]
