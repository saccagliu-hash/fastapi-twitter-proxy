import edge_tts
from pydub import AudioSegment

RECOMMENDED_VOICES: dict[str, dict[str, str]] = {
    "it": {"female": "it-IT-IsabellaNeural", "male": "it-IT-DiegoNeural"},
    "en": {"female": "en-US-JennyNeural", "male": "en-US-GuyNeural"},
    "es": {"female": "es-ES-ElviraNeural", "male": "es-ES-AlvaroNeural"},
    "fr": {"female": "fr-FR-DeniseNeural", "male": "fr-FR-HenriNeural"},
    "de": {"female": "de-DE-KatjaNeural", "male": "de-DE-ConradNeural"},
}


async def generate_tts(text: str, voice: str, output_path: str) -> float:
    """Generate TTS audio and return duration in seconds."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)
    audio = AudioSegment.from_mp3(output_path)
    return len(audio) / 1000.0


async def list_voices() -> list[dict]:
    return await edge_tts.list_voices()
