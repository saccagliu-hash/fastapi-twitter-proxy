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
    "ru": "Русский",
    "ja": "日本語",
    "zh": "中文",
    "ar": "العربية",
}

RECOMMENDED_VOICES: dict[str, dict[str, str]] = {
    lang: {"female": lang, "male": lang} for lang in SUPPORTED_LANGUAGES
}


def _parse_lang(voice: str) -> str:
    """Accepts 'it', 'it-IT', or legacy 'it-IT-IsabellaNeural' — returns 'it'."""
    return voice.split("-")[0].lower()


def generate_tts(text: str, voice: str, output_path: str) -> float:
    """Generate TTS audio via Google TTS and return duration in seconds."""
    lang = _parse_lang(voice)
    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(output_path)
    audio = AudioSegment.from_mp3(output_path)
    return len(audio) / 1000.0


async def list_voices() -> list[dict]:
    return [
        {"Name": lang, "Locale": lang, "Gender": "Neutral", "DisplayName": name}
        for lang, name in SUPPORTED_LANGUAGES.items()
    ]
