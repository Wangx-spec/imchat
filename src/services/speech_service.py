from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx


logger = logging.getLogger("services.speech")


@dataclass(frozen=True)
class ElevenLabsConfig:
    api_key: str
    voice_id: str
    tts_model: str
    stt_model: str
    timeout_ms: int = 30000
    base_url: str = "https://api.elevenlabs.io/v1"


class ElevenLabsClient:
    def __init__(self, cfg: ElevenLabsConfig) -> None:
        self.cfg = cfg

    def text_to_speech(self, text: str) -> bytes:
        body = (text or "").strip()
        if not body:
            raise ValueError("text cannot be empty")

        url = f"{self.cfg.base_url.rstrip('/')}/text-to-speech/{self.cfg.voice_id}"
        headers = {
            "xi-api-key": self.cfg.api_key,
            "accept": "audio/mpeg",
            "content-type": "application/json",
        }
        payload = {
            "text": body,
            "model_id": self.cfg.tts_model,
        }
        try:
            with httpx.Client(timeout=self.cfg.timeout_ms / 1000) as client:
                resp = client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            return resp.content
        except Exception as exc:
            raise RuntimeError(f"elevenlabs_tts_failed: {exc}") from exc

    def speech_to_text(self, audio_bytes: bytes, mime_type: str | None = None) -> str:
        if not audio_bytes:
            raise ValueError("audio_bytes cannot be empty")

        url = f"{self.cfg.base_url.rstrip('/')}/speech-to-text"
        headers = {"xi-api-key": self.cfg.api_key}
        files = {
            "file": ("audio", audio_bytes, mime_type or "audio/mpeg"),
        }
        data = {"model_id": self.cfg.stt_model}
        try:
            with httpx.Client(timeout=self.cfg.timeout_ms / 1000) as client:
                resp = client.post(url, headers=headers, data=data, files=files)
            resp.raise_for_status()
            payload = resp.json()
            text = str(payload.get("text", "")).strip()
            if not text:
                raise RuntimeError("elevenlabs_stt_empty_text")
            return text
        except Exception as exc:
            raise RuntimeError(f"elevenlabs_stt_failed: {exc}") from exc


_client: ElevenLabsClient | None = None


def init(settings) -> None:
    global _client
    enabled = bool(getattr(settings, "elevenlabs_enabled", False))
    if not enabled:
        _client = None
        logger.info("[ELEVENLABS] disabled")
        return

    api_key = (getattr(settings, "elevenlabs_api_key", "") or "").strip()
    if not api_key:
        _client = None
        logger.warning("[ELEVENLABS] enabled but api key missing")
        return

    _client = ElevenLabsClient(
        ElevenLabsConfig(
            api_key=api_key,
            voice_id=str(getattr(settings, "elevenlabs_voice_id", "Rachel") or "Rachel"),
            tts_model=str(getattr(settings, "elevenlabs_tts_model", "eleven_multilingual_v2") or "eleven_multilingual_v2"),
            stt_model=str(getattr(settings, "elevenlabs_stt_model", "scribe_v1") or "scribe_v1"),
            timeout_ms=int(getattr(settings, "elevenlabs_timeout_ms", 30000) or 30000),
        )
    )
    logger.info("[ELEVENLABS] initialized")


def is_ready() -> bool:
    return _client is not None


def text_to_speech(text: str) -> bytes:
    if _client is None:
        raise RuntimeError("speech_service_not_initialized")
    return _client.text_to_speech(text)


def speech_to_text(audio_bytes: bytes, mime_type: str | None = None) -> str:
    if _client is None:
        raise RuntimeError("speech_service_not_initialized")
    return _client.speech_to_text(audio_bytes, mime_type=mime_type)
