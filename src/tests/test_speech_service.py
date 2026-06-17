from types import SimpleNamespace

import pytest

from services import speech_service


def test_speech_service_init_disabled():
    settings = SimpleNamespace(elevenlabs_enabled=False)
    speech_service.init(settings)
    assert speech_service.is_ready() is False


def test_speech_service_init_missing_key():
    settings = SimpleNamespace(
        elevenlabs_enabled=True,
        elevenlabs_api_key="",
    )
    speech_service.init(settings)
    assert speech_service.is_ready() is False


def test_speech_service_requires_init():
    speech_service._client = None
    with pytest.raises(RuntimeError):
        speech_service.text_to_speech("hello")
    with pytest.raises(RuntimeError):
        speech_service.speech_to_text(b"audio")
