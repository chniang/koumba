"""TTS client tests use fake HTTP and never call the Modal service."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest

import koumba.speak as speak_module
from koumba.speak import (
    VOICE_REPLY_NOT_CONFIGURED,
    SpeakError,
    SpeakNotConfigured,
    SpeakServiceError,
    speak,
)


def _set_tts_env(monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_TTS_URL", "https://example.modal.run/")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")
    monkeypatch.setattr(
        speak_module,
        "load_dotenv",
        lambda *args, **kwargs: None,
    )


def _fake_post(
    monkeypatch,
    *,
    status_code: int = 200,
    content: bytes = b"RIFF-test",
    error: Exception | None = None,
) -> list[dict]:
    calls: list[dict] = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        if error is not None:
            raise error
        return SimpleNamespace(status_code=status_code, content=content)

    monkeypatch.setattr(speak_module.httpx, "post", post)
    return calls


def test_speak_posts_text_with_key_and_timeout(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    calls = _fake_post(monkeypatch)

    audio = speak("  Maa ngi Sahm.  ")

    assert audio == b"RIFF-test"
    assert calls == [
        {
            "url": "https://example.modal.run/speak",
            "json": {"text": "Maa ngi Sahm."},
            "headers": {"X-Koumba-Key": "test-secret"},
            "timeout": 60.0,
        }
    ]


def test_speak_requires_the_tts_url(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    monkeypatch.setenv("KOUMBA_TTS_URL", "")
    calls = _fake_post(monkeypatch)

    with pytest.raises(SpeakNotConfigured) as error:
        speak("Maa ngi Sahm.")

    assert str(error.value) == VOICE_REPLY_NOT_CONFIGURED
    assert calls == []


def test_speak_requires_the_shared_key(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    monkeypatch.setenv("KOUMBA_ASR_KEY", "")
    calls = _fake_post(monkeypatch)

    with pytest.raises(SpeakNotConfigured):
        speak("Maa ngi Sahm.")

    assert calls == []


def test_speak_rejects_empty_text(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    calls = _fake_post(monkeypatch)

    with pytest.raises(SpeakError, match="empty"):
        speak("  ")

    assert calls == []


def test_speak_reports_authentication_failure(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    _fake_post(monkeypatch, status_code=401)

    with pytest.raises(SpeakServiceError, match="authentication"):
        speak("Maa ngi Sahm.")


def test_speak_reports_http_failure(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    _fake_post(monkeypatch, status_code=500)

    with pytest.raises(SpeakServiceError, match="HTTP 500"):
        speak("Maa ngi Sahm.")


def test_speak_reports_network_failure(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    _fake_post(
        monkeypatch,
        error=httpx.ConnectError("connection refused"),
    )

    with pytest.raises(SpeakServiceError, match="unavailable"):
        speak("Maa ngi Sahm.")


def test_speak_reports_empty_audio(monkeypatch) -> None:
    _set_tts_env(monkeypatch)
    _fake_post(monkeypatch, content=b"")

    with pytest.raises(SpeakServiceError, match="no audio"):
        speak("Maa ngi Sahm.")
