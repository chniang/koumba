"""Send a short spoken answer to the Koumba TTS service on Modal.

The app sends text over HTTPS and receives WAV bytes in memory.  A missing
endpoint or a service failure must never prevent the text answer from being
shown in the chat.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

# A cold Oolel container can take well over a minute to wake up.  Modal then
# answers with a 303 redirect, so the client follows it within this budget.
TTS_TIMEOUT_SECONDS = 150.0
VOICE_REPLY_NOT_CONFIGURED = (
    "Voice reply not configured. Add KOUMBA_TTS_URL and KOUMBA_ASR_KEY to "
    "your environment or .env file."
)


class SpeakError(RuntimeError):
    """Base error that the app can show without leaking service details."""


class SpeakNotConfigured(SpeakError):
    """Raised when the Modal TTS URL or shared secret is missing."""


class SpeakServiceError(SpeakError):
    """Raised when Modal rejects a request or cannot be reached."""


def speak_config() -> tuple[str, str]:
    """Require the Modal TTS endpoint URL and shared secret."""

    load_dotenv()
    url = (os.environ.get("KOUMBA_TTS_URL") or "").strip().rstrip("/")
    key = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
    if not url or not key:
        raise SpeakNotConfigured(VOICE_REPLY_NOT_CONFIGURED)
    return url, key


def speak(text: str) -> bytes:
    """Return WAV bytes for one short spoken sentence."""

    sentence = (text or "").strip()
    if not sentence:
        raise SpeakError("The spoken reply is empty.")

    url, key = speak_config()
    try:
        response = httpx.post(
            f"{url}/speak",
            json={"text": sentence},
            headers={"X-Koumba-Key": key},
            timeout=TTS_TIMEOUT_SECONDS,
            follow_redirects=True,
        )
    except httpx.HTTPError as exc:
        raise SpeakServiceError(
            "Voice reply service is unavailable. The text answer is still "
            "available."
        ) from exc

    if response.status_code == 401:
        raise SpeakServiceError(
            "Voice reply authentication failed. Check KOUMBA_ASR_KEY."
        )
    if response.status_code >= 400:
        raise SpeakServiceError(
            f"Voice reply service returned HTTP {response.status_code}."
        )

    audio = response.content
    if not audio:
        raise SpeakServiceError("Voice reply service returned no audio.")
    return audio
