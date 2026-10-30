"""Voice input helpers for Koumba.

The microphone is prepared as a small in-memory WAV and sent to Meta's
Omnilingual ASR service on Modal.  The raw transcript goes straight to the
agent, along with local place hints, so there is no second model on the voice
path.  Nothing is written to permanent storage.
"""

from __future__ import annotations

import io
import os
import re
import time
import wave
from functools import lru_cache
from typing import Any

import httpx
import numpy as np
from dotenv import load_dotenv
from rapidfuzz import fuzz, process

from .data import load_data
from .places import normalize_name

TARGET_SAMPLE_RATE = 16_000
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0
DEFAULT_ASR_LANG = "wol_Latn"

ASR_MODEL_LABEL = "Omnilingual ASR on Modal"
LISTENING_STATUS = "Koumba is listening..."

PLACE_HINT_LIMIT = 6
PLACE_HINT_MIN_SCORE = 80.0
PLACE_HINT_MIN_RANK = 90.0

# Short words that are common in a Wolof bus request.  They are still
# checked by the matcher, but they should not become place hints on their own.
_HINT_STOPWORDS = {
    "a",
    "ak",
    "au",
    "aux",
    "ba",
    "begg",
    "begga",
    "bi",
    "bu",
    "ci",
    "dafay",
    "dama",
    "de",
    "def",
    "dem",
    "des",
    "dina",
    "du",
    "et",
    "fi",
    "foo",
    "je",
    "la",
    "le",
    "leegi",
    "les",
    "maa",
    "man",
    "mon",
    "naa",
    "naka",
    "nekk",
    "nga",
    "ngi",
    "nun",
    "ou",
    "pour",
    "suis",
    "ta",
    "tes",
    "ton",
    "un",
    "une",
    "vais",
    "vers",
    "wacc",
    "walla",
    "waaw",
}

VOICE_NOT_CONFIGURED = (
    "Voice not configured. Add KOUMBA_ASR_URL and KOUMBA_ASR_KEY to your "
    "environment or .env file."
)

_LANGUAGES = {
    "wolof": "Wolof",
    "wol": "Wolof",
    "wol_latn": "Wolof",
    "french": "French",
    "français": "French",
    "francais": "French",
    "fra": "French",
    "fra_latn": "French",
    "english": "English",
    "anglais": "English",
    "eng": "English",
    "eng_latn": "English",
}


class VoiceError(RuntimeError):
    """Base error that the app can show without leaking service details."""


class VoiceNotConfigured(VoiceError):
    """Raised when the Modal URL or shared secret is missing."""


class VoiceServiceError(VoiceError):
    """Raised when Modal rejects a request or cannot be reached."""


@lru_cache(maxsize=1)
def _raw_place_names() -> set[str]:
    data = load_data()
    names: set[str] = set()
    for frame, column in (
        (data.arrets, "arret"),
        (data.lieux, "lieu"),
        (data.itineraires, "arret"),
        (data.lignes, "terminus_a"),
        (data.lignes, "terminus_b"),
    ):
        if column not in frame.columns:
            continue
        for value in frame[column].dropna():
            name = str(value).strip()
            if name:
                names.add(name)
    return names


def _display_rank(name: str) -> tuple[bool, bool, int, str]:
    """Prefer normal title case over all-lower or all-caps duplicates."""

    return (name.islower(), name.isupper(), len(name), name.casefold())


@lru_cache(maxsize=1)
def place_names() -> tuple[tuple[str, str], ...]:
    """Return one display spelling for each normalized stop or place name."""

    chosen: dict[str, str] = {}
    for name in _raw_place_names():
        normalized = normalize_name(name)
        if not normalized:
            continue
        current = chosen.get(normalized)
        if current is None or _display_rank(name) < _display_rank(current):
            chosen[normalized] = name
    return tuple(
        (chosen[normalized], normalized) for normalized in sorted(chosen)
    )


def find_place_hints(
    raw_text: str,
    limit: int = PLACE_HINT_LIMIT,
) -> list[str]:
    """Find likely place names in the raw ASR words before calling the LLM.

    Every word and every pair or triple of adjacent words is compared with
    the official place names.  The score uses ``partial_ratio`` so a place
    can be found inside a merged word such as ``mangisam`` -> ``Sahm``.
    """

    text = (raw_text or "").strip()
    tokens = re.findall(r"[^\W_]+", text, re.UNICODE)
    if not tokens:
        return []

    entries = place_names()
    if not entries:
        return []

    choices = [normalized for _display, normalized in entries]
    display_by_norm = {
        normalized: display for display, normalized in entries
    }
    found: dict[str, tuple[float, float, str]] = {}

    for size in (1, 2, 3):
        for start in range(len(tokens) - size + 1):
            heard = " ".join(tokens[start : start + size])
            query = normalize_name(heard)
            if len(query) < 3 or query in _HINT_STOPWORDS:
                continue

            exact = display_by_norm.get(query)
            best = (150.0, 100.0, query) if exact else None
            if best is None:
                for normalized, partial, _index in process.extract(
                    query,
                    choices,
                    scorer=fuzz.partial_ratio,
                    limit=20,
                ):
                    if partial < PLACE_HINT_MIN_SCORE:
                        break
                    full = fuzz.ratio(query, normalized)
                    # A three-letter fragment is only believable when the
                    # whole word is close, not merely a substring.
                    if len(query) <= 3 and full < 75.0:
                        continue
                    gap = abs(len(query) - len(normalized))
                    rank = partial + (full / 2.0) - (1.5 * gap)
                    if rank < PLACE_HINT_MIN_RANK:
                        continue
                    if best is None or rank > best[0]:
                        best = (rank, partial, normalized)

            if best is None:
                continue
            rank, partial, normalized = best
            if normalized not in found or rank > found[normalized][0]:
                found[normalized] = (rank, partial, heard)

    ranked = sorted(found.items(), key=lambda item: (-item[1][0], item[0]))
    return [
        f"{display_by_norm[normalized]} (from {heard!r})"
        for normalized, (_rank, _partial, heard) in ranked[:limit]
    ]


def _to_float(samples: Any) -> np.ndarray:
    array = np.asarray(samples)
    if array.ndim == 0:
        raise ValueError("Audio samples must be a waveform, not a single value.")
    if array.ndim > 2:
        raise ValueError("Audio must be mono or stereo.")

    if np.issubdtype(array.dtype, np.unsignedinteger):
        # Unsigned PCM, such as 8-bit WAV, is centred around the midpoint.
        info = np.iinfo(array.dtype)
        midpoint = (info.max + 1) / 2.0
        waveform = (array.astype(np.float64) - midpoint) / midpoint
    elif np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = max(abs(info.min), info.max)
        waveform = array.astype(np.float64) / scale
    else:
        waveform = array.astype(np.float64)

    return np.nan_to_num(waveform, nan=0.0, posinf=1.0, neginf=-1.0)


def _to_mono(samples: Any) -> np.ndarray:
    waveform = _to_float(samples)
    if waveform.ndim == 1:
        return waveform

    rows, columns = waveform.shape
    if rows <= 2 and columns > 2:
        # Some recorders return (channels, samples).
        waveform = waveform.mean(axis=0)
    elif columns <= 2:
        # Gradio and most recorders use (samples, channels).
        waveform = waveform.mean(axis=1)
    else:
        raise ValueError("Audio must be mono or stereo.")
    return waveform


def _resample(waveform: np.ndarray, sample_rate: float) -> np.ndarray:
    target_length = max(
        1,
        int(round(len(waveform) * TARGET_SAMPLE_RATE / sample_rate)),
    )
    if int(round(sample_rate)) == TARGET_SAMPLE_RATE:
        return waveform

    # Linear interpolation is enough for speech here and avoids ffmpeg.
    source_positions = np.arange(target_length) * (
        sample_rate / TARGET_SAMPLE_RATE
    )
    return np.interp(
        source_positions,
        np.arange(len(waveform)),
        waveform,
    )


def prepare_audio(sample_rate: float, samples: Any) -> bytes:
    """Convert microphone samples to a mono 16 kHz, 16-bit PCM WAV."""

    try:
        rate = float(sample_rate)
    except (TypeError, ValueError) as exc:
        raise ValueError("The recording has no usable sample rate.") from exc
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("The recording has no usable sample rate.")

    waveform = _to_mono(samples)
    if len(waveform) / rate < MIN_SECONDS:
        raise ValueError(
            "The recording is too short. Please speak for at least 0.5 "
            "seconds."
        )

    waveform = _resample(waveform, rate)
    max_frames = int(round(MAX_SECONDS * TARGET_SAMPLE_RATE))
    waveform = np.clip(waveform[:max_frames], -1.0, 1.0)

    pcm = np.rint(waveform * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(TARGET_SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


def voice_config() -> tuple[str, str]:
    """Require the Modal endpoint URL and shared secret."""

    load_dotenv()
    url = (os.environ.get("KOUMBA_ASR_URL") or "").strip().rstrip("/")
    key = (os.environ.get("KOUMBA_ASR_KEY") or "").strip()
    if not url or not key:
        raise VoiceNotConfigured(VOICE_NOT_CONFIGURED)
    return url, key


def _asr_transcribe(wav_bytes: bytes, lang: str) -> dict[str, Any]:
    url, key = voice_config()
    try:
        response = httpx.post(
            f"{url}/transcribe",
            params={"lang": lang},
            content=wav_bytes,
            headers={
                "X-Koumba-Key": key,
                "Content-Type": "audio/wav",
            },
            timeout=120.0,
        )
    except httpx.HTTPError as exc:
        raise VoiceServiceError(
            "Voice service is unavailable. Please try again or type your "
            "message."
        ) from exc

    if response.status_code == 401:
        raise VoiceServiceError(
            "Voice authentication failed. Check KOUMBA_ASR_KEY."
        )
    if response.status_code >= 400:
        raise VoiceServiceError(
            f"Voice service returned HTTP {response.status_code}."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise VoiceServiceError(
            "Voice service returned an invalid response."
        ) from exc
    if not isinstance(payload, dict):
        raise VoiceServiceError("Voice service returned an invalid response.")

    text = str(payload.get("text") or "").strip()
    if not text:
        raise VoiceServiceError(
            "The voice service heard no words. Please try again or type your "
            "message."
        )

    return {
        "text": text,
        "lang": str(payload.get("lang") or lang),
        "seconds": payload.get("seconds"),
        "infer_seconds": payload.get("infer_seconds"),
        "model": str(payload.get("model") or "omniASR_CTC_1B"),
    }


def transcribe(
    wav_bytes: bytes,
    lang: str = DEFAULT_ASR_LANG,
) -> dict[str, Any]:
    """Run Modal ASR and keep its raw words for the agent.

    The local place hints are cheap and give the agent the likely official
    spelling without adding another model call to the voice path.
    """

    asr_started = time.perf_counter()
    raw = _asr_transcribe(wav_bytes, lang)
    asr_seconds = time.perf_counter() - asr_started
    text = str(raw["text"]).strip()
    return {
        "text": text,
        "raw_text": text,
        "language": _LANGUAGES.get(str(raw["lang"]).casefold(), "Unknown"),
        "asr_lang": raw["lang"],
        "hints": find_place_hints(text),
        "seconds": raw.get("seconds"),
        "infer_seconds": raw.get("infer_seconds"),
        "asr_model": raw.get("model"),
        "models": {"transcription": ASR_MODEL_LABEL},
        "timings": {"asr": round(asr_seconds, 3)},
    }
