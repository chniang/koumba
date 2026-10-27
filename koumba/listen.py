"""Voice input helpers for Koumba.

The microphone is turned into a small in-memory WAV file and sent to the
configured OpenAI-compatible model as an ``audio_url`` content part.  Nothing
is written to disk and the audio is never kept after the request.
"""

from __future__ import annotations

import base64
import io
import json
import re
import wave
from functools import lru_cache
from typing import Any

import numpy as np

from .data import load_data

TARGET_SAMPLE_RATE = 16_000
MIN_SECONDS = 0.5
MAX_SECONDS = 30.0

PROMPT = (
    "Transcribe exactly what the speaker says. They speak Wolof from Dakar, "
    "often mixed with French. Write Wolof in the usual Latin spelling, keep "
    "place names in their usual French spelling. These are Dakar bus stop and "
    "place names that may appear: {vocabulary}. Answer only with JSON: "
    '{"text": ..., "language": ...}.'
)

_LANGUAGES = {
    "wolof": "Wolof",
    "french": "French",
    "français": "French",
    "francais": "French",
    "english": "English",
    "anglais": "English",
}


@lru_cache(maxsize=1)
def vocabulary() -> str:
    """Return all unique stop and place names, built once for the prompt."""

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
    return ", ".join(sorted(names, key=str.casefold))


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


def _message_content(response: Any) -> Any:
    if isinstance(response, dict):
        choices = response.get("choices") or []
        message = choices[0].get("message") if choices else None
        return message.get("content") if isinstance(message, dict) else None

    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    return getattr(message, "content", None)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _last_json_object(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content

    text = _content_text(content)
    if not text.strip():
        raise ValueError("The transcriber returned an empty answer.")

    # Muse Spark reasons before answering, so only the final JSON matters.
    text = re.sub(r"```[A-Za-z0-9_-]*", "", text).replace("```", "")
    decoder = json.JSONDecoder()
    objects: list[dict[str, Any]] = []
    for match in re.finditer(r"\{", text):
        try:
            value, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append(value)

    for value in reversed(objects):
        if isinstance(value.get("text"), str) and isinstance(
            value.get("language"), str
        ):
            return value
    raise ValueError("The transcriber did not return the expected JSON.")


def transcribe(client: Any, wav_bytes: bytes) -> dict[str, str]:
    """Send one WAV request to the provider and return text plus language."""

    encoded = base64.b64encode(wav_bytes).decode("ascii")
    prompt = PROMPT.replace("{vocabulary}", vocabulary())
    response = client.complete(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "audio_url",
                        "audio_url": {
                            "url": f"data:audio/wav;base64,{encoded}"
                        },
                    },
                ],
            }
        ],
    )

    value = _last_json_object(_message_content(response))
    text = str(value.get("text") or "").strip()
    language_text = str(value.get("language") or "").strip().rstrip(".")
    language = _LANGUAGES.get(language_text.casefold())
    if not text:
        raise ValueError("The transcription was empty. Please try again.")
    if language is None:
        raise ValueError("The transcription language was not recognized.")
    return {"text": text, "language": language}
