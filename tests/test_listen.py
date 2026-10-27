"""Voice helpers are tested with synthetic audio and a fake model client."""

from __future__ import annotations

import base64
import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import koumba.listen as listen
from koumba.listen import prepare_audio, transcribe


def _read_wav(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        return (
            wav_file.getnchannels(),
            wav_file.getsampwidth(),
            wav_file.getframerate(),
            wav_file.getnframes(),
            np.frombuffer(wav_file.readframes(wav_file.getnframes()), "<i2"),
        )


def _response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=[])
            )
        ]
    )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict] = []

    def complete(
        self,
        messages,
        tools=None,
        tool_choice="auto",
        temperature=0.2,
    ):
        self.calls.append(
            {
                "messages": messages,
                "tools": tools,
                "temperature": temperature,
            }
        )
        return _response(self.content)


def test_prepare_audio_mixes_stereo_to_mono() -> None:
    left = np.full(16_000, 1000, dtype=np.int16)
    right = np.full(16_000, 3000, dtype=np.int16)
    stereo = np.column_stack([left, right])

    channels, width, rate, frames, pcm = _read_wav(
        prepare_audio(16_000, stereo)
    )

    assert channels == 1
    assert width == 2
    assert rate == 16_000
    assert frames == 16_000
    assert np.median(pcm) == pytest.approx(2000, abs=1)


def test_prepare_audio_resamples_44100_to_16000() -> None:
    samples = np.zeros(44_100, dtype=np.int16)

    _channels, _width, rate, frames, _pcm = _read_wav(
        prepare_audio(44_100, samples)
    )

    assert rate == 16_000
    assert frames == 16_000


def test_prepare_audio_rejects_a_too_short_recording() -> None:
    with pytest.raises(ValueError, match="too short"):
        prepare_audio(44_100, np.zeros(100, dtype=np.int16))


def test_prepare_audio_trims_at_thirty_seconds() -> None:
    samples = np.zeros(44_100 * 31, dtype=np.int16)

    _channels, _width, _rate, frames, _pcm = _read_wav(
        prepare_audio(44_100, samples)
    )

    assert frames == 16_000 * 30


def test_transcribe_sends_one_audio_url_request(monkeypatch) -> None:
    monkeypatch.setattr(listen, "vocabulary", lambda: "Sahm, Malika")
    client = _FakeClient(
        '{"text": "Maa ngi Sahm, bëgg naa dem Malika", "language": "Wolof"}'
    )
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    result = transcribe(client, wav_bytes)

    assert result == {
        "text": "Maa ngi Sahm, bëgg naa dem Malika",
        "language": "Wolof",
    }
    assert len(client.calls) == 1
    content = client.calls[0]["messages"][0]["content"]
    assert content[0]["type"] == "text"
    assert "Sahm, Malika" in content[0]["text"]
    assert content[1]["type"] == "audio_url"
    url = content[1]["audio_url"]["url"]
    assert url.startswith("data:audio/wav;base64,")
    audio = base64.b64decode(url.split(",", 1)[1])
    assert audio.startswith(b"RIFF")


def test_transcribe_uses_the_last_json_object_after_reasoning(
    monkeypatch,
) -> None:
    monkeypatch.setattr(listen, "vocabulary", lambda: "Sahm, Malika")
    content = (
        "The speaker starts with a place.\n"
        "```json\n"
        '{"text": "wrong", "language": "French"}\n'
        "```\n"
        "After listening again, the final answer is:\n"
        "```json\n"
        '{"text": "Maa ngi Sahm, bëgg naa dem Malika", '
        '"language": "wolof"}\n'
        "```\n"
        "That is the final JSON."
    )
    client = _FakeClient(content)
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    assert transcribe(client, wav_bytes) == {
        "text": "Maa ngi Sahm, bëgg naa dem Malika",
        "language": "Wolof",
    }
