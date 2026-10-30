"""Voice helpers are tested with synthetic audio, fake HTTP, and a fake LLM."""

from __future__ import annotations

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


def _set_voice_env(monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_ASR_URL", "https://example.modal.run/")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "test-secret")


def _fake_asr_post(
    monkeypatch,
    text: str = "Maa ngi Sam, bëgg naa dem Malika",
    status_code: int = 200,
) -> list[dict]:
    calls: list[dict] = []

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return SimpleNamespace(
            status_code=status_code,
            json=lambda: {
                "text": text,
                "lang": "wol_Latn",
                "seconds": 1.0,
                "infer_seconds": 0.25,
                "model": "omniASR_CTC_1B",
            },
        )

    monkeypatch.setattr(listen.httpx, "post", post)
    return calls


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


def test_transcribe_posts_wav_with_the_shared_secret(monkeypatch) -> None:
    _set_voice_env(monkeypatch)
    monkeypatch.setattr(
        listen,
        "find_place_hints",
        lambda _raw: ["Sahm (from 'Sam')"],
    )
    calls = _fake_asr_post(monkeypatch)
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    result = transcribe(wav_bytes)

    assert len(calls) == 1
    call = calls[0]
    assert call["url"] == "https://example.modal.run/transcribe"
    assert call["params"] == {"lang": "wol_Latn"}
    assert call["headers"]["X-Koumba-Key"] == "test-secret"
    assert call["headers"]["Content-Type"] == "audio/wav"
    assert call["content"] == wav_bytes
    assert call["content"].startswith(b"RIFF")
    assert call["timeout"] == 120.0

    assert result["raw_text"] == "Maa ngi Sam, bëgg naa dem Malika"
    assert result["text"] == "Maa ngi Sam, bëgg naa dem Malika"
    assert result["language"] == "Wolof"
    assert result["hints"] == ["Sahm (from 'Sam')"]
    assert result["asr_lang"] == "wol_Latn"
    assert result["asr_model"] == "omniASR_CTC_1B"
    assert result["infer_seconds"] == 0.25
    assert result["models"] == {
        "transcription": "Omnilingual ASR on Modal",
    }
    assert "clean_text" not in result
    assert "places_fixed" not in result
    assert set(result["timings"]) == {"asr"}
    assert result["timings"]["asr"] >= 0


def test_transcribe_rejects_a_bad_shared_secret(monkeypatch) -> None:
    _set_voice_env(monkeypatch)
    _fake_asr_post(monkeypatch, status_code=401)
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    with pytest.raises(listen.VoiceServiceError, match="authentication"):
        transcribe(wav_bytes)


def test_transcribe_reports_missing_voice_configuration(monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_ASR_URL", "")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "")
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    with pytest.raises(listen.VoiceNotConfigured, match="Voice not configured"):
        transcribe(wav_bytes)


def test_place_hints_find_places_inside_merged_asr_words() -> None:
    hints = listen.find_place_hints("mangisam dama bagadam malika")

    joined = " ".join(hints)
    assert "Sahm (from 'mangisam')" in joined
    assert "Malika (from 'malika')" in joined


def test_place_hints_join_a_split_place_name() -> None:
    hints = listen.find_place_hints("me dina")

    assert hints == ["Médina (from 'me dina')"]


def test_place_hints_keep_an_exact_place_name() -> None:
    hints = listen.find_place_hints("malika")

    assert hints == ["Malika (from 'malika')"]
