"""Voice helpers are tested with synthetic audio, fake HTTP, and a fake LLM."""

from __future__ import annotations

import io
import wave
from types import SimpleNamespace

import numpy as np
import pytest

import koumba.listen as listen
from koumba.listen import cleanup_transcript, prepare_audio, transcribe


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
    monkeypatch.setattr(listen, "vocabulary", lambda: "Sahm, Malika")
    monkeypatch.setattr(
        listen,
        "find_place_hints",
        lambda _raw: ["Sahm (from 'Sam')"],
    )
    calls = _fake_asr_post(monkeypatch)
    client = _FakeClient(
        "Reasoning first.\n"
        "```json\n"
        '{"text": "Maa ngi Sahm, bëgg naa dem Malika", '
        '"language": "Wolof", "places_fixed": ["Sahm", "Malika"]}\n'
        "```"
    )
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    result = transcribe(client, wav_bytes)

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
    assert result["text"] == "Maa ngi Sahm, bëgg naa dem Malika"
    assert result["clean_text"] == "Maa ngi Sahm, bëgg naa dem Malika"
    assert result["language"] == "Wolof"
    assert result["places_fixed"] == ["Sahm", "Malika"]
    assert result["hints"] == ["Sahm (from 'Sam')"]
    assert result["asr_lang"] == "wol_Latn"
    assert result["asr_model"] == "omniASR_CTC_1B"
    assert result["infer_seconds"] == 0.25
    assert result["models"] == {
        "transcription": "Omnilingual ASR on Modal",
        "cleanup": "Muse Spark Open on Baseten",
    }
    assert set(result["timings"]) == {"asr", "cleanup"}
    assert result["timings"]["asr"] >= 0
    assert result["timings"]["cleanup"] >= 0

    assert len(client.calls) == 1
    prompt = client.calls[0]["messages"][0]["content"]
    assert isinstance(prompt, str)
    assert "Maa ngi Sam, bëgg naa dem Malika" in prompt
    assert "Sahm, Malika" in prompt
    assert "Sahm (from 'Sam')" in prompt
    assert "wol_Latn" in prompt
    assert "audio" not in prompt.lower()


def test_transcribe_rejects_a_bad_shared_secret(monkeypatch) -> None:
    _set_voice_env(monkeypatch)
    _fake_asr_post(monkeypatch, status_code=401)
    client = _FakeClient("unused")
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    with pytest.raises(listen.VoiceServiceError, match="authentication"):
        transcribe(client, wav_bytes)

    assert client.calls == []


def test_transcribe_reports_missing_voice_configuration(monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_ASR_URL", "")
    monkeypatch.setenv("KOUMBA_ASR_KEY", "")
    client = _FakeClient("unused")
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    with pytest.raises(listen.VoiceNotConfigured, match="Voice not configured"):
        transcribe(client, wav_bytes)


def test_transcribe_keeps_raw_text_when_cleanup_fails(monkeypatch) -> None:
    _set_voice_env(monkeypatch)
    _fake_asr_post(monkeypatch, text="Maa ngi Sam")
    wav_bytes = prepare_audio(
        16_000, np.zeros(16_000, dtype=np.int16)
    )

    with pytest.raises(
        listen.VoiceServiceError,
        match="Raw transcript: Maa ngi Sam",
    ):
        transcribe(_FakeClient("not json"), wav_bytes)


def test_cleanup_uses_the_last_json_after_reasoning(monkeypatch) -> None:
    monkeypatch.setattr(listen, "vocabulary", lambda: "Sahm, Malika")
    monkeypatch.setattr(
        listen,
        "find_place_hints",
        lambda _raw: ["Sahm (from 'Sam')"],
    )
    content = (
        "The speaker starts with a place.\n"
        "```json\n"
        '{"text": "wrong", "language": "French", "places_fixed": []}\n'
        "```\n"
        "After checking the bus names, the final answer is:\n"
        "```json\n"
        '{"text": "Maa ngi Sahm, bëgg naa dem Malika", '
        '"language": "wolof", "places_fixed": ["Sahm"]}\n'
        "```\n"
        "That is the final JSON."
    )
    client = _FakeClient(content)

    result = cleanup_transcript(
        client,
        "Maa ngi Sam, bëgg naa dem Malika",
    )

    assert result["raw_text"] == "Maa ngi Sam, bëgg naa dem Malika"
    assert result["text"] == "Maa ngi Sahm, bëgg naa dem Malika"
    assert result["clean_text"] == "Maa ngi Sahm, bëgg naa dem Malika"
    assert result["language"] == "Wolof"
    assert result["places_fixed"] == ["Sahm"]
    assert result["hints"] == ["Sahm (from 'Sam')"]
    prompt = client.calls[0]["messages"][0]["content"]
    assert "Maa ngi Sam, bëgg naa dem Malika" in prompt
    assert "Sahm, Malika" in prompt
    assert "Sahm (from 'Sam')" in prompt
    assert "audio" not in prompt.lower()


def test_cleanup_prompt_teaches_wolof_splitting_and_place_hints(
    monkeypatch,
) -> None:
    monkeypatch.setattr(listen, "vocabulary", lambda: "Sahm, Médina, Malika")
    monkeypatch.setattr(
        listen,
        "find_place_hints",
        lambda _raw: [
            "Sahm (from 'mangisam')",
            "Médina (from 'me dina')",
        ],
    )
    client = _FakeClient(
        '{"text": "Maa ngi Sahm, bëgg naa dem Malika", '
        '"language": "Wolof", "places_fixed": ["Sahm", "Malika"]}'
    )

    cleanup_transcript(client, "mangisam dama bagadam malika")

    prompt = client.calls[0]["messages"][0]["content"]
    assert "standard Dakar Wolof Latin spelling" in prompt
    assert "Possible places heard:" in prompt
    assert "Sahm (from 'mangisam')" in prompt
    assert "Médina (from 'me dina')" in prompt
    assert "raw: mangisam" in prompt
    assert "clean: Maa ngi Sahm" in prompt
    assert "raw: me dina" in prompt
    assert "clean: Médina" in prompt
    assert "clean: bëgga dem" in prompt
    assert "clean: bëgg naa dem" in prompt
    assert "clean: dama nekk" in prompt
    assert "Never invent a place" in prompt


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
