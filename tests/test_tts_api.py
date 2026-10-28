"""The TTS ASGI app is tested with a fake synthesizer and no Modal runtime."""

from __future__ import annotations

import pytest

pytest.importorskip("modal")

from fastapi.testclient import TestClient

from modal_app.tts import MAX_TEXT_CHARS, create_api

SECRET = "test-secret"
WAV_BYTES = b"RIFF-test-wave"


class _FakeSynthesizer:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, text: str) -> bytes:
        self.calls.append(text)
        return WAV_BYTES


def _client() -> tuple[TestClient, _FakeSynthesizer]:
    synthesizer = _FakeSynthesizer()
    api = create_api(
        synthesizer,
        lambda key: key == SECRET,
    )
    return TestClient(api), synthesizer


def _auth_header() -> dict[str, str]:
    return {"X-Koumba-Key": SECRET}


def test_speak_accepts_json_body_with_header() -> None:
    client, synthesizer = _client()

    response = client.post(
        "/speak",
        json={"text": "  Maa ngi Sahm, bëgg naa dem Malika  "},
        headers=_auth_header(),
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content == WAV_BYTES
    assert synthesizer.calls == ["maa ngi sahm, bëgg naa dem malika"]


def test_speak_requires_the_shared_header() -> None:
    client, synthesizer = _client()

    response = client.post(
        "/speak",
        json={"text": "Maa ngi Sahm."},
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Unauthorized"
    assert synthesizer.calls == []


def test_speak_rejects_a_wrong_key() -> None:
    client, synthesizer = _client()

    response = client.post(
        "/speak",
        json={"text": "Maa ngi Sahm."},
        headers={"X-Koumba-Key": "wrong-secret"},
    )

    assert response.status_code == 401
    assert synthesizer.calls == []


def test_speak_missing_text_is_422() -> None:
    client, synthesizer = _client()

    response = client.post(
        "/speak",
        json={},
        headers=_auth_header(),
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "text"]
    assert synthesizer.calls == []


def test_speak_overlong_text_is_422() -> None:
    client, synthesizer = _client()

    response = client.post(
        "/speak",
        json={"text": "a" * (MAX_TEXT_CHARS + 1)},
        headers=_auth_header(),
    )

    assert response.status_code == 422
    assert synthesizer.calls == []
