"""The Modal services read the warm-container switch at deploy time."""

from __future__ import annotations

import pytest

pytest.importorskip("modal")

from modal_app.asr import _read_keep_warm as read_asr_keep_warm
from modal_app.tts import _read_keep_warm as read_tts_keep_warm
from modal_app.web import _read_keep_warm as read_web_keep_warm

READERS = (read_asr_keep_warm, read_tts_keep_warm, read_web_keep_warm)


def test_keep_warm_defaults_to_zero(monkeypatch) -> None:
    monkeypatch.delenv("KOUMBA_KEEP_WARM", raising=False)

    assert [reader() for reader in READERS] == [0, 0, 0]


def test_keep_warm_accepts_one(monkeypatch) -> None:
    monkeypatch.setenv("KOUMBA_KEEP_WARM", "1")

    assert [reader() for reader in READERS] == [1, 1, 1]


@pytest.mark.parametrize("value", ["yes", "2", "-1"])
def test_keep_warm_rejects_other_values(monkeypatch, value: str) -> None:
    monkeypatch.setenv("KOUMBA_KEEP_WARM", value)

    for reader in READERS:
        with pytest.raises(ValueError, match="KOUMBA_KEEP_WARM"):
            reader()
