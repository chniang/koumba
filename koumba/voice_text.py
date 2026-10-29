"""Prepare the exact Wolof sentence that the TTS service should speak.

Oolel-Voices sounds best on a complete sentence, so this module keeps the
displayed punctuation and wording.  It only swaps bus and place names for the
reviewed ``say_as`` spelling from ``data/voice_lexicon.csv``.  A place that is
not in the lexicon still gets the whole-word overrides from
``data/voice_word_overrides.csv``.

The CSV files are read once per process.  Callers pass the same route objects
used to build the displayed answer, which lets us reuse the existing
``bus_id`` and ``place_id`` helpers without reconstructing a second sentence.
"""

from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .phrases import short_bus_name
from .voice_lexicon import apply_word_overrides, bus_id, place_id

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LEXICON_PATH = DATA_DIR / "voice_lexicon.csv"
WORD_OVERRIDES_PATH = DATA_DIR / "voice_word_overrides.csv"


@lru_cache(maxsize=1)
def _lexicon() -> dict[str, dict[str, str]]:
    with LEXICON_PATH.open("r", encoding="utf-8", newline="") as handle:
        return {
            row["id"]: row
            for row in csv.DictReader(handle)
            if row.get("id")
        }


@lru_cache(maxsize=1)
def _word_overrides() -> dict[str, str]:
    with WORD_OVERRIDES_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return {
            row["word"]: row["say_as"]
            for row in csv.DictReader(handle)
            if row.get("word") and row.get("say_as")
        }


def _say_as(
    row_id: str,
    text: str,
    *,
    apply_overrides: bool,
) -> str:
    row = _lexicon().get(row_id)
    if row:
        reviewed = str(row.get("say_as") or "").strip()
        if reviewed:
            return reviewed
    if apply_overrides:
        return apply_word_overrides(text, _word_overrides())
    return text


def _add_place(replacements: dict[str, str], text: Any) -> None:
    name = str(text or "").strip()
    if not name:
        return
    replacements.setdefault(
        name,
        _say_as(place_id(name), name, apply_overrides=True),
    )


def _add_bus(
    replacements: dict[str, str],
    operator: Any,
    line: Any,
) -> None:
    name = short_bus_name(operator, line)
    if not name:
        return
    replacements.setdefault(
        name,
        _say_as(
            bus_id(operator, line),
            name,
            apply_overrides=False,
        ),
    )


def _add_trip(
    replacements: dict[str, str],
    trip: Any,
) -> None:
    if not isinstance(trip, dict):
        return

    legs = trip.get("legs")
    if isinstance(legs, list):
        for leg in legs:
            if not isinstance(leg, dict):
                continue
            _add_bus(
                replacements,
                leg.get("operator"),
                leg.get("line"),
            )
            for key in ("board_stop", "alight_stop", "direction"):
                _add_place(replacements, leg.get(key))

    walks = trip.get("walks")
    if isinstance(walks, list):
        for walk in walks:
            if not isinstance(walk, dict):
                continue
            _add_place(replacements, walk.get("from_place"))
            _add_place(replacements, walk.get("to_place"))


def _replace_names(text: str, replacements: dict[str, str]) -> str:
    """Replace longer names first so overlapping names cannot corrupt each other."""

    placeholders: dict[str, str] = {}
    for index, (name, say_as) in enumerate(
        sorted(replacements.items(), key=lambda item: len(item[0]), reverse=True)
    ):
        if name == say_as:
            continue
        pattern = re.compile(rf"(?<![\w-]){re.escape(name)}(?![\w-])")
        if not pattern.search(text):
            continue
        placeholder = f"\x00{index}\x00"
        text = pattern.sub(placeholder, text)
        placeholders[placeholder] = say_as

    for placeholder, say_as in placeholders.items():
        text = text.replace(placeholder, say_as)
    return text


def tts_text(
    spoken: str,
    *,
    trip: dict[str, Any] | None = None,
    origin: str | None = None,
    check: dict[str, Any] | None = None,
    destination: str | None = None,
    alternative: dict[str, Any] | None = None,
) -> str:
    """Return the displayed Wolof sentence with reviewed names substituted."""

    text = str(spoken or "")
    if not text.strip():
        return text

    replacements: dict[str, str] = {}
    _add_place(replacements, origin)
    _add_trip(replacements, trip)

    if isinstance(check, dict):
        _add_bus(
            replacements,
            check.get("operator"),
            check.get("line"),
        )
        _add_place(
            replacements,
            destination or check.get("destination"),
        )
        _add_place(replacements, check.get("alight_stop"))
        _add_trip(replacements, check.get("trip"))
        _add_trip(replacements, check.get("best_alternative"))

    _add_trip(replacements, alternative)
    return _replace_names(text, replacements)
