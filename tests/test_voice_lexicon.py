"""The mechanical place suggestions never guess vowel length."""

from __future__ import annotations

import csv
from pathlib import Path

from koumba.voice_lexicon import (
    CONNECTOR_TEXTS,
    bus_id,
    connector_id,
    place_id,
    place_needs_review,
    safe_place_suggestion,
    slug,
)

ROOT = Path(__file__).resolve().parent.parent


def test_place_suggestion_applies_only_the_safe_letter_conversions() -> None:
    assert safe_place_suggestion("Ouakam") == "Uakam"
    assert safe_place_suggestion("Keur Massar") == "Kër Massar"
    assert safe_place_suggestion("Tchic") == "Cic"
    assert safe_place_suggestion("Sahm") == "Sahm"
    assert safe_place_suggestion("PAI") == "PAI"


def test_place_suggestion_keeps_vowel_length_untouched() -> None:
    assert safe_place_suggestion("Tataa") == "Tataa"
    assert safe_place_suggestion("Saahm") == "Saahm"


def test_french_spelling_patterns_are_sent_to_review() -> None:
    assert place_needs_review("Ouakam")
    assert place_needs_review("Cheikh")
    assert place_needs_review("Keur Massar")
    assert not place_needs_review("Sahm")
    assert not place_needs_review("Grand Yoff")


def test_ids_are_stable_and_case_sensitive_where_it_matters() -> None:
    assert slug("Déedéet") == "deedeet"
    assert connector_id("Duggal ci") == connector_id("duggal ci")
    assert bus_id("Tata AFTU", "Ligne 38") == "bus:tata-aftu:ligne-38"
    assert place_id("Sahm") != place_id("SAHM")
    assert place_id("Sahm") == place_id("Sahm")


def _lexicon_rows() -> list[dict[str, str]]:
    path = ROOT / "data" / "voice_lexicon.csv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_generated_lexicon_has_one_row_per_unique_id() -> None:
    rows = _lexicon_rows()

    assert list(rows[0]) == [
        "id",
        "kind",
        "text",
        "say_as",
        "suggestion",
        "status",
    ]
    assert len(rows) == len({row["id"] for row in rows})
    assert {row["status"] for row in rows} == {"validated", "auto", "review"}


def test_connectors_and_validated_bus_numbers_are_in_the_lexicon() -> None:
    rows = _lexicon_rows()
    by_text = {row["text"]: row for row in rows}

    for text in CONNECTOR_TEXTS:
        row = by_text[text]
        assert row["kind"] == "connector"
        assert row["say_as"] == text
        assert row["status"] == "validated"

    assert by_text["Tata 38"]["say_as"] == "Tataa fanweer ak juróom ñett"
    assert by_text["Tata 38"]["status"] == "validated"
    assert by_text["Dem Dikk 16A"]["say_as"] == "Dem Dikk 16A"
    assert by_text["Dem Dikk 16A"]["status"] == "review"


def test_place_overrides_win_over_the_mechanical_suggestion() -> None:
    rows = _lexicon_rows()
    places = {row["text"]: row for row in rows if row["kind"] == "place"}

    assert places["Sahm"]["say_as"] == "Saahm"
    assert places["SAHM"]["say_as"] == "Saahm"
    assert places["PAI"]["say_as"] == "Paay"
    assert places["Sahm"]["status"] == "validated"
    assert places["Ouakam"]["say_as"] == "Ouakam"
    assert places["Ouakam"]["suggestion"] == "Uakam"
    assert places["Ouakam"]["status"] == "review"


def test_review_shortlist_has_fifty_readable_places() -> None:
    lines = (ROOT / "data" / "voice_review_top50.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 50
    assert all("  ->  " in line for line in lines)
