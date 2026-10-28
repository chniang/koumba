"""Voice lexicon artifacts and the conservative pronunciation helpers."""

from __future__ import annotations

import csv
from pathlib import Path

from koumba.voice_lexicon import (
    CONNECTOR_TEXTS,
    apply_word_overrides,
    bus_id,
    connector_id,
    place_id,
    place_needs_review,
    safe_place_suggestion,
    slug,
)

ROOT = Path(__file__).resolve().parent.parent
WORD_OVERRIDES = {
    "Sahm": "Saahm",
    "PAI": "Paay",
    "Ouakam": "Wakaam",
    "Bargny": "Barñi",
    "Fass": "Faass",
    "Mbédou": "Mbeddou",
    "UCAD": "Ikaad",
    "MTOA": "Emtowaa",
    "LSS": "El Ess Ess",
}


def test_place_suggestion_applies_only_the_safe_letter_conversions() -> None:
    assert safe_place_suggestion("Ouakam") == "Uakam"
    assert safe_place_suggestion("Keur Massar") == "Kër Massar"
    assert safe_place_suggestion("Tchic") == "Cic"
    assert safe_place_suggestion("Sahm") == "Sahm"
    assert safe_place_suggestion("PAI") == "PAI"


def test_place_suggestion_keeps_vowel_length_untouched() -> None:
    assert safe_place_suggestion("Tataa") == "Tataa"
    assert safe_place_suggestion("Saahm") == "Saahm"


def test_word_overrides_replace_whole_words_case_insensitively() -> None:
    assert apply_word_overrides("Fass Mbao", WORD_OVERRIDES) == "Faass Mbao"
    assert apply_word_overrides("Mbédou Fass", WORD_OVERRIDES) == "Mbeddou Faass"
    assert (
        apply_word_overrides("Aéroport LSS", WORD_OVERRIDES)
        == "Aéroport El Ess Ess"
    )
    assert (
        apply_word_overrides("Croisement PAI", WORD_OVERRIDES)
        == "Croisement Paay"
    )
    assert apply_word_overrides("FASS", WORD_OVERRIDES) == "Faass"
    assert apply_word_overrides("ucad", WORD_OVERRIDES) == "Ikaad"


def test_word_overrides_do_not_touch_longer_words() -> None:
    assert apply_word_overrides("Fassa", WORD_OVERRIDES) == "Fassa"
    assert apply_word_overrides("PAIX", WORD_OVERRIDES) == "PAIX"


def test_french_spelling_patterns_are_still_detected_for_review_notes() -> None:
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


def test_place_overrides_drive_say_as_and_status() -> None:
    rows = _lexicon_rows()
    places = {row["text"]: row for row in rows if row["kind"] == "place"}

    assert places["Sahm"]["say_as"] == "Saahm"
    assert places["SAHM"]["say_as"] == "Saahm"
    assert places["PAI"]["say_as"] == "Paay"
    assert places["Ouakam"]["say_as"] == "Wakaam"
    assert places["Fass Mbao"]["say_as"] == "Faass Mbao"
    assert places["Mbédou Fass"]["say_as"] == "Mbeddou Faass"
    assert (
        places["Aéroport LSS"]["say_as"]
        == "Aéroport El Ess Ess"
    )
    assert places["Croisement PAI"]["say_as"] == "Croisement Paay"

    for text in (
        "Sahm",
        "SAHM",
        "PAI",
        "Ouakam",
        "Fass Mbao",
        "Mbédou Fass",
        "Aéroport LSS",
        "Croisement PAI",
    ):
        assert places[text]["status"] == "validated"


def test_reviewed_top50_places_are_validated_as_written() -> None:
    rows = _lexicon_rows()
    places = {row["text"]: row for row in rows if row["kind"] == "place"}
    reviewed = [
        line.split("  ->  ", 1)[0]
        for line in (ROOT / "data" / "voice_review_top50.txt")
        .read_text(encoding="utf-8")
        .splitlines()
    ]

    assert len(reviewed) == 50
    for text in reviewed:
        assert places[text]["status"] == "validated"
        word_changed = apply_word_overrides(text, WORD_OVERRIDES) != text
        full_name_changed = text in {"Sahm", "SAHM", "PAI"}
        if not word_changed and not full_name_changed:
            assert places[text]["say_as"] == text


def test_unreviewed_places_stay_auto() -> None:
    rows = _lexicon_rows()
    places = {row["text"]: row for row in rows if row["kind"] == "place"}

    assert places["Yoff-Village"]["say_as"] == "Yoff-Village"
    assert places["Yoff-Village"]["status"] == "auto"
    assert {place["status"] for place in places.values()} == {
        "validated",
        "auto",
    }


def test_mechanical_suggestion_is_never_copied_into_say_as() -> None:
    rows = _lexicon_rows()
    places = {row["text"]: row for row in rows if row["kind"] == "place"}

    assert places["Boune"]["say_as"] == "Boune"
    assert places["Boune"]["suggestion"] == "Bune"
    assert places["Boune"]["status"] == "auto"


def test_review_shortlist_has_fifty_readable_places() -> None:
    lines = (ROOT / "data" / "voice_review_top50.txt").read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 50
    assert all("  ->  " in line for line in lines)
