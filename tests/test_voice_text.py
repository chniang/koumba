"""The TTS text keeps the displayed sentence and only swaps reviewed names."""

from __future__ import annotations

from koumba.phrases import bus_check_spoken, spoken_answer
from koumba.voice_text import tts_text


def _leg(
    operator: str,
    line: str,
    board_stop: str,
    alight_stop: str,
    direction: str,
) -> dict:
    return {
        "operator": operator,
        "line": line,
        "board_stop": board_stop,
        "alight_stop": alight_stop,
        "direction": direction,
    }


def _trip(legs: list[dict], walks: list[dict] | None = None) -> dict:
    return {
        "total_minutes": 80,
        "legs": legs,
        "walks": walks or [],
    }


def _access_walk(
    from_place: str,
    to_place: str,
    minutes: float,
) -> dict:
    return {
        "kind": "access",
        "from_place": from_place,
        "to_place": to_place,
        "minutes": minutes,
        "after_leg": 0,
    }


def _transfer_trip() -> dict:
    return _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 38",
                "Sahm",
                "PAI",
                "Cité des Enseignants",
            ),
            _leg("Tata AFTU", "Ligne 80", "PAI", "Malika", "Darou Thioub"),
        ]
    )


def test_transfer_replaces_sahm_pai_and_the_two_bus_names() -> None:
    trip = _transfer_trip()
    displayed = spoken_answer(trip, "Wolof", "Sahm")

    assert tts_text(displayed, trip=trip, origin="Sahm") == (
        "Yaangi Saahm, duggal ci Tataa fanweer ak juróom ñett bi jëm Cité des "
        "Enseignants, wàcci Paay, nga jël Tataa juróom ñett fukk bi jëm Darou "
        "Thioub, wàcci Malika."
    )


def test_walk_first_answer_keeps_its_punctuation_and_uses_tataa() -> None:
    trip = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 3",
                "Avenue Blaise Diagne",
                "Yoff-Village",
                "Yoff-Village",
            )
        ],
        [_access_walk("Médina", "Avenue Blaise Diagne", 1.9)],
    )
    displayed = spoken_answer(trip, "Wolof", "Médina")

    assert tts_text(displayed, trip=trip, origin="Médina") == (
        "Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tataa ñett "
        "bi jëm Yoff-Village, wàcci Yoff-Village."
    )


def test_bus_check_yes_replaces_the_bus_and_destination() -> None:
    check = {
        "serves_trip": True,
        "operator": "Tata AFTU",
        "line": "Ligne 38",
        "alight_stop": "PAI",
    }
    displayed = bus_check_spoken(check, "Wolof", "PAI")

    assert tts_text(
        displayed,
        check=check,
        destination="PAI",
    ) == (
        "Waaw, Tataa fanweer ak juróom ñett bi dafay dem Paay. "
        "Duggal ci, wàcci Paay."
    )


def test_bus_check_no_replaces_the_bus_and_its_alternative() -> None:
    check = {
        "serves_trip": False,
        "operator": "Tata AFTU",
        "line": "Ligne 38",
        "can_help_part_way": True,
    }
    alternative = _transfer_trip()
    displayed = bus_check_spoken(
        check,
        "Wolof",
        "Malika",
        alternative,
        "Sahm",
    )

    assert tts_text(
        displayed,
        check=check,
        destination="Malika",
        alternative=alternative,
        origin="Sahm",
    ) == (
        "Déedéet, Tataa fanweer ak juróom ñett bi du dem Malika. "
        "Yaangi Saahm, duggal ci Tataa fanweer ak juróom ñett bi jëm Cité des "
        "Enseignants, wàcci Paay, nga jël Tataa juróom ñett fukk bi jëm Darou "
        "Thioub, wàcci Malika."
    )


def test_place_missing_from_the_lexicon_still_uses_word_overrides() -> None:
    displayed = (
        "Yaangi Sahm Extension, duggal ci Tata 3 bi jëm Yoff-Village."
    )

    assert tts_text(displayed, origin="Sahm Extension") == (
        "Yaangi Saahm Extension, duggal ci Tata 3 bi jëm Yoff-Village."
    )
