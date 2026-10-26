"""The spoken answers use the project owner's validated short templates."""

from __future__ import annotations

from koumba.phrases import (
    clarification_question,
    compose_chat_message,
    round_minutes,
    short_bus_name,
    spoken_answer,
)


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


def _trip(legs: list[dict], minutes: float) -> dict:
    return {"total_minutes": minutes, "legs": legs}


def test_direct_wolof_answer_matches_the_validated_template() -> None:
    trip = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 38",
                "Sahm",
                "Grand Yoff",
                "Cité des Enseignants",
            )
        ],
        42.5,
    )

    assert spoken_answer(trip, "Wolof", "Sahm") == (
        "Ci Sahm, duggu ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "Grand Yoff. Lu tollu ci 45 minit."
    )


def test_one_transfer_wolof_answer_matches_the_validated_template() -> None:
    trip = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 38",
                "Sahm",
                "PAI",
                "Cité des Enseignants",
            ),
            _leg(
                "Tata AFTU",
                "Ligne 80",
                "PAI",
                "Malika",
                "Darou Thioub",
            ),
        ],
        80,
    )

    assert spoken_answer(trip, "Wolof", "Sahm") == (
        "Ci Sahm, duggu ci Tata 38 bi jëm Cité des Enseignants, wàcci PAI, "
        "nga jël Tata 80 bi jëm Darou Thioub, wàcci Malika. Lu tollu ci 80 "
        "minit."
    )


def test_two_transfers_chain_the_same_wolof_pattern() -> None:
    trip = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 38",
                "Sahm",
                "PAI",
                "Cité des Enseignants",
            ),
            _leg("Tata AFTU", "Ligne 80", "PAI", "Malika", "Darou Thioub"),
            _leg(
                "Dakar Dem Dikk",
                "Ligne 7",
                "Malika",
                "Palais 2",
                "Palais 2",
            ),
        ],
        92.5,
    )

    assert spoken_answer(trip, "Wolof", "Sahm") == (
        "Ci Sahm, duggu ci Tata 38 bi jëm Cité des Enseignants, wàcci PAI, "
        "nga jël Tata 80 bi jëm Darou Thioub, wàcci Malika, nga jël "
        "Dem Dikk 7 bi jëm Palais 2, wàcci Palais 2. Lu tollu ci 95 minit."
    )


def test_french_and_english_answers_are_short() -> None:
    trip = _trip(
        [
            _leg(
                "Dakar Dem Dikk",
                "Ligne 7",
                "Ouakam",
                "Palais 2",
                "Palais 2",
            )
        ],
        40,
    )

    assert spoken_answer(trip, "French", "Ouakam") == (
        "À Ouakam, prends le Dem Dikk 7 direction Palais 2, descends à "
        "Palais 2. Environ 40 minutes."
    )
    assert spoken_answer(trip, "English", "Ouakam") == (
        "From Ouakam, take Dem Dikk 7 toward Palais 2, get off at Palais 2. "
        "About 40 minutes."
    )


def test_clarification_question_matches_the_validated_template() -> None:
    assert clarification_question("Yoff-Village", "Grand Yoff") == (
        "Yoff-Village nga bëgga dem walla Grand Yoff ?"
    )


def test_bus_names_are_short() -> None:
    assert short_bus_name("Tata AFTU", "Ligne 38") == "Tata 38"
    assert short_bus_name("Dakar Dem Dikk", "Ligne 7") == "Dem Dikk 7"


def test_minutes_round_to_the_nearest_five() -> None:
    assert round_minutes(42.4) == 40
    assert round_minutes(42.5) == 45
    assert round_minutes(7.4) == 5
    assert round_minutes(7.5) == 10
    assert round_minutes(None) == 0


def test_chat_message_adds_only_a_model_clarification_question() -> None:
    spoken = (
        "Ci Sahm, duggu ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "Grand Yoff. Lu tollu ci 45 minit."
    )

    assert compose_chat_message(
        spoken,
        "Here is a long explanation. Yoff-Village walla Grand Yoff?",
    ) == f"{spoken} Yoff-Village walla Grand Yoff?"
    assert compose_chat_message(
        spoken,
        "Here is a long explanation with no question.",
    ) == spoken
    assert compose_chat_message("", "A short reply.") == "A short reply."
