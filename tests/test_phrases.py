"""The spoken answers use the project owner's validated short templates."""

from __future__ import annotations

from koumba.phrases import (
    bus_check_spoken,
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


def _trip(
    legs: list[dict],
    minutes: float,
    walks: list[dict] | None = None,
) -> dict:
    return {
        "total_minutes": minutes,
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
        "Yaangi Sahm, duggal ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "Grand Yoff."
    )


def test_direct_answer_mentions_the_walk_to_the_first_stop() -> None:
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
        46.8,
        [_access_walk("Médina", "Avenue Blaise Diagne", 1.9)],
    )

    assert spoken_answer(trip, "Wolof", "Médina") == (
        "Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tata 3 bi "
        "jëm Yoff-Village, wàcci Yoff-Village."
    )
    assert spoken_answer(trip, "French", "Médina") == (
        "Tu es à Médina : marche jusqu'à Avenue Blaise Diagne, prends le "
        "Tata 3 direction Yoff-Village, descends à Yoff-Village."
    )
    assert spoken_answer(trip, "English", "Médina") == (
        "You are at Médina: walk to Avenue Blaise Diagne, take Tata 3 toward "
        "Yoff-Village, get off at Yoff-Village."
    )


def test_transfer_answer_mentions_the_walk_to_the_first_stop() -> None:
    trip = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 3",
                "Avenue Blaise Diagne",
                "PAI",
                "Cité des Enseignants",
            ),
            _leg("Tata AFTU", "Ligne 80", "PAI", "Malika", "Darou Thioub"),
        ],
        80,
        [_access_walk("Médina", "Avenue Blaise Diagne", 2)],
    )

    assert spoken_answer(trip, "Wolof", "Médina") == (
        "Yaangi Médina, doxal ba Avenue Blaise Diagne, duggal ci Tata 3 bi "
        "jëm Cité des Enseignants, wàcci PAI, nga jël Tata 80 bi jëm Darou "
        "Thioub, wàcci Malika."
    )


def test_short_or_same_name_walk_is_not_spoken() -> None:
    short_walk = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 3",
                "Avenue Blaise Diagne",
                "Yoff-Village",
                "Yoff-Village",
            )
        ],
        46.8,
        [_access_walk("Médina", "Avenue Blaise Diagne", 0.8)],
    )
    same_name = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 3",
                "Médina",
                "Yoff-Village",
                "Yoff-Village",
            )
        ],
        46.8,
        [_access_walk("Médina", "Médina", 2)],
    )

    assert "doxal ba" not in spoken_answer(short_walk, "Wolof", "Médina")
    assert "marche jusqu'à" not in spoken_answer(
        short_walk,
        "French",
        "Médina",
    )
    assert "doxal ba" not in spoken_answer(same_name, "Wolof", "Médina")


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
        "Yaangi Sahm, duggal ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "PAI, nga jël Tata 80 bi jëm Darou Thioub, wàcci Malika."
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
        "Yaangi Sahm, duggal ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "PAI, nga jël Tata 80 bi jëm Darou Thioub, wàcci Malika, nga jël "
        "Dem Dikk 7 bi jëm Palais 2, wàcci Palais 2."
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
        "Palais 2."
    )
    assert spoken_answer(trip, "English", "Ouakam") == (
        "From Ouakam, take Dem Dikk 7 toward Palais 2, get off at Palais 2."
    )


def test_bus_check_yes_uses_the_validated_wolof_wording() -> None:
    check = {
        "serves_trip": True,
        "operator": "Tata AFTU",
        "line": "Ligne 38",
        "alight_stop": "PAI",
        "minutes": 48.1,
    }

    assert bus_check_spoken(check, "Wolof", "PAI") == (
        "Waaw, Tata 38 bi dafay dem PAI. Duggal ci, wàcci PAI."
    )
    assert bus_check_spoken(check, "French", "PAI") == (
        "Oui, le Tata 38 va à PAI. Monte et descends à PAI."
    )
    assert bus_check_spoken(check, "English", "PAI") == (
        "Yes, Tata 38 goes to PAI. Board it and get off at PAI."
    )


def test_bus_check_no_adds_the_best_alternative() -> None:
    check = {
        "serves_trip": False,
        "operator": "Tata AFTU",
        "line": "Ligne 38",
        "can_help_part_way": True,
    }
    alternative = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 38",
                "Sahm",
                "PAI",
                "Cité des Enseignants",
            ),
            _leg("Tata AFTU", "Ligne 80", "PAI", "Malika", "Darou Thioub"),
        ],
        80,
    )

    assert bus_check_spoken(
        check,
        "Wolof",
        "Malika",
        alternative,
        "Sahm",
    ) == (
        "Déedéet, Tata 38 bi du dem Malika. Yaangi Sahm, duggal ci Tata 38 "
        "bi jëm Cité des Enseignants, wàcci PAI, nga jël Tata 80 bi jëm Darou "
        "Thioub, wàcci Malika."
    )
    assert bus_check_spoken(check, "French", "Malika").startswith(
        "Non, le Tata 38 ne va pas à Malika."
    )
    assert bus_check_spoken(check, "English", "Malika").startswith(
        "No, Tata 38 does not go to Malika."
    )


def test_bus_check_no_alternative_mentions_the_first_walk() -> None:
    check = {
        "serves_trip": False,
        "operator": "Tata AFTU",
        "line": "Ligne 5",
    }
    alternative = _trip(
        [
            _leg(
                "Tata AFTU",
                "Ligne 3",
                "Avenue Blaise Diagne",
                "Yoff-Village",
                "Yoff-Village",
            )
        ],
        46.8,
        [_access_walk("Médina", "Avenue Blaise Diagne", 1.9)],
    )

    assert bus_check_spoken(
        check,
        "Wolof",
        "Yoff",
        alternative,
        "Médina",
    ) == (
        "Déedéet, Tata 5 bi du dem Yoff. Yaangi Médina, doxal ba Avenue "
        "Blaise Diagne, duggal ci Tata 3 bi jëm Yoff-Village, wàcci "
        "Yoff-Village."
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
        "Yaangi Sahm, duggal ci Tata 38 bi jëm Cité des Enseignants, wàcci "
        "Grand Yoff."
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
