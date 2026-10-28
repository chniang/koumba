"""Segment ids follow the exact order of the validated Wolof answers."""

from __future__ import annotations

from koumba.phrases import spoken_segments
from koumba.voice_lexicon import bus_id, connector_id, place_id


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


def test_direct_trip_segments() -> None:
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

    assert spoken_segments(trip, "Wolof", "Sahm") == [
        connector_id("Yaangi"),
        place_id("Sahm"),
        connector_id("duggal ci"),
        bus_id("Tata AFTU", "Ligne 38"),
        connector_id("bi jëm"),
        place_id("Cité des Enseignants"),
        connector_id("wàcci"),
        place_id("Grand Yoff"),
    ]


def test_one_transfer_segments_stay_in_sentence_order() -> None:
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
        ],
        80,
    )

    assert spoken_segments(trip, "Wolof", "Sahm") == [
        connector_id("Yaangi"),
        place_id("Sahm"),
        connector_id("duggal ci"),
        bus_id("Tata AFTU", "Ligne 38"),
        connector_id("bi jëm"),
        place_id("Cité des Enseignants"),
        connector_id("wàcci"),
        place_id("PAI"),
        connector_id("nga jël"),
        bus_id("Tata AFTU", "Ligne 80"),
        connector_id("bi jëm"),
        place_id("Darou Thioub"),
        connector_id("wàcci"),
        place_id("Malika"),
    ]


def test_walk_first_segments_include_the_boarding_stop() -> None:
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

    assert spoken_segments(trip, "Wolof", "Médina") == [
        connector_id("Yaangi"),
        place_id("Médina"),
        connector_id("doxal ba"),
        place_id("Avenue Blaise Diagne"),
        connector_id("duggal ci"),
        bus_id("Tata AFTU", "Ligne 3"),
        connector_id("bi jëm"),
        place_id("Yoff-Village"),
        connector_id("wàcci"),
        place_id("Yoff-Village"),
    ]


def test_bus_check_yes_segments() -> None:
    check = {
        "serves_trip": True,
        "operator": "Tata AFTU",
        "line": "Ligne 38",
        "alight_stop": "PAI",
    }

    assert spoken_segments(
        check=check,
        language="Wolof",
        destination="PAI",
    ) == [
        connector_id("Waaw"),
        bus_id("Tata AFTU", "Ligne 38"),
        connector_id("bi dafay dem"),
        place_id("PAI"),
        connector_id("duggal ci"),
        connector_id("wàcci"),
        place_id("PAI"),
    ]


def test_bus_check_no_segments_append_the_alternative() -> None:
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

    assert spoken_segments(
        check=check,
        language="Wolof",
        destination="Malika",
        alternative=alternative,
        origin="Sahm",
    ) == [
        connector_id("Déedéet"),
        bus_id("Tata AFTU", "Ligne 38"),
        connector_id("bi du dem"),
        place_id("Malika"),
        *spoken_segments(alternative, "Wolof", "Sahm"),
    ]


def test_non_wolof_answers_have_no_segments() -> None:
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

    assert spoken_segments(trip, "French", "Ouakam") == []
    assert spoken_segments(trip, "English", "Ouakam") == []
