"""Tool handlers return JSON, including recoverable errors."""

from __future__ import annotations

from koumba.router import NoRouteFoundError
from koumba.tools import execute_tool


class _NoRouteEngine:
    def __init__(self, place_index) -> None:
        self.place_index = place_index

    def find_trips(self, *args, **kwargs):
        raise NoRouteFoundError("No bus trip found.")


def test_resolve_place_returns_scored_candidates(engine) -> None:
    result = execute_tool(
        "resolve_place",
        {"query": "Malika"},
        engine,
        engine.place_index,
    )

    assert result["candidates"]
    names = {candidate["name"] for candidate in result["candidates"]}
    assert "Malika" in names
    assert all(candidate["score"] >= 80 for candidate in result["candidates"])


def test_resolve_place_lists_exact_matches_first(engine) -> None:
    result = execute_tool(
        "resolve_place",
        {"query": "Yoff"},
        engine,
        engine.place_index,
    )

    candidates = result["candidates"]
    assert candidates[0]["name"] == "Yoff"
    assert candidates[0]["normalized"] == "yoff"
    assert any(
        candidate["normalized"] == "grand yoff"
        for candidate in candidates
    )

    accented = execute_tool(
        "resolve_place",
        {"query": "medina"},
        engine,
        engine.place_index,
    )
    assert accented["candidates"][0]["normalized"] == "medina"
    assert accented["candidates"][0]["name"] == "Médina"


def test_resolve_place_unknown_returns_suggestions(engine) -> None:
    result = execute_tool(
        "resolve_place",
        {"query": "Atlantis"},
        engine,
        engine.place_index,
    )

    assert result["error"] == "place_not_found"
    assert result["suggestions"]


def test_find_trips_tool_filters_operator_and_returns_plain_json(engine) -> None:
    result = execute_tool(
        "find_trips",
        {
            "origin": "Ouakam",
            "destination": "Palais",
            "operator": "Dakar Dem Dikk",
        },
        engine,
        engine.place_index,
    )

    assert result["trips"]
    for trip in result["trips"]:
        assert {"transfers", "total_minutes", "legs", "walks"} <= set(trip)
        for leg in trip["legs"]:
            assert leg["operator"] == "Dakar Dem Dikk"
            assert {
                "operator",
                "line",
                "board_stop",
                "alight_stop",
                "direction",
                "num_stops",
                "minutes",
            } <= set(leg)


def test_find_trips_tool_returns_place_not_found_json(engine) -> None:
    result = execute_tool(
        "find_trips",
        {"origin": "Atlantis", "destination": "Malika"},
        engine,
        engine.place_index,
    )

    assert result["error"] == "place_not_found"
    assert result["suggestions"]


def test_find_trips_tool_returns_no_route_json(engine) -> None:
    result = execute_tool(
        "find_trips",
        {"origin": "Ouakam", "destination": "Palais"},
        _NoRouteEngine(engine.place_index),
        engine.place_index,
    )

    assert result["error"] == "no_route"
    assert "suggestions" in result


def test_find_trips_tool_rejects_invalid_operator(engine) -> None:
    result = execute_tool(
        "find_trips",
        {
            "origin": "Ouakam",
            "destination": "Palais",
            "operator": "Unknown",
        },
        engine,
        engine.place_index,
    )

    assert result["error"] == "invalid_operator"
    assert result["allowed_operators"]


def test_unknown_tool_is_a_json_error(engine) -> None:
    result = execute_tool("not_a_tool", {}, engine, engine.place_index)

    assert result["error"] == "unknown_tool"
