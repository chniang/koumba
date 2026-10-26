"""Tool schemas and handlers exposed to the Muse Spark agent.

The handlers always return plain dictionaries.  A bad place name or a route
with no connection becomes an ``{"error": ...}`` result instead of an
exception, so a model tool round can recover and ask the user a question.
"""

from __future__ import annotations

import json
from typing import Any

from .places import PlaceIndex, normalize_name
from .router import NoRouteFoundError, PlaceNotFoundError, RouteEngine, Trip

OPERATORS = ("Dakar Dem Dikk", "Tata AFTU")
MAX_TOOL_TRIPS = 5

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "resolve_place",
            "description": (
                "Resolve a Dakar neighbourhood, landmark, or bus stop name. "
                "Use this when the place is ambiguous or when you want to "
                "show the user possible matches before searching."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The place name exactly as the user said it.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_trips",
            "description": (
                "Find bus trips in Dakar using the Koumba route engine. "
                "Returns direct trips first, then trips with transfers. "
                "Use the real stop or place names from the user, not invented "
                "line numbers. If the result is an error, ask a short question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {
                        "type": "string",
                        "description": "Where the user is starting.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Where the user wants to go.",
                    },
                    "max_trips": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_TOOL_TRIPS,
                        "default": 3,
                        "description": "How many distinct trips to return.",
                    },
                    "operator": {
                        "anyOf": [
                            {
                                "type": "string",
                                "enum": list(OPERATORS),
                            },
                            {"type": "null"},
                        ],
                        "description": (
                            "Optional operator filter. Use null to allow both "
                            "Dakar Dem Dikk and Tata AFTU."
                        ),
                    },
                },
                "required": ["origin", "destination"],
            },
        },
    },
]


def _round(value: float | None, digits: int = 1) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _suggestions(items: list[tuple[str, float]]) -> list[dict[str, Any]]:
    return [
        {"name": name, "score": round(float(score), 1)}
        for name, score in items
    ]


def _trip_to_dict(trip: Trip) -> dict[str, Any]:
    """Turn a Trip object into the JSON shape the model and UI can read."""

    legs = [
        {
            "operator": leg.operator,
            "line": leg.line,
            "board_stop": leg.board_stop,
            "alight_stop": leg.alight_stop,
            "direction": leg.direction,
            "num_stops": leg.num_stops,
            "minutes": _round(leg.minutes),
            "distance_m": _round(leg.distance_m, 0),
        }
        for leg in trip.legs
    ]
    walks = [
        {
            "kind": walk.kind,
            "from_place": walk.from_place,
            "to_place": walk.to_place,
            "minutes": _round(walk.minutes),
            "distance_m": _round(walk.distance_m, 0),
            "after_leg": walk.after_leg,
        }
        for walk in trip.walks
    ]
    return {
        "transfers": trip.transfers,
        "total_minutes": _round(trip.total_minutes),
        "total_distance_m": _round(trip.total_distance_m, 0),
        "straight_line_m": _round(trip.straight_line_m, 0),
        "detour": trip.detour,
        "via_downtown": trip.via_downtown,
        "line_combo": [
            [operator, line] for operator, line in trip.line_combo
        ],
        "legs": legs,
        "walks": walks,
    }


def resolve_place(
    place_index: PlaceIndex,
    query: str,
) -> dict[str, Any]:
    """Return the best place candidates, with a short error for no match."""

    query = (query or "").strip()
    if not query:
        return {
            "error": "invalid_query",
            "message": "The place name is empty.",
        }

    matches = place_index.match(query)
    if not matches:
        return {
            "error": "place_not_found",
            "query": query,
            "message": f"I could not find {query!r} in the Dakar stop data.",
            "suggestions": _suggestions(place_index.suggestions(query)),
        }

    query_norm = normalize_name(query)
    matches = sorted(
        matches,
        key=lambda candidate: candidate.normalized != query_norm,
    )

    return {
        "query": query,
        "candidates": [
            {
                "name": candidate.name,
                "score": round(candidate.score, 1),
                "normalized": candidate.normalized,
                "is_stop": candidate.is_stop,
                "is_place": candidate.is_place,
                "has_coordinates": candidate.has_coordinates,
                "stop_names": list(candidate.stop_names),
            }
            for candidate in matches[:10]
        ],
    }


def find_trips(
    engine: RouteEngine,
    origin: str,
    destination: str,
    max_trips: int = 3,
    operator: str | None = None,
) -> dict[str, Any]:
    """Run the real route engine and always return a JSON-safe result."""

    origin = str(origin or "").strip()
    destination = str(destination or "").strip()
    if not origin or not destination:
        return {
            "error": "invalid_request",
            "message": "Both an origin and a destination are needed.",
        }

    try:
        max_trips = int(max_trips)
    except (TypeError, ValueError):
        max_trips = 3
    max_trips = max(1, min(max_trips, MAX_TOOL_TRIPS))

    if operator is not None:
        operator = str(operator).strip()
        if operator.lower() in {"", "null", "none", "any", "all"}:
            operator = None
        else:
            for allowed in OPERATORS:
                if operator.lower() == allowed.lower():
                    operator = allowed
                    break
            else:
                return {
                    "error": "invalid_operator",
                    "message": (
                        "The operator must be Dakar Dem Dikk, Tata AFTU, "
                        "or null."
                    ),
                    "allowed_operators": list(OPERATORS),
                }

    try:
        trips = engine.find_trips(
            origin,
            destination,
            max_trips=max_trips,
            operator=operator,
        )
    except PlaceNotFoundError as exc:
        return {
            "error": "place_not_found",
            "query": exc.query,
            "message": f"I could not find {exc.query!r} in the Dakar stop data.",
            "suggestions": _suggestions(exc.suggestions),
        }
    except NoRouteFoundError as exc:
        return {
            "error": "no_route",
            "message": str(exc),
            "origin": origin,
            "destination": destination,
            "operator": operator,
            "suggestions": [],
        }
    except Exception as exc:  # Keep a tool mistake from breaking the chat.
        return {
            "error": "tool_error",
            "message": "The route engine could not complete this search.",
            "detail": str(exc),
        }

    return {
        "origin": origin,
        "destination": destination,
        "operator": operator,
        "count": len(trips),
        "trips": [_trip_to_dict(trip) for trip in trips],
    }


def execute_tool(
    name: str,
    arguments: dict[str, Any] | str | None,
    engine: RouteEngine,
    place_index: PlaceIndex,
) -> dict[str, Any]:
    """Dispatch a model tool call to its handler."""

    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments or "{}")
        except json.JSONDecodeError:
            return {
                "error": "invalid_arguments",
                "message": "The tool arguments were not valid JSON.",
            }
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        return {
            "error": "invalid_arguments",
            "message": "The tool arguments must be a JSON object.",
        }

    if name == "resolve_place":
        return resolve_place(place_index, arguments.get("query", ""))
    if name == "find_trips":
        return find_trips(
            engine,
            arguments.get("origin", ""),
            arguments.get("destination", ""),
            arguments.get("max_trips", 3),
            arguments.get("operator"),
        )
    return {
        "error": "unknown_tool",
        "message": f"Unknown tool {name!r}.",
        "available_tools": ["resolve_place", "find_trips"],
    }
