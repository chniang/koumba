"""Tool schemas and handlers exposed to the Muse Spark agent.

The handlers always return plain dictionaries.  A bad place name or a route
with no connection becomes an ``{"error": ...}`` result instead of an
exception, so a model tool round can recover and ask the user a question.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from .network import Line, LineKey
from .places import PlaceIndex, normalize_name
from .router import NoRouteFoundError, PlaceNotFoundError, RouteEngine, Trip

OPERATORS = ("Dakar Dem Dikk", "Tata AFTU")
MAX_TOOL_TRIPS = 5
MAX_CLOSEST_LINES = 5


@dataclass(frozen=True)
class _ResolvedLine:
    line: Line
    key: LineKey

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
    {
        "type": "function",
        "function": {
            "name": "check_bus_line",
            "description": (
                "Answer whether one bus line can take the user from origin to "
                "destination. Use this for a bus photo or a question such as "
                "'Is this my bus?'. It returns board and alight stops when the "
                "line works directly, or the best alternative and whether the "
                "line helps part of the way."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operator": {
                        "type": "string",
                        "description": (
                            "Bus company, for example Tata AFTU or "
                            "Dakar Dem Dikk."
                        ),
                    },
                    "line": {
                        "type": "string",
                        "description": (
                            "Line number or name from the bus, for example "
                            "38, Ligne 38, or TAF TAF."
                        ),
                    },
                    "origin": {
                        "type": "string",
                        "description": "Where the user is starting.",
                    },
                    "destination": {
                        "type": "string",
                        "description": "Where the user wants to go.",
                    },
                },
                "required": ["operator", "line", "origin", "destination"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "line_info",
            "description": (
                "List the ordered stops and both directions for one bus line. "
                "Use this after reading a bus number from a photo when the "
                "destination is not known yet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operator": {
                        "type": "string",
                        "description": "Bus company for the line.",
                    },
                    "line": {
                        "type": "string",
                        "description": "Line number or name to look up.",
                    },
                },
                "required": ["operator", "line"],
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


def normalize_operator(value: Any) -> str | None:
    """Turn common operator spellings into the dataset's two names."""

    if value is None:
        return None
    compact = normalize_name(value).replace(" ", "")
    if not compact or compact in {"null", "none", "any", "all"}:
        return None
    if compact in {"tata", "aftu", "tataaftu"}:
        return "Tata AFTU"
    if compact in {"demdikk", "ddd", "dakardemdikk"}:
        return "Dakar Dem Dikk"
    return None


def _line_alias(value: Any) -> str:
    tokens = normalize_name(value).split()
    while tokens and tokens[0] in {"l", "ligne", "line", "bus", "numero", "no"}:
        tokens.pop(0)
    text = " ".join(tokens).strip()
    match = re.search(r"\d+[a-z]?", text)
    return match.group(0) if match else text


def _closest_lines(
    engine: RouteEngine,
    operator: str,
    query: str,
    limit: int = MAX_CLOSEST_LINES,
) -> list[str]:
    target = _line_alias(query) or normalize_name(query)
    scored: list[tuple[float, str]] = []
    for (company, _), line in engine.network.lines.items():
        if company != operator:
            continue
        score = max(
            fuzz.WRatio(target, _line_alias(line.name)),
            fuzz.WRatio(normalize_name(query), normalize_name(line.name)),
        )
        scored.append((score, line.name))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [name for _, name in scored[:limit]]


def _resolve_line(
    engine: RouteEngine,
    operator: Any,
    line: Any,
) -> tuple[_ResolvedLine | None, dict[str, Any] | None]:
    raw_operator = str(operator or "").strip()
    raw_line = str(line or "").strip()
    normalized_operator = normalize_operator(raw_operator)
    if normalized_operator is None:
        return None, {
            "error": "invalid_operator",
            "message": (
                "The operator must be Dakar Dem Dikk or Tata AFTU."
            ),
            "allowed_operators": list(OPERATORS),
        }
    if not raw_line:
        return None, {
            "error": "line_not_found",
            "operator": normalized_operator,
            "line": raw_line,
            "message": "The bus line is empty.",
            "closest_lines": _closest_lines(
                engine, normalized_operator, raw_line
            ),
        }

    target = _line_alias(raw_line)
    target_norm = normalize_name(raw_line)
    for key, line_object in engine.network.lines.items():
        if key[0] != normalized_operator:
            continue
        if (
            _line_alias(line_object.name) == target
            or normalize_name(line_object.name) == target_norm
        ):
            return _ResolvedLine(line=line_object, key=key), None

    closest = _closest_lines(engine, normalized_operator, raw_line)
    return None, {
        "error": "line_not_found",
        "operator": normalized_operator,
        "line": raw_line,
        "message": (
            f"I could not find {raw_line!r} for {normalized_operator}."
        ),
        "closest_lines": closest,
    }


def _line_info_dict(line: Line) -> dict[str, Any]:
    return {
        "operator": line.company,
        "line": line.name,
        "stops": list(line.stops),
        "terminus_a": line.terminus_a or line.stops[0],
        "terminus_b": line.terminus_b or line.stops[-1],
        "directions": [
            {
                "from": line.stops[0],
                "towards": line.terminus_for(0),
                "stops": list(line.stops),
            },
            {
                "from": line.stops[-1],
                "towards": line.terminus_for(1),
                "stops": list(line.stops_for(1)),
            },
        ],
    }


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
        raw_operator = str(operator).strip()
        operator = normalize_operator(raw_operator)
        if operator is None and raw_operator.lower() not in {
            "",
            "null",
            "none",
            "any",
            "all",
        }:
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


def check_bus_line(
    engine: RouteEngine,
    operator: str,
    line: str,
    origin: str,
    destination: str,
) -> dict[str, Any]:
    """Check whether one bus line can make the requested trip."""

    origin = str(origin or "").strip()
    destination = str(destination or "").strip()
    if not origin or not destination:
        return {
            "error": "invalid_request",
            "message": "Both an origin and a destination are needed.",
        }

    resolved, error = _resolve_line(engine, operator, line)
    if error is not None or resolved is None:
        return error or {
            "error": "line_not_found",
            "message": "I could not resolve that bus line.",
        }

    try:
        requested_trips = engine.find_trips(
            origin,
            destination,
            max_trips=MAX_TOOL_TRIPS,
            first_line=resolved.key,
        )
    except PlaceNotFoundError as exc:
        return {
            "error": "place_not_found",
            "query": exc.query,
            "message": f"I could not find {exc.query!r} in the Dakar stop data.",
            "suggestions": _suggestions(exc.suggestions),
        }
    except NoRouteFoundError:
        requested_trips = []
    except Exception as exc:
        return {
            "error": "tool_error",
            "message": "The route engine could not complete this search.",
            "detail": str(exc),
        }

    direct = next(
        (trip for trip in requested_trips if trip.transfers == 0),
        None,
    )
    if direct is not None:
        leg = direct.legs[0]
        return {
            "serves_trip": True,
            "operator": resolved.key[0],
            "line": resolved.key[1],
            "board_stop": leg.board_stop,
            "alight_stop": leg.alight_stop,
            "direction": leg.direction,
            "minutes": _round(direct.total_minutes),
            "trip": _trip_to_dict(direct),
        }

    try:
        alternatives = engine.find_trips(
            origin,
            destination,
            max_trips=1,
        )
    except NoRouteFoundError:
        alternatives = []
    except PlaceNotFoundError as exc:
        return {
            "error": "place_not_found",
            "query": exc.query,
            "message": f"I could not find {exc.query!r} in the Dakar stop data.",
            "suggestions": _suggestions(exc.suggestions),
        }
    except Exception as exc:
        return {
            "error": "tool_error",
            "message": "The route engine could not complete this search.",
            "detail": str(exc),
        }

    best_alternative = (
        _trip_to_dict(alternatives[0]) if alternatives else None
    )
    return {
        "serves_trip": False,
        "operator": resolved.key[0],
        "line": resolved.key[1],
        "best_alternative": best_alternative,
        "can_help_part_way": bool(requested_trips),
    }


def line_info(
    engine: RouteEngine,
    operator: str,
    line: str,
) -> dict[str, Any]:
    """Return one line's ordered stops and directions."""

    resolved, error = _resolve_line(engine, operator, line)
    if error is not None or resolved is None:
        return error or {
            "error": "line_not_found",
            "message": "I could not resolve that bus line.",
        }
    return _line_info_dict(resolved.line)


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
    if name == "check_bus_line":
        return check_bus_line(
            engine,
            arguments.get("operator", ""),
            arguments.get("line", ""),
            arguments.get("origin", ""),
            arguments.get("destination", ""),
        )
    if name == "line_info":
        return line_info(
            engine,
            arguments.get("operator", ""),
            arguments.get("line", ""),
        )
    return {
        "error": "unknown_tool",
        "message": f"Unknown tool {name!r}.",
        "available_tools": [
            "resolve_place",
            "find_trips",
            "check_bus_line",
            "line_info",
        ],
    }
