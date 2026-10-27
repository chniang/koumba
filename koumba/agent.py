"""The Koumba agent: Muse Spark plus our route tools.

The agent is deliberately small.  It sends the conversation to the model,
runs any requested tools against the real route engine, sends the tool
results back, and stops after at most five tool rounds.  Tests use a fake
client, so this module never calls the network.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .places import PlaceIndex, prefer_exact_matches
from .phrases import bus_check_spoken, spoken_answer
from .router import RouteEngine
from .tools import TOOL_SCHEMAS, execute_tool
from .vision import image_data_url

MAX_TOOL_ROUNDS = 5
MAX_HISTORY_MESSAGES = 6

SYSTEM_PROMPT = """You are Koumba, a bus assistant for Dakar.

The user may speak Wolof, French or English, often mixing them. Dakar place
names are often written in French or with Wolof spelling. For example,
"Maa ngi Sahm, bëgg naa dem Malika" means "I am at Sahm, I want to go to
Malika".

Always use the tools to resolve places and find trips. Never invent a bus
line, a stop, or a route. The tools are:
- resolve_place(query): find the best matching stops or places.
- find_trips(origin, destination, max_trips, operator): run the real Koumba
  route engine. Leave operator empty unless the user asks for a specific bus
  company.
- check_bus_line(operator, line, origin, destination): answer whether one bus
  can make the trip, with boarding and alighting stops or an alternative.
- line_info(operator, line): list a line's stops and directions.

Use the conversation history to understand follow-up answers. If the user
answers a clarification with just a place name, reuse the earlier origin or
destination.

When a photo is attached, read the operator and line number or sign first. If
they are unreadable, say so and ask for the number; never guess. If you can
read the bus and the destination is known, call check_bus_line. If the
destination is unknown, call line_info and help the user. If the origin is
unknown, ask where the user is. Answer a bus check with a clear yes or no.

If a tool returns an error, explain it briefly and ask the user to clarify.
Answer in the user's language using at most two short sentences. Never repeat
the user's sentence. Never list every trip in text; the app shows trip cards.
If a place is genuinely ambiguous, ask one short clarification question.
Otherwise give only the key boarding and alighting instruction."""

_FALLBACK_REPLIES = {
    "Wolof": "Baal ma, tool bi jotul. Jéemaat.",
    "French": "Désolé, la recherche n'a pas abouti. Réessayez.",
    "English": "Sorry, the search did not finish. Please try again.",
    "Unknown": "Sorry, the search did not finish. Please try again.",
}

_WOLOF_PHRASES = (
    "maa ngi",
    "bëgg",
    "begg",
    "beugg",
    "nekk",
    "jërëjëf",
    "jerejef",
    "fan la",
    "naka nga",
)
_WOLOF_WORDS = {
    "dama",
    "dem",
    "fii",
    "ci",
    "ak",
    "naa",
    "waaw",
    "déedéet",
    "deedeet",
    "sama",
    "yow",
    "naka",
    "leen",
    "nekk",
    "beugg",
    "bëgg",
    "bëgga",
    "jël",
    "jëm",
    "wàcci",
    "duggu",
    "nga",
}

_FRENCH_PHRASES = (
    "je suis",
    "je vais",
    "je veux",
    "je dois",
    "je cherche",
    "bonjour",
    "merci",
    "s'il vous",
    "où est",
    "ou est",
    "arrêt",
    "arret",
)
_FRENCH_WORDS = {
    "depuis",
    "vers",
    "au",
    "aux",
    "pour",
    "avec",
    "bus",
    "arrêt",
    "arret",
    "gare",
}

_ENGLISH_PHRASES = (
    "i am",
    "i'm",
    "i want",
    "i need",
    "i have",
    "take me",
    "how do i",
    "where is",
    "bus stop",
    "please",
)
_ENGLISH_WORDS = {
    "from",
    "to",
    "where",
    "how",
    "please",
    "thanks",
    "bus",
    "stop",
}


@dataclass
class AgentResult:
    """What the UI needs after one user turn."""

    reply: str
    language: str
    origin: str | None = None
    destination: str | None = None
    trips: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    rounds: int = 0
    spoken: str = ""


def _clean_history(history: Any) -> list[dict[str, str]]:
    """Keep the last six text-only user and assistant messages."""

    if not isinstance(history, (list, tuple)):
        return []

    cleaned: list[dict[str, str]] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")
        if role not in {"user", "assistant"}:
            continue
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned[-MAX_HISTORY_MESSAGES:]


def _history_language(history: list[dict[str, str]]) -> str:
    """Use the most recent user language when the current text has none."""

    for item in reversed(history):
        if item["role"] != "user":
            continue
        language = detect_language(item["content"])
        if language != "Unknown":
            return language
    return "Unknown"


def detect_language(text: str) -> str:
    """A small heuristic for the language badge in the demo.

    It is intentionally simple.  The model itself is told to answer in the
    user's language; this function only labels the turn.
    """

    lowered = (text or "").lower()
    tokens = set(re.findall(r"[a-zà-ÿ']+", lowered))

    if any(phrase in lowered for phrase in _WOLOF_PHRASES) or (
        tokens & _WOLOF_WORDS
    ):
        return "Wolof"
    if any(phrase in lowered for phrase in _FRENCH_PHRASES) or (
        tokens & _FRENCH_WORDS
    ):
        return "French"
    if any(phrase in lowered for phrase in _ENGLISH_PHRASES) or (
        tokens & _ENGLISH_WORDS
    ):
        return "English"
    return "Unknown"


def _field(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an SDK object or a plain dict."""

    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _message_payload(message: Any) -> dict[str, Any]:
    """Convert the assistant message back into the Chat Completions shape."""

    payload: dict[str, Any] = {
        "role": "assistant",
        "content": _field(message, "content"),
    }
    calls = _field(message, "tool_calls") or []
    if calls:
        payload["tool_calls"] = [_tool_call_payload(call) for call in calls]
    return payload


def _tool_call_payload(call: Any) -> dict[str, Any]:
    function = _field(call, "function")
    return {
        "id": _field(call, "id", ""),
        "type": _field(call, "type", "function"),
        "function": {
            "name": _field(function, "name", ""),
            "arguments": _field(function, "arguments", "{}") or "{}",
        },
    }


def _parse_tool_call(
    call: Any,
) -> tuple[str, str, dict[str, Any] | None, str | None]:
    """Return (id, name, arguments, parse_error)."""

    function = _field(call, "function")
    call_id = _field(call, "id", "") or ""
    name = _field(function, "name", "") or ""
    raw_arguments = _field(function, "arguments", "{}") or "{}"

    try:
        arguments = json.loads(raw_arguments)
    except (TypeError, json.JSONDecodeError):
        return call_id, name, None, "The tool arguments were not valid JSON."
    if not isinstance(arguments, dict):
        return call_id, name, None, "The tool arguments must be a JSON object."
    return call_id, name, arguments, None


def _response_message(response: Any) -> Any:
    choices = _field(response, "choices") or []
    if not choices:
        raise RuntimeError("The model returned no choices.")
    return _field(choices[0], "message")


class Agent:
    """Run one user turn through Muse Spark and the Koumba tools."""

    def __init__(
        self,
        client: Any,
        engine: RouteEngine,
        place_index: PlaceIndex | None = None,
        max_tool_rounds: int = MAX_TOOL_ROUNDS,
    ) -> None:
        self.client = client
        self.engine = engine
        self.place_index = place_index or engine.place_index
        self.max_tool_rounds = max(1, max_tool_rounds)

    def _official_place_name(self, query: str | None) -> str | None:
        text = str(query or "").strip()
        if not text:
            return None
        matches = prefer_exact_matches(self.place_index.match(text), text)
        return matches[0].name if matches else text

    def _result(
        self,
        *,
        reply: str,
        language: str,
        origin: str | None,
        destination: str | None,
        trips: list[dict[str, Any]] | None,
        tool_calls: list[dict[str, Any]],
        rounds: int,
        bus_check: dict[str, Any] | None = None,
    ) -> AgentResult:
        best_trip = trips[0] if trips else None
        spoken = ""
        if bus_check is not None:
            alternative = bus_check.get("best_alternative")
            spoken = bus_check_spoken(
                bus_check,
                language,
                destination,
                alternative if isinstance(alternative, dict) else None,
                origin,
            )
        if not spoken:
            spoken = spoken_answer(best_trip, language, origin)
        return AgentResult(
            reply=reply,
            language=language,
            spoken=spoken,
            origin=origin,
            destination=destination,
            trips=trips,
            tool_calls=tool_calls,
            rounds=rounds,
        )

    def run(
        self,
        user_text: str,
        history: list[dict[str, Any]] | None = None,
        image_path: str | None = None,
    ) -> AgentResult:
        text = (user_text or "").strip()
        memory = _clean_history(history)
        language = detect_language(text)
        if language == "Unknown":
            language = _history_language(memory)
        user_content: str | list[dict[str, Any]] = text
        if image_path:
            user_content = [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_url(image_path)},
                },
            ]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *memory,
            {"role": "user", "content": user_content},
        ]

        tool_calls_made: list[dict[str, Any]] = []
        origin: str | None = None
        destination: str | None = None
        trips: list[dict[str, Any]] | None = None
        bus_check: dict[str, Any] | None = None
        rounds_used = 0

        for round_number in range(1, self.max_tool_rounds + 1):
            rounds_used = round_number
            try:
                response = self.client.complete(
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=0.2,
                )
                message = _response_message(response)
            except Exception:
                return self._result(
                    reply=_FALLBACK_REPLIES.get(
                        language, _FALLBACK_REPLIES["Unknown"]
                    ),
                    language=language,
                    origin=origin,
                    destination=destination,
                    trips=trips,
                    tool_calls=tool_calls_made,
                    rounds=rounds_used,
                    bus_check=bus_check,
                )

            messages.append(_message_payload(message))
            model_calls = _field(message, "tool_calls") or []
            if not model_calls:
                reply = str(_field(message, "content") or "").strip()
                if not reply:
                    reply = _FALLBACK_REPLIES.get(
                        language, _FALLBACK_REPLIES["Unknown"]
                    )
                return self._result(
                    reply=reply,
                    language=language,
                    origin=origin,
                    destination=destination,
                    trips=trips,
                    tool_calls=tool_calls_made,
                    rounds=rounds_used,
                    bus_check=bus_check,
                )

            for call in model_calls:
                call_id, name, arguments, parse_error = _parse_tool_call(call)
                if not call_id:
                    call_id = f"call_{round_number}_{len(tool_calls_made)}"

                if parse_error is not None:
                    result: dict[str, Any] = {
                        "error": "invalid_arguments",
                        "message": parse_error,
                    }
                else:
                    result = execute_tool(
                        name,
                        arguments or {},
                        self.engine,
                        self.place_index,
                    )

                safe_arguments = arguments if isinstance(arguments, dict) else {}
                tool_calls_made.append(
                    {
                        "round": round_number,
                        "name": name,
                        "arguments": safe_arguments,
                        "result": result,
                    }
                )

                if name == "find_trips":
                    origin = self._official_place_name(
                        safe_arguments.get("origin")
                    ) or origin
                    destination = self._official_place_name(
                        safe_arguments.get("destination")
                    ) or destination
                    found = result.get("trips")
                    trips = found if isinstance(found, list) else None
                elif name == "check_bus_line":
                    origin = self._official_place_name(
                        safe_arguments.get("origin")
                    ) or origin
                    destination = self._official_place_name(
                        safe_arguments.get("destination")
                    ) or destination
                    if isinstance(result.get("serves_trip"), bool):
                        bus_check = result

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": name,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        return self._result(
            reply=_FALLBACK_REPLIES.get(language, _FALLBACK_REPLIES["Unknown"]),
            language=language,
            origin=origin,
            destination=destination,
            trips=trips,
            tool_calls=tool_calls_made,
            rounds=rounds_used,
            bus_check=bus_check,
        )
