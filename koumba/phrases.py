"""Short, deterministic spoken answers for the best Koumba trip.

The Wolof templates in this module were supplied by the project owner for a
native-speaker review.  Keep their wording and punctuation unchanged.  The
UI uses these sentences instead of the model's longer explanation.
"""

from __future__ import annotations

import math
import re
from typing import Any

SUPPORTED_LANGUAGES = ("Wolof", "French", "English")


def _language_name(language: str) -> str:
    lowered = (language or "").strip().lower()
    if lowered == "wolof":
        return "Wolof"
    if lowered == "french" or lowered == "français" or lowered == "francais":
        return "French"
    return "English"


def round_minutes(value: Any) -> int:
    """Round a trip duration to the nearest five minutes."""

    try:
        minutes = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(minutes):
        return 0
    return max(0, int(math.floor(minutes / 5.0 + 0.5) * 5))


def _line_number(line: Any) -> str:
    text = str(line or "").strip()
    match = re.search(r"\d+[A-Za-z]?", text)
    if match:
        return match.group(0)
    return re.sub(r"(?i)\bligne\b", "", text).strip()


def short_bus_name(operator: Any, line: Any) -> str:
    """Return the short name used in speech, such as ``Tata 38``."""

    operator_text = str(operator or "").strip()
    number = _line_number(line)
    lowered = operator_text.lower()
    if "dem dikk" in lowered:
        return f"Dem Dikk {number}".strip()
    if "tata" in lowered:
        return f"Tata {number}".strip()
    return f"{operator_text} {number}".strip()


def _leg_value(leg: dict[str, Any], name: str) -> str:
    return str(leg.get(name) or "").strip()


def _leg_speech(leg: dict[str, Any]) -> tuple[str, str, str] | None:
    bus = short_bus_name(
        _leg_value(leg, "operator"),
        _leg_value(leg, "line"),
    )
    direction = _leg_value(leg, "direction")
    alight = _leg_value(leg, "alight_stop")
    if not bus or not direction or not alight:
        return None
    return bus, direction, alight


def spoken_answer(
    trip: dict[str, Any] | None,
    language: str,
    origin: str | None = None,
) -> str:
    """Build the short best-trip sentence in the detected language."""

    if not isinstance(trip, dict):
        return ""
    legs = trip.get("legs")
    if not isinstance(legs, list) or not legs:
        return ""

    speech = []
    for leg in legs:
        if not isinstance(leg, dict):
            return ""
        item = _leg_speech(leg)
        if item is None:
            return ""
        speech.append(item)

    origin_text = str(origin or "").strip()
    if not origin_text:
        origin_text = _leg_value(legs[0], "board_stop")
    if not origin_text:
        return ""

    minutes = round_minutes(trip.get("total_minutes"))
    selected = _language_name(language)

    if selected == "Wolof":
        first_bus, first_direction, first_alight = speech[0]
        if len(speech) == 1:
            sentence = (
                f"Ci {origin_text}, duggu ci {first_bus} bi jëm "
                f"{first_direction}, wàcci {first_alight}."
            )
        else:
            sentence = (
                f"Ci {origin_text}, duggu ci {first_bus} bi jëm "
                f"{first_direction}, wàcci {first_alight}"
            )
            for bus, direction, alight in speech[1:]:
                sentence += (
                    f", nga jël {bus} bi jëm {direction}, wàcci {alight}"
                )
            sentence += "."
        return f"{sentence} Lu tollu ci {minutes} minit."

    if selected == "French":
        first_bus, first_direction, first_alight = speech[0]
        if len(speech) == 1:
            sentence = (
                f"À {origin_text}, prends le {first_bus} direction "
                f"{first_direction}, descends à {first_alight}."
            )
        else:
            sentence = (
                f"À {origin_text}, prends le {first_bus} direction "
                f"{first_direction}, descends à {first_alight}"
            )
            for bus, direction, alight in speech[1:]:
                sentence += (
                    f", puis prends le {bus} direction {direction}, "
                    f"descends à {alight}"
                )
            sentence += "."
        return f"{sentence} Environ {minutes} minutes."

    first_bus, first_direction, first_alight = speech[0]
    if len(speech) == 1:
        sentence = (
            f"From {origin_text}, take {first_bus} toward "
            f"{first_direction}, get off at {first_alight}."
        )
    else:
        sentence = (
            f"From {origin_text}, take {first_bus} toward "
            f"{first_direction}, get off at {first_alight}"
        )
        for bus, direction, alight in speech[1:]:
            sentence += (
                f", then take {bus} toward {direction}, get off at {alight}"
            )
        sentence += "."
    return f"{sentence} About {minutes} minutes."


def clarification_question(
    option_a: str,
    option_b: str,
    language: str = "Wolof",
) -> str:
    """Ask which of two genuinely ambiguous places the user meant."""

    first = str(option_a or "").strip()
    second = str(option_b or "").strip()
    if not first or not second:
        return ""

    selected = _language_name(language)
    if selected == "Wolof":
        return f"{first} nga bëgga dem walla {second} ?"
    if selected == "French":
        return f"Tu veux aller à {first} ou à {second} ?"
    return f"Do you want {first} or {second}?"


def _first_question(reply: str) -> str:
    match = re.search(r"[^.!?\n]*\?", reply or "")
    return match.group(0).strip() if match else ""


def compose_chat_message(spoken: str, model_reply: str) -> str:
    """Use the spoken sentence, adding only a model clarification question."""

    spoken_text = str(spoken or "").strip()
    reply_text = str(model_reply or "").strip()
    if not spoken_text:
        return reply_text

    question = _first_question(reply_text)
    if question and question.lower() not in spoken_text.lower():
        return f"{spoken_text} {question}".strip()
    return spoken_text
