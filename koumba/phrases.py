"""Short, deterministic spoken answers for the best Koumba trip.

The templates were reviewed by a native speaker.  They deliberately contain no
travel-time estimate: Dakar traffic makes minutes misleading in speech.  The
UI uses these sentences instead of the model's longer explanation, while the
route JSON keeps minutes for ranking and card hints.
"""

from __future__ import annotations

import math
import re
from typing import Any

from .places import normalize_name
from .voice_lexicon import bus_id, connector_id, place_id

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


def _walk_to_first_stop(
    trip: dict[str, Any],
    first_leg: dict[str, Any],
    origin_text: str,
) -> str | None:
    """Return the boarding stop when the user must walk at least a minute."""

    board_stop = _leg_value(first_leg, "board_stop")
    if not board_stop:
        return None
    if normalize_name(board_stop) == normalize_name(origin_text):
        return None

    walks = trip.get("walks")
    if not isinstance(walks, list):
        return None
    for walk in walks:
        if not isinstance(walk, dict) or walk.get("kind") != "access":
            continue
        try:
            after_leg = int(walk.get("after_leg") or 0)
            minutes = float(walk.get("minutes") or 0)
        except (TypeError, ValueError):
            continue
        if after_leg != 0:
            continue
        if minutes < 1:
            continue
        to_place = str(walk.get("to_place") or "").strip()
        if to_place and normalize_name(to_place) != normalize_name(board_stop):
            continue
        return board_stop
    return None


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

    selected = _language_name(language)
    board_stop = _walk_to_first_stop(trip, legs[0], origin_text)

    if selected == "Wolof":
        first_bus, first_direction, first_alight = speech[0]
        if board_stop:
            opening = (
                f"Yaangi {origin_text}, doxal ba {board_stop}, "
                f"duggal ci {first_bus} bi jëm "
            )
        else:
            opening = f"Yaangi {origin_text}, duggal ci {first_bus} bi jëm "
        if len(speech) == 1:
            sentence = (
                f"{opening}{first_direction}, wàcci {first_alight}."
            )
        else:
            sentence = (
                f"{opening}{first_direction}, wàcci {first_alight}"
            )
            for bus, direction, alight in speech[1:]:
                sentence += (
                    f", nga jël {bus} bi jëm {direction}, wàcci {alight}"
                )
            sentence += "."
        return sentence

    if selected == "French":
        first_bus, first_direction, first_alight = speech[0]
        if board_stop:
            opening = (
                f"Tu es à {origin_text} : marche jusqu'à {board_stop}, "
                f"prends le {first_bus} direction {first_direction}, "
                f"descends à {first_alight}"
            )
        else:
            opening = (
                f"À {origin_text}, prends le {first_bus} direction "
                f"{first_direction}, descends à {first_alight}"
            )
        if len(speech) == 1:
            sentence = f"{opening}."
        else:
            sentence = opening
            for bus, direction, alight in speech[1:]:
                sentence += (
                    f", puis prends le {bus} direction {direction}, "
                    f"descends à {alight}"
                )
            sentence += "."
        return sentence

    first_bus, first_direction, first_alight = speech[0]
    if board_stop:
        opening = (
            f"You are at {origin_text}: walk to {board_stop}, take "
            f"{first_bus} toward {first_direction}, get off at "
            f"{first_alight}"
        )
    else:
        opening = (
            f"From {origin_text}, take {first_bus} toward "
            f"{first_direction}, get off at {first_alight}"
        )
    if len(speech) == 1:
        sentence = f"{opening}."
    else:
        sentence = opening
        for bus, direction, alight in speech[1:]:
            sentence += (
                f", then take {bus} toward {direction}, get off at {alight}"
            )
        sentence += "."
    return sentence


def bus_check_spoken(
    check: dict[str, Any] | None,
    language: str,
    destination: str | None = None,
    alternative: dict[str, Any] | None = None,
    origin: str | None = None,
) -> str:
    """Answer "is this my bus?" with the validated short wording."""

    if not isinstance(check, dict):
        return ""
    serves = check.get("serves_trip")
    if not isinstance(serves, bool):
        return ""

    bus = short_bus_name(check.get("operator"), check.get("line"))
    destination_text = str(
        destination or check.get("destination") or ""
    ).strip()
    if not bus or not destination_text:
        return ""

    selected = _language_name(language)
    if serves:
        alight = str(check.get("alight_stop") or "").strip()
        if not alight:
            return ""
        if selected == "Wolof":
            return (
                f"Waaw, {bus} bi dafay dem {destination_text}. "
                f"Duggal ci, wàcci {alight}."
            )
        if selected == "French":
            return (
                f"Oui, le {bus} va à {destination_text}. Monte et descends "
                f"à {alight}."
            )
        return (
            f"Yes, {bus} goes to {destination_text}. Board it and get off "
            f"at {alight}."
        )

    if selected == "Wolof":
        answer = f"Déedéet, {bus} bi du dem {destination_text}."
    elif selected == "French":
        answer = f"Non, le {bus} ne va pas à {destination_text}."
    else:
        answer = f"No, {bus} does not go to {destination_text}."

    follow_up = spoken_answer(alternative, language, origin)
    return f"{answer} {follow_up}".strip()


def _trip_segments(
    trip: dict[str, Any] | None,
    origin: str | None,
) -> list[str]:
    """Lexicon ids for the Wolof best-trip sentence, in speech order."""

    if not isinstance(trip, dict):
        return []
    legs = trip.get("legs")
    if not isinstance(legs, list) or not legs:
        return []

    speech = []
    for leg in legs:
        if not isinstance(leg, dict):
            return []
        item = _leg_speech(leg)
        if item is None:
            return []
        speech.append(item)

    origin_text = str(origin or "").strip()
    if not origin_text:
        origin_text = _leg_value(legs[0], "board_stop")
    if not origin_text:
        return []

    board_stop = _walk_to_first_stop(trip, legs[0], origin_text)
    parts = [connector_id("Yaangi"), place_id(origin_text)]
    if board_stop:
        parts.extend([connector_id("doxal ba"), place_id(board_stop)])

    _, first_direction, first_alight = speech[0]
    first_leg = legs[0]
    parts.extend(
        [
            connector_id("duggal ci"),
            bus_id(
                _leg_value(first_leg, "operator"),
                _leg_value(first_leg, "line"),
            ),
            connector_id("bi jëm"),
            place_id(first_direction),
            connector_id("wàcci"),
            place_id(first_alight),
        ]
    )

    # Each transfer repeats the short "take ... go to ... get off" pattern.
    for leg, (_bus, direction, alight) in zip(legs[1:], speech[1:]):
        parts.extend(
            [
                connector_id("nga jël"),
                bus_id(
                    _leg_value(leg, "operator"),
                    _leg_value(leg, "line"),
                ),
                connector_id("bi jëm"),
                place_id(direction),
                connector_id("wàcci"),
                place_id(alight),
            ]
        )
    return parts


def _bus_check_segments(
    check: dict[str, Any] | None,
    destination: str | None,
    alternative: dict[str, Any] | None,
    origin: str | None,
) -> list[str]:
    """Lexicon ids for the Wolof bus-check answer, including any alternative."""

    if not isinstance(check, dict):
        return []
    serves = check.get("serves_trip")
    if not isinstance(serves, bool):
        return []

    bus = short_bus_name(check.get("operator"), check.get("line"))
    destination_text = str(
        destination or check.get("destination") or ""
    ).strip()
    if not bus or not destination_text:
        return []

    bus_lexicon_id = bus_id(check.get("operator"), check.get("line"))
    if serves:
        alight = str(check.get("alight_stop") or "").strip()
        if not alight:
            return []
        return [
            connector_id("Waaw"),
            bus_lexicon_id,
            connector_id("bi dafay dem"),
            place_id(destination_text),
            connector_id("duggal ci"),
            connector_id("wàcci"),
            place_id(alight),
        ]

    parts = [
        connector_id("Déedéet"),
        bus_lexicon_id,
        connector_id("bi du dem"),
        place_id(destination_text),
    ]
    parts.extend(_trip_segments(alternative, origin))
    return parts


def spoken_segments(
    trip: dict[str, Any] | None = None,
    language: str = "Wolof",
    origin: str | None = None,
    *,
    check: dict[str, Any] | None = None,
    destination: str | None = None,
    alternative: dict[str, Any] | None = None,
) -> list[str]:
    """Return lexicon ids in exactly the order of the short spoken answer.

    Call it like :func:`spoken_answer` for a trip, for example
    ``spoken_segments(trip, "Wolof", "Sahm")``.  For a bus check, pass the
    check and optional alternative by keyword:

    ``spoken_segments(check=check, language="Wolof", destination="PAI")``.

    Only Wolof is segmented; French and English return an empty list.
    """

    if _language_name(language) != "Wolof":
        return []
    if check is not None:
        return _bus_check_segments(check, destination, alternative, origin)
    return _trip_segments(trip, origin)


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
