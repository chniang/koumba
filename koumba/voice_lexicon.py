"""Shared ids and Conservative spelling helpers for the voice lexicon.

The CSV in ``data/voice_lexicon.csv`` is generated ahead of time.  At runtime we
only need the ids: the same connector, bus, or place must produce the same id
in the generator and in ``koumba.phrases`` or the audio segments will not line
up.

The place suggestion is deliberately mechanical.  It applies only the letter
conversions below and never adds or removes a vowel.  Vowel length changes the
meaning in Wolof, so those decisions stay with a native speaker.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

# Every fixed Wolof group used by the phrase templates.  "Duggal ci" in the
# bus-check answer uses the same audio as "duggal ci"; only the capital letter
# comes from sentence position, so it is not a second lexicon row.
CONNECTOR_TEXTS = (
    "Yaangi",
    "doxal ba",
    "duggal ci",
    "bi jëm",
    "wàcci",
    "nga jël",
    "Waaw",
    "Déedéet",
    "bi dafay dem",
    "bi du dem",
    "nga bëgga dem walla",
)

# Longest / most specific patterns first.  In particular, "tch" is handled
# before the French-looking "ch" marker is checked.
_SAFE_REPLACEMENTS = (
    ("tch", "c"),
    ("ou", "u"),
    ("dj", "j"),
    ("kh", "x"),
    ("gn", "ñ"),
    ("eu", "ë"),
    ("è", "e"),
)

# Marks that a place probably contains French-influenced spelling.  "ch" is a
# review trigger but has no automatic replacement here: we do not know whether
# the intended Wolof sound is "c", "s", or something else.
_FRENCH_MARKERS = ("ou", "dj", "kh", "gn", "è", "eu", "tch", "ch")


def slug(text: Any) -> str:
    """Make a stable ASCII id fragment without changing the display text."""

    folded = (
        unicodedata.normalize("NFKD", str(text or ""))
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    return re.sub(r"[^a-z0-9]+", "-", folded).strip("-") or "x"


def connector_id(text: Any) -> str:
    """Id for a fixed phrase group, independent of sentence capitalization."""

    return f"connector:{slug(text)}"


def bus_id(operator: Any, line: Any) -> str:
    """Id for one operator/line pair as it appears in a trip leg."""

    return f"bus:{slug(operator or 'unknown')}:{slug(line or 'unknown')}"


def place_id(text: Any) -> str:
    """Id for an exact stop or place spelling.

    The short hash keeps ``Sahm`` and ``SAHM`` separate.  They are the same
    physical stop, but the dataset can display either form and the audio cache
    should be able to tell which one was reviewed.
    """

    name = str(text or "").strip()
    digest = hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]
    return f"place:{slug(name)}-{digest}"


def _match_case(match: re.Match[str], replacement: str) -> str:
    """Apply a replacement without turning a capitalized name lowercase."""

    matched = match.group(0)
    if matched.isupper():
        return replacement.upper()
    if matched[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def safe_place_suggestion(text: Any) -> str:
    """Offer a safer Wolof spelling using only approved letter conversions.

    This is a suggestion for human review, not an automatic ``say_as`` value.
    In particular, vowel length is left exactly as it was in the source name.
    """

    result = unicodedata.normalize("NFC", str(text or ""))
    for old, new in _SAFE_REPLACEMENTS:
        result = re.sub(
            re.escape(old),
            lambda match, replacement=new: _match_case(match, replacement),
            result,
            flags=re.IGNORECASE,
        )
    return result


def place_needs_review(text: Any) -> bool:
    """True when the spelling contains a French-influenced pattern."""

    lowered = unicodedata.normalize("NFC", str(text or "")).lower()
    return any(marker in lowered for marker in _FRENCH_MARKERS)


def apply_word_overrides(
    text: Any,
    overrides: Mapping[str, str],
) -> str:
    """Replace reviewed whole words while leaving the rest of the name alone.

    Matching is case-insensitive, but the replacement is copied exactly from
    the CSV.  The lookarounds make ``Fass`` match ``Fass Mbao`` and
    ``Mbédou Fass`` without touching the made-up name ``Fassa``.
    """

    result = str(text or "")
    for word, say_as in overrides.items():
        if not word:
            continue
        pattern = rf"(?<!\w){re.escape(word)}(?!\w)"
        result = re.sub(
            pattern,
            lambda _match, replacement=say_as: replacement,
            result,
            flags=re.IGNORECASE,
        )
    return result
