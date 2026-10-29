"""Wolof number words used in bus-line names.

The spellings below follow the official Senegalese Wolof orthography.  A
doubled vowel is long and a single vowel is short, so forms such as ``ñaar``,
``ñeent``, and ``juróom`` must not be "simplified" later.

We cover 1..599 only.  The hundred words and their "agreement" forms are
validated; anything larger still needs a native-speaker decision.
"""

from __future__ import annotations

MIN_NUMBER = 1
MAX_NUMBER = 599

_ONES = {
    1: "benn",
    2: "ñaar",
    3: "ñett",
    4: "ñeent",
    5: "juróom",
    6: "juróom benn",
    7: "juróom ñaar",
    8: "juróom ñett",
    9: "juróom ñeent",
}

# Thirty is the irregular one: Wolof says "fanweer", not "ñett fukk".
_TENS = {
    20: "ñaar fukk",
    30: "fanweer",
    40: "ñeent fukk",
    50: "juróom fukk",
    60: "juróom benn fukk",
    70: "juróom ñaar fukk",
    80: "juróom ñett fukk",
    90: "juróom ñeent fukk",
}

# "One hundred" keeps the plain noun form; 200..500 add the matching spelled
# agreement before the same noun.  Do not regenerate these from morphology:
# they were checked with the Oolel voice.
_HUNDREDS = {
    1: "téeméer",
    2: "ñaari téeméer",
    3: "ñetti téeméer",
    4: "ñeenti téeméer",
    5: "juróomi téeméer",
}


def _below_hundred(number: int) -> str:
    """Build 1..99, joining the remainder with the spoken word ``ak``."""

    if number < 10:
        return _ONES[number]
    if number == 10:
        return "fukk"
    if number < 20:
        return f"fukk ak {_ONES[number - 10]}"

    tens = (number // 10) * 10
    remainder = number % 10
    words = _TENS[tens]
    if remainder:
        words += f" ak {_ONES[remainder]}"
    return words


def wolof_number(number: int) -> str:
    """Return the Wolof words for an integer from 1 to 599.

    Examples: 38 -> ``fanweer ak juróom ñett``, 208 -> ``ñaari téeméer ak
    juróom ñett``.
    """

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError("Wolof numbers must be integers")
    if not MIN_NUMBER <= number <= MAX_NUMBER:
        raise ValueError(
            f"Wolof numbers are defined only from {MIN_NUMBER} to {MAX_NUMBER}"
        )

    if number < 100:
        return _below_hundred(number)

    hundreds, remainder = divmod(number, 100)
    words = _HUNDREDS[hundreds]
    if remainder:
        words += f" ak {_below_hundred(remainder)}"
    return words
