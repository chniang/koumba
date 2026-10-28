"""Wolof number words used in bus-line names.

The spellings below follow the official Senegalese Wolof orthography.  A
doubled vowel is long and a single vowel is short, so forms such as ``ñaar``,
``ñeent``, and ``juróom`` must not be "simplified" later.

We only cover 1..199 on purpose.  That is the validated range for bus numbers;
larger numbers need a native-speaker decision instead of a guessed form.
"""

from __future__ import annotations

MIN_NUMBER = 1
MAX_NUMBER = 199

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
    """Return the Wolof words for an integer from 1 to 199.

    Examples: 38 -> ``fanweer ak juróom ñett``, 105 -> ``téeméer ak juróom``.
    """

    if isinstance(number, bool) or not isinstance(number, int):
        raise TypeError("Wolof numbers must be integers")
    if not MIN_NUMBER <= number <= MAX_NUMBER:
        raise ValueError(
            f"Wolof numbers are defined only from {MIN_NUMBER} to {MAX_NUMBER}"
        )

    if number < 100:
        return _below_hundred(number)
    if number == 100:
        return "téeméer"
    return f"téeméer ak {_below_hundred(number - 100)}"
