"""Wolof number words stay within the validated 1..599 vocabulary."""

from __future__ import annotations

import pytest

from koumba.wolof_numbers import wolof_number


@pytest.mark.parametrize(
    ("number", "expected"),
    [
        (1, "benn"),
        (2, "ñaar"),
        (5, "juróom"),
        (6, "juróom benn"),
        (9, "juróom ñeent"),
        (10, "fukk"),
        (11, "fukk ak benn"),
        (20, "ñaar fukk"),
        (23, "ñaar fukk ak ñett"),
        (30, "fanweer"),
        (31, "fanweer ak benn"),
        (38, "fanweer ak juróom ñett"),
        (40, "ñeent fukk"),
        (60, "juróom benn fukk"),
        (90, "juróom ñeent fukk"),
        (100, "téeméer"),
        (101, "téeméer ak benn"),
        (105, "téeméer ak juróom"),
        (110, "téeméer ak fukk"),
        (120, "téeméer ak ñaar fukk"),
        (130, "téeméer ak fanweer"),
        (138, "téeméer ak fanweer ak juróom ñett"),
        (199, "téeméer ak juróom ñeent fukk ak juróom ñeent"),
        (200, "ñaari téeméer"),
        (208, "ñaari téeméer ak juróom ñett"),
        (213, "ñaari téeméer ak fukk ak ñett"),
        (300, "ñetti téeméer"),
        (400, "ñeenti téeméer"),
        (500, "juróomi téeméer"),
        (501, "juróomi téeméer ak benn"),
        (502, "juróomi téeméer ak ñaar"),
        (509, "juróomi téeméer ak juróom ñeent"),
        (510, "juróomi téeméer ak fukk"),
        (599, "juróomi téeméer ak juróom ñeent fukk ak juróom ñeent"),
    ],
)
def test_wolof_number_words(number: int, expected: str) -> None:
    assert wolof_number(number) == expected


@pytest.mark.parametrize("number", [0, 600, -1])
def test_numbers_outside_the_validated_range_are_rejected(number: int) -> None:
    with pytest.raises(ValueError):
        wolof_number(number)


@pytest.mark.parametrize("value", [True, "38", 38.0])
def test_non_integer_values_are_rejected(value: object) -> None:
    with pytest.raises(TypeError):
        wolof_number(value)  # type: ignore[arg-type]
