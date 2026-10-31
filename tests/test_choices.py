"""Clarification answers must be understood without relying on the model."""

from __future__ import annotations

import pytest

from koumba.choices import match_choice

OPTIONS = ["Palais 1", "Palais 2"]


@pytest.mark.parametrize(
    "answer",
    [
        "palais 2",
        "palais deux",
        "le palais deux",
        "bu ñaareel",
        "bi ñaareel",
        "ñaar",
        "deuxième",
        "second",
        "the second one",
    ],
)
def test_second_option_is_recognized(answer: str) -> None:
    assert match_choice(answer, OPTIONS) == 1


@pytest.mark.parametrize(
    "answer",
    [
        "palais 1",
        "palais un",
        "bu njëkk",
        "bi njëkk",
        "benn",
        "premier",
        "première",
        "first",
    ],
)
def test_first_option_is_recognized(answer: str) -> None:
    assert match_choice(answer, OPTIONS) == 0


def test_unrelated_answer_is_left_open() -> None:
    assert match_choice("I do not know", OPTIONS) is None
    assert match_choice("palais", OPTIONS) is None
