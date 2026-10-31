"""Recognize a spoken or typed answer to a two-option clarification."""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from .places import normalize_name

_FIRST_WORDS = (
    "bu njekk",
    "bi njekk",
    "benn",
    "premier",
    "premiere",
    "first",
)
_SECOND_WORDS = (
    "bu naareel",
    "bi naareel",
    "naar",
    "deuxieme",
    "second",
    "seconde",
    "seconds",
)
_NUMBER_WORDS = {
    "un": "1",
    "une": "1",
    "one": "1",
    "deux": "2",
    "two": "2",
}


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def _ordinal_choice(text: str) -> int | None:
    first = any(_has_word(text, word) for word in _FIRST_WORDS)
    second = any(_has_word(text, word) for word in _SECOND_WORDS)
    if first == second:
        return None
    return 0 if first else 1


def _canonical_numbers(text: str) -> str:
    return " ".join(_NUMBER_WORDS.get(token, token) for token in text.split())


def match_choice(text: str, options: list[str] | tuple[str, ...]) -> int | None:
    """Return 0 or 1 when the answer clearly picks one option.

    The caller keeps the question open when the answer is unrelated or when
    both options remain possible.
    """

    cleaned = [str(option or "").strip() for option in options[:2]]
    if len(cleaned) < 2 or not all(cleaned):
        return None

    normalized = normalize_name(text)
    if not normalized:
        return None

    ordinal = _ordinal_choice(normalized)
    if ordinal is not None:
        return ordinal

    answer = _canonical_numbers(normalized)
    matches: list[int] = []
    for index, option in enumerate(cleaned):
        candidate = _canonical_numbers(normalize_name(option))
        if candidate and f" {candidate} " in f" {answer} ":
            matches.append(index)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        lengths = [len(normalize_name(cleaned[index])) for index in matches]
        longest = max(lengths)
        winners = [index for index, size in zip(matches, lengths) if size == longest]
        return winners[0] if len(winners) == 1 else None

    scored = [
        (fuzz.WRatio(answer, _canonical_numbers(normalize_name(option))), index)
        for index, option in enumerate(cleaned)
    ]
    scored.sort(reverse=True)
    best_score, best_index = scored[0]
    if len(scored) > 1 and best_score - scored[1][0] < 8:
        return None
    return best_index if best_score >= 88 else None
