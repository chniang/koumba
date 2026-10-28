"""Build the text-only Wolof voice lexicon from the Dakar bus dataset.

Run it from the repository root:

    .\\.venv\\Scripts\\python.exe scripts\\build_voice_lexicon.py

This script does not synthesize audio.  It writes:

* ``data/voice_lexicon.csv`` - connectors, bus names, and place names.
* ``data/voice_review_top50.txt`` - the busiest places, ready for a native
  speaker to add vowel length where needed.

The place ``say_as`` column stays identical to the source spelling unless a
human override exists.  The ``suggestion`` column is the only place where the
mechanical, conservative letter conversions from ``koumba.voice_lexicon`` are
applied.
"""

from __future__ import annotations

import csv
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from koumba.data import load_data
from koumba.phrases import short_bus_name
from koumba.voice_lexicon import (
    CONNECTOR_TEXTS,
    bus_id,
    connector_id,
    place_id,
    place_needs_review,
    safe_place_suggestion,
)
from koumba.wolof_numbers import wolof_number

DATA_DIR = ROOT / "data"
LEXICON_PATH = DATA_DIR / "voice_lexicon.csv"
OVERRIDES_PATH = DATA_DIR / "voice_overrides.csv"
REVIEW_PATH = DATA_DIR / "voice_review_top50.txt"

FIELDNAMES = ("id", "kind", "text", "say_as", "suggestion", "status")


def _clean(value: Any) -> str:
    """Turn a CSV value into a stripped string without leaking ``nan``."""

    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _row(
    row_id: str,
    kind: str,
    text: str,
    say_as: str,
    suggestion: str,
    status: str,
) -> dict[str, str]:
    return {
        "id": row_id,
        "kind": kind,
        "text": text,
        "say_as": say_as,
        "suggestion": suggestion,
        "status": status,
    }


def _connector_rows() -> list[dict[str, str]]:
    rows = []
    for text in CONNECTOR_TEXTS:
        # These groups were already validated in the phrase templates.  Their
        # spelling is the pronunciation guide, so say_as is exactly the text.
        rows.append(
            _row(
                connector_id(text),
                "connector",
                text,
                text,
                text,
                "validated",
            )
        )
    return rows


def _plain_line_number(line: str) -> int | None:
    """Return an integer only for a line such as ``Ligne 38``.

    S suffixes (``15A``), ranges, and names such as ``TAF TAF`` are left for
    review rather than being squeezed into a number word.
    """

    raw = re.sub(r"(?i)^\s*ligne\s+", "", line).strip()
    if not raw.isdigit():
        return None
    return int(raw)


def _bus_rows(lignes) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in lignes.itertuples(index=False):
        operator = _clean(row.compagnie)
        line = _clean(row.ligne)
        text = short_bus_name(operator, line)
        if not text:
            continue

        row_id = bus_id(operator, line)
        if row_id in seen:
            continue
        seen.add(row_id)

        number = _plain_line_number(line)
        lowered_operator = operator.lower()
        if number is not None and 1 <= number <= 199:
            if "tata" in lowered_operator:
                say_as = f"Tataa {wolof_number(number)}"
            elif "dem dikk" in lowered_operator:
                say_as = f"Dem Dikk {wolof_number(number)}"
            else:
                say_as = text
            status = "validated"
        else:
            # A line number outside the validated range is not guessed.  It is
            # kept as-is and marked for a human, just like 16A and TAF TAF.
            say_as = text
            status = "review"

        rows.append(_row(row_id, "bus", text, say_as, say_as, status))

    rows.sort(key=lambda item: (item["text"].casefold(), item["text"]))
    return rows


def _place_names(data) -> list[str]:
    """Collect every exact stop/place spelling that can reach a spoken answer."""

    names: set[str] = set()
    for value in data.arrets["arret"].tolist():
        text = _clean(value)
        if text:
            names.add(text)
    for value in data.lieux["lieu"].tolist():
        text = _clean(value)
        if text:
            names.add(text)
    for value in data.itineraires["arret"].tolist():
        text = _clean(value)
        if text:
            names.add(text)
    for column in ("terminus_a", "terminus_b"):
        for value in data.lignes[column].tolist():
            text = _clean(value)
            if text:
                names.add(text)

    return sorted(names, key=lambda text: (text.casefold(), text))


def _place_rows(data) -> list[dict[str, str]]:
    rows = []
    for text in _place_names(data):
        suggestion = safe_place_suggestion(text)
        status = "review" if place_needs_review(text) else "auto"
        # Keep the dataset spelling until a person has checked vowel length.
        rows.append(
            _row(place_id(text), "place", text, text, suggestion, status)
        )
    return rows


def _read_overrides(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    overrides: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or set(reader.fieldnames) != {"text", "say_as"}:
            raise ValueError(f"{path} must have exactly text,say_as columns")
        for row in reader:
            text = _clean(row.get("text"))
            say_as = _clean(row.get("say_as"))
            if not text or not say_as:
                raise ValueError(f"{path} has an empty text or say_as value")
            overrides[text] = say_as
    return overrides


def _apply_overrides(
    rows: list[dict[str, str]],
    overrides: dict[str, str],
) -> None:
    """Apply reviewed spellings last, so they always win over generation."""

    by_text = {row["text"]: row for row in rows if row["kind"] == "place"}
    for text, say_as in overrides.items():
        row = by_text.get(text)
        if row is None:
            # An explicit human override is useful even if a future dataset
            # drops that stop, so keep it instead of silently discarding it.
            row = _row(place_id(text), "place", text, say_as, say_as, "validated")
            rows.append(row)
            by_text[text] = row
            continue

        row["say_as"] = say_as
        row["suggestion"] = say_as
        row["status"] = "validated"


def _line_counts(data) -> Counter[str]:
    """Count distinct (operator, line) pairs that serve each exact stop."""

    seen: set[tuple[str, str, str]] = set()
    counts: Counter[str] = Counter()
    for row in data.itineraires.itertuples(index=False):
        operator = _clean(row.compagnie)
        line = _clean(row.ligne)
        stop = _clean(row.arret)
        if not operator or not line or not stop:
            continue
        key = (operator, line, stop)
        if key in seen:
            continue
        seen.add(key)
        counts[stop] += 1
    return counts


def _write_lexicon(rows: list[dict[str, str]]) -> None:
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("voice lexicon contains duplicate ids")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with LEXICON_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDNAMES,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_review_file(data, rows: list[dict[str, str]]) -> int:
    suggestions = {
        row["text"]: row["suggestion"]
        for row in rows
        if row["kind"] == "place"
    }
    counts = _line_counts(data)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )[:50]

    with REVIEW_PATH.open("w", encoding="utf-8", newline="") as handle:
        for text, _count in ranked:
            suggestion = suggestions.get(text) or safe_place_suggestion(text)
            handle.write(f"{text}  ->  {suggestion}\n")
    return len(ranked)


def _print_counts(rows: list[dict[str, str]]) -> None:
    kind_counts = Counter(row["kind"] for row in rows)
    status_counts = Counter(row["status"] for row in rows)

    print("Rows by kind:")
    for kind in ("connector", "bus", "place"):
        print(f"  {kind}: {kind_counts[kind]}")
    print("Rows by status:")
    for status in ("validated", "auto", "review"):
        print(f"  {status}: {status_counts[status]}")


def main() -> int:
    data = load_data()
    rows = _connector_rows() + _bus_rows(data.lignes) + _place_rows(data)
    _apply_overrides(rows, _read_overrides(OVERRIDES_PATH))

    _write_lexicon(rows)
    review_count = _write_review_file(data, rows)

    _print_counts(rows)
    print(f"Wrote {LEXICON_PATH.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"Wrote {REVIEW_PATH.relative_to(ROOT)} ({review_count} places)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
