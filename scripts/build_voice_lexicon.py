"""Build the text-only Wolof voice lexicon from the Dakar bus dataset.

Run it from the repository root:

    .\\.venv\\Scripts\\python.exe scripts\\build_voice_lexicon.py

This script does not synthesize audio.  It writes:

* ``data/voice_lexicon.csv`` - connectors, bus names, and place names.
* ``data/voice_review_top50.txt`` - the busiest places, ready for a native
  speaker to review.

French words and names stay exactly as written.  The only place spellings that
change are the reviewed whole words in ``data/voice_word_overrides.csv`` and
the full-name overrides in ``data/voice_overrides.csv``.  The latter are applied
last and win.  The ``suggestion`` column keeps the old mechanical note, but it
is never copied into ``say_as``.
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
    apply_word_overrides,
    bus_id,
    connector_id,
    place_id,
    safe_place_suggestion,
)
from koumba.wolof_numbers import MAX_NUMBER, MIN_NUMBER, wolof_number

DATA_DIR = ROOT / "data"
LEXICON_PATH = DATA_DIR / "voice_lexicon.csv"
WORD_OVERRIDES_PATH = DATA_DIR / "voice_word_overrides.csv"
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


def _line_number_and_suffix(line: str) -> tuple[int, str] | None:
    """Split ``Ligne 15A`` into (15, "A") and ``Ligne 38`` into (38, "")."""

    raw = re.sub(r"(?i)^\s*ligne\s+", "", line).strip()
    match = re.fullmatch(r"(\d+)([A-Za-z]*)?", raw)
    if match is None:
        return None
    return int(match.group(1)), match.group(2).upper() or ""


def _pronounced_bus_name(
    operator: str,
    line: str,
    display_name: str,
) -> str:
    """Use the Oolel voice spelling for a validated bus name.

    A plain line speaks its number.  A line such as ``15A`` speaks the number,
    then the capital letter.  ``TAF TAF`` is its own validated name.
    """

    parsed = _line_number_and_suffix(line)
    lowered_operator = operator.lower()
    if parsed is not None and MIN_NUMBER <= parsed[0] <= MAX_NUMBER:
        number, suffix = parsed
        spoken_line = wolof_number(number)
        if suffix:
            spoken_line += f" {suffix}"
        if "tata" in lowered_operator:
            return f"Tataa {spoken_line}"
        if "dem dikk" in lowered_operator:
            return f"Dem Dikk {spoken_line}"
        return display_name

    if line.strip().upper() == "TAF TAF":
        return f"Dem Dikk Taf Taf"

    return display_name


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

        say_as = _pronounced_bus_name(operator, line, text)
        rows.append(_row(row_id, "bus", text, say_as, say_as, "validated"))

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
        # The suggestion is just a note for a human.  It never becomes say_as:
        # French-looking spellings may be exactly what the Oolel voice needs.
        suggestion = safe_place_suggestion(text)
        rows.append(
            _row(place_id(text), "place", text, text, suggestion, "auto")
        )
    return rows


def _read_override_csv(path: Path, key_column: str) -> dict[str, str]:
    """Read a two-column pronunciation override file, if it exists."""

    if not path.exists():
        return {}

    overrides: dict[str, str] = {}
    expected = {key_column, "say_as"}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or set(reader.fieldnames) != expected:
            columns = ",".join((key_column, "say_as"))
            raise ValueError(f"{path} must have exactly {columns} columns")
        for row in reader:
            key = _clean(row.get(key_column))
            say_as = _clean(row.get("say_as"))
            if not key or not say_as:
                raise ValueError(f"{path} has an empty {key_column} or say_as")
            overrides[key] = say_as
    return overrides


def _apply_word_overrides(
    rows: list[dict[str, str]],
    overrides: dict[str, str],
) -> None:
    """Replace reviewed words inside every place's say_as."""

    for row in rows:
        if row["kind"] == "place":
            row["say_as"] = apply_word_overrides(row["say_as"], overrides)


def _apply_full_name_overrides(
    rows: list[dict[str, str]],
    overrides: dict[str, str],
) -> None:
    """Apply whole-name spellings after the word pass, so they win."""

    by_text = {row["text"]: row for row in rows if row["kind"] == "place"}
    for text, say_as in overrides.items():
        row = by_text.get(text)
        if row is None:
            # An explicit human override is useful even if a future dataset
            # drops that stop, so keep it instead of silently discarding it.
            row = _row(place_id(text), "place", text, say_as, say_as, "auto")
            rows.append(row)
            by_text[text] = row
            continue

        row["say_as"] = say_as
        row["suggestion"] = say_as


def _mark_place_statuses(
    rows: list[dict[str, str]],
    reviewed_names: set[str],
) -> None:
    """Mark changed places and the reviewed top-50 places as validated."""

    for row in rows:
        if row["kind"] != "place":
            continue
        changed = row["say_as"] != row["text"]
        row["status"] = (
            "validated"
            if changed or row["text"] in reviewed_names
            else "auto"
        )


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


def _review_ranked(data) -> list[tuple[str, int]]:
    """Return the 50 busiest places, with deterministic tie-breaking."""

    counts = _line_counts(data)
    return sorted(
        counts.items(),
        key=lambda item: (-item[1], item[0].casefold(), item[0]),
    )[:50]


def _write_review_file(
    ranked: list[tuple[str, int]],
    rows: list[dict[str, str]],
) -> int:
    suggestions = {
        row["text"]: row["suggestion"]
        for row in rows
        if row["kind"] == "place"
    }

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

    word_overrides = _read_override_csv(WORD_OVERRIDES_PATH, "word")
    full_overrides = _read_override_csv(OVERRIDES_PATH, "text")
    _apply_word_overrides(rows, word_overrides)
    _apply_full_name_overrides(rows, full_overrides)

    ranked = _review_ranked(data)
    _mark_place_statuses(rows, {text for text, _count in ranked})

    _write_lexicon(rows)
    review_count = _write_review_file(ranked, rows)

    _print_counts(rows)
    print(f"Wrote {LEXICON_PATH.relative_to(ROOT)} ({len(rows)} rows)")
    print(f"Wrote {REVIEW_PATH.relative_to(ROOT)} ({review_count} places)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
