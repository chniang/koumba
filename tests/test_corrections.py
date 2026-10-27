"""Field corrections are small, auditable fixes from local rider knowledge."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from koumba.corrections import (
    DEFAULT_CORRECTIONS_PATH,
    apply_corrections,
    load_corrections,
)
from koumba.network import Line, Network


def _line() -> Line:
    return Line(
        company="Tata AFTU",
        name="Ligne Test",
        stops=("A", "B", "C", "D"),
        terminus_a="A",
        terminus_b="D",
    )


def _correction(op: str, **values) -> dict:
    return {
        "op": op,
        "operator": "Tata AFTU",
        "line": "Ligne Test",
        "reason": "Seen on the field.",
        **values,
    }


def test_repository_corrections_file_starts_empty() -> None:
    assert json.loads(DEFAULT_CORRECTIONS_PATH.read_text(encoding="utf-8")) == []


def test_each_supported_correction_changes_the_itinerary(capsys) -> None:
    cases = [
        (
            _correction("remove_stop", stop="B"),
            ("A", "C", "D"),
        ),
        (
            _correction("replace_stop", stop="C", by="X"),
            ("A", "B", "X", "D"),
        ),
        (
            _correction("insert_stop_after", after="B", stop="X"),
            ("A", "B", "X", "C", "D"),
        ),
        (
            _correction("truncate_after", stop="C"),
            ("A", "B", "C"),
        ),
    ]

    for correction, expected in cases:
        result = apply_corrections(
            [_line()],
            [correction],
            {"A", "B", "C", "D", "X"},
        )

        assert result.applied == 1
        assert result.skipped == 0
        assert result.lines[0].stops == expected
        assert result.lines[0].terminus_a == expected[0]
        assert result.lines[0].terminus_b == expected[-1]

    assert "[corrections] 1 applied, 0 skipped" in capsys.readouterr().out


def test_bad_corrections_warn_and_are_skipped(capsys) -> None:
    corrections = [
        {
            "op": "remove_stop",
            "operator": "Tata AFTU",
            "line": "Ligne Missing",
            "stop": "A",
            "reason": "Unknown line.",
        },
        _correction("remove_stop", stop="Nowhere"),
        _correction("replace_stop", stop="A", by="Unknown"),
        _correction("insert_stop_after", after="Nowhere", stop="A"),
        _correction("insert_stop_after", after="A", stop="Unknown"),
        _correction("truncate_after", stop="Nowhere"),
        _correction("unknown_operation", stop="A"),
        {
            "op": "remove_stop",
            "operator": "Tata AFTU",
            "line": "Ligne Test",
            "stop": "A",
        },
    ]

    result = apply_corrections(
        [_line()],
        corrections,
        {"A", "B", "C", "D"},
    )

    assert result.applied == 0
    assert result.skipped == len(corrections)
    assert result.lines[0].stops == ("A", "B", "C", "D")
    output = capsys.readouterr().out
    assert "unknown line" in output
    assert "unknown stop" in output
    assert "unknown replacement stop" in output
    assert "needs a reason" in output
    assert "unsupported operation" in output
    assert "[corrections] 0 applied, 8 skipped" in output


def test_network_applies_corrections_before_building_nodes(
    tmp_path,
    capsys,
) -> None:
    data = SimpleNamespace(
        lignes=pd.DataFrame(
            [
                {
                    "compagnie": "Tata AFTU",
                    "ligne": "Ligne Test",
                    "terminus_a": "A",
                    "terminus_b": "C",
                }
            ]
        ),
        itineraires=pd.DataFrame(
            [
                {"compagnie": "Tata AFTU", "ligne": "Ligne Test", "ordre": 1, "arret": "A"},
                {"compagnie": "Tata AFTU", "ligne": "Ligne Test", "ordre": 2, "arret": "B"},
                {"compagnie": "Tata AFTU", "ligne": "Ligne Test", "ordre": 3, "arret": "C"},
            ]
        ),
        arrets=pd.DataFrame(
            [
                {"arret": "A", "lat": 14.70, "lon": -17.45, "confiance": "bonne"},
                {"arret": "B", "lat": 14.71, "lon": -17.46, "confiance": "bonne"},
                {"arret": "C", "lat": 14.72, "lon": -17.47, "confiance": "bonne"},
                {"arret": "X", "lat": 14.715, "lon": -17.465, "confiance": "bonne"},
            ]
        ),
        lieux=pd.DataFrame([{"lieu": "A", "lat": 14.70, "lon": -17.45}]),
    )
    path = tmp_path / "corrections.json"
    path.write_text(
        json.dumps(
            [
                {
                    "op": "insert_stop_after",
                    "operator": "Tata AFTU",
                    "line": "Ligne Test",
                    "after": "B",
                    "stop": "X",
                    "reason": "The bus turns at X before C.",
                }
            ]
        ),
        encoding="utf-8",
    )

    network = Network.from_data(data, corrections_path=path)

    assert network.lines[("Tata AFTU", "Ligne Test")].stops == (
        "A",
        "B",
        "X",
        "C",
    )
    assert network.nodes_at_stop("X")
    assert "[corrections] 1 applied, 0 skipped" in capsys.readouterr().out


def test_missing_corrections_file_does_not_crash(tmp_path, capsys) -> None:
    path = tmp_path / "missing.json"

    assert load_corrections(path) == []
    assert "was not found" in capsys.readouterr().out
