from __future__ import annotations

from koumba.network import Line, Network
from koumba.places import (
    PlaceIndex,
    normalize_name,
    representative_coordinates,
)


def test_normalization_handles_accents_and_abbreviations() -> None:
    assert normalize_name("Rond-Point Sahm") == "rond point sahm"
    assert normalize_name("Guédiawaye") == "guediawaye"
    assert normalize_name("Av. Malick Sy") == "avenue malick sy"
    assert normalize_name("Rte des Niayes") == "route des niayes"


def test_neighbourhood_keeps_all_good_stops() -> None:
    network = Network(
        [
            Line(
                "Op",
                "L1",
                ("Sahm", "Malika", "Malika Plage", "Malika Cimetière"),
            )
        ],
        {
            "Sahm": (14.6856, -17.4543),
            "Malika": (14.8015, -17.3376),
            "Malika Plage": (14.8067, -17.3389),
            "Malika Cimetière": (14.7957, -17.3363),
        },
    )
    index = PlaceIndex.from_network(network)

    matches = index.match("Malika")

    names = {candidate.name for candidate in matches}
    assert {"Malika", "Malika Plage", "Malika Cimetière"} <= names
    assert all(candidate.score >= 80 for candidate in matches)
    assert representative_coordinates(matches) is not None


def test_accent_insensitive_match() -> None:
    network = Network(
        [Line("Op", "L1", ("Guédiawaye", "Palais 1"))],
        {"Guédiawaye": (14.7771, -17.3901), "Palais 1": (14.6510, -17.4335)},
    )
    index = PlaceIndex.from_network(network)

    matches = index.match("Guediawaye")

    assert matches
    assert matches[0].name == "Guédiawaye"
    assert matches[0].score == 100


def test_unknown_place_has_no_match() -> None:
    network = Network(
        [Line("Op", "L1", ("Origin", "Dest"))],
        {"Origin": (14.7, -17.45), "Dest": (14.71, -17.45)},
    )
    index = PlaceIndex.from_network(network)

    assert index.match("Atlantis") == []
    assert index.suggestions("Atlantis")
