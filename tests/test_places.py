from __future__ import annotations

from koumba.network import Line, Network
from koumba.places import (
    PlaceIndex,
    exact_normalized_names,
    normalize_name,
    prefer_exact_matches,
    representative_coordinates,
    spoken_alias,
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


def test_reviewed_spoken_forms_are_reversed_and_accent_insensitive() -> None:
    assert spoken_alias("saam") == "Sahm"
    assert spoken_alias("paay") == "PAI"
    assert spoken_alias("wakaam") == "Ouakam"
    assert spoken_alias("barñi") == "Bargny"
    assert spoken_alias("barni") == "Bargny"
    assert spoken_alias("el ess ess") == "LSS"


def test_spoken_alias_wins_over_a_similar_official_name() -> None:
    network = Network(
        [Line("Op", "L1", ("Sahm", "Sam Notaire"))],
        {"Sahm": (14.68, -17.45), "Sam Notaire": (14.69, -17.46)},
    )
    index = PlaceIndex.from_network(network)

    matches = index.match("saam")

    assert matches[0].name == "Sahm"
    assert matches[0].score == 100.0
    assert [item.normalized for item in prefer_exact_matches(matches, "saam")] == [
        "sahm"
    ]
    assert exact_normalized_names("saam") == {"saam", "sahm"}
