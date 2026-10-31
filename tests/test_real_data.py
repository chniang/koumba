"""Integration checks on the public Dakar dataset."""

from __future__ import annotations

import pytest

from koumba.data import load_data
from koumba.network import Network
from koumba.phrases import spoken_answer
from koumba.places import (
    PlaceIndex,
    normalize_name,
    prefer_exact_matches,
)
from koumba.router import RouteEngine, Trip
from koumba.tools import _trip_to_dict


@pytest.fixture(scope="module")
def data():
    return load_data()


@pytest.fixture(scope="module")
def engine(data) -> RouteEngine:
    network = Network.from_data(data)
    place_index = PlaceIndex.from_data(data)
    return RouteEngine(network, place_index)


def _trip_text(trip: Trip) -> str:
    legs = " | ".join(
        f"{leg.operator} {leg.line}: {leg.board_stop} -> {leg.alight_stop} "
        f"(towards {leg.direction}, {leg.num_stops} stops)"
        for leg in trip.legs
    )
    return (
        f"{trip.transfers} transfer(s), {trip.total_minutes:.0f} min, "
        f"combo={trip.line_combo}: {legs}"
    )


def test_schema_matches_the_public_dataset(data) -> None:
    assert list(data.lignes.columns) == [
        "compagnie",
        "ligne",
        "terminus_a",
        "terminus_b",
        "categorie",
        "source",
    ]
    assert list(data.itineraires.columns) == ["compagnie", "ligne", "ordre", "arret"]
    assert list(data.arrets.columns) == [
        "arret",
        "lat",
        "lon",
        "quartier",
        "commune",
        "confiance",
    ]
    assert list(data.lieux.columns) == ["lieu", "lat", "lon"]
    assert len(data.lignes) > 100
    assert len(data.itineraires) > 1_000


def test_rond_point_sahm_matches_only_the_sahm_stop(data) -> None:
    index = PlaceIndex.from_data(data)

    matches = index.match("Rond-Point Sahm")

    # The old loose matcher also returned unrelated "Rond-Point ..." stops.
    assert [candidate.name for candidate in matches] == ["Sahm"]


def test_sahm_is_shown_in_normal_case_in_cards_and_speech(
    engine: RouteEngine,
) -> None:
    network = engine.network
    index = engine.place_index

    assert "SAHM" not in network.stop_names
    assert "Sahm" in network.stop_names
    assert index.match("SAHM")[0].name == "Sahm"

    trips = engine.find_trips("Rond-Point Sahm", "Malika", max_trips=1)
    trip = _trip_to_dict(trips[0])

    assert all(
        "SAHM" not in leg["board_stop"]
        and "SAHM" not in leg["alight_stop"]
        and "SAHM" not in leg["direction"]
        for leg in trip["legs"]
    )
    spoken = spoken_answer(trip, "English", "Sahm")
    assert "SAHM" not in spoken
    assert "Sahm" in spoken


def test_reviewed_stop_spellings_display_and_merge_in_the_graph(data) -> None:
    network = Network.from_data(data)
    index = PlaceIndex.from_data(data)

    assert "Case-Ba" not in network.stop_names
    assert "Pattesd'oie" not in network.stop_names
    assert "Case Ba" in network.stop_names
    assert "Patte d'Oie" in network.stop_names

    assert index.match("Case-Ba")[0].name == "Case Ba"
    assert index.match("Pattesd'oie")[0].name == "Patte d'Oie"

    # Ligne 38 used the "Pattesd'oie" spelling; it now uses the display name.
    ligne_38 = network.lines[("Tata AFTU", "Ligne 38")]
    assert "Patte d'Oie" in ligne_38.stops
    assert "Pattesd'oie" not in ligne_38.stops

    # Ligne 33 used "Case-Ba", while other lines used "Case Ba".  Both
    # spellings now point at the same stop nodes.
    assert any(
        node.line == ("Tata AFTU", "Ligne 33")
        for node in network.nodes_at_stop("Case Ba")
    )


def test_exact_yoff_match_does_not_route_via_a_longer_name(
    engine: RouteEngine,
) -> None:
    matches = engine.place_index.match("Yoff")

    assert any(candidate.normalized == "yoff" for candidate in matches)
    assert any(candidate.normalized == "grand yoff" for candidate in matches)
    exact = prefer_exact_matches(matches, "Yoff")
    assert [candidate.normalized for candidate in exact] == ["yoff"]

    trips = engine.find_trips("Yoff", "Palais", max_trips=1)

    assert trips
    assert normalize_name(trips[0].legs[0].board_stop) == "yoff"


def test_accent_insensitive_medina_is_an_exact_match(
    engine: RouteEngine,
) -> None:
    matches = engine.place_index.match("medina")

    exact = prefer_exact_matches(matches, "medina")

    assert [candidate.normalized for candidate in exact] == ["medina"]
    assert exact[0].name == "Médina"


def test_rond_point_sahm_to_malika_avoids_palais(engine: RouteEngine) -> None:
    trips = engine.find_trips("Rond-Point Sahm", "Malika")

    assert trips
    print("\nRond-Point Sahm -> Malika")
    for trip in trips:
        print("  " + _trip_text(trip))
        assert not trip.detour
        assert not trip.via_downtown
        assert not any(
            normalize_name(stop)
            in {"palais", "palais 1", "palais 2", "centre ville"}
            for stop in trip.ridden_stops
        )

    # There is no direct line serving both places, so the best option
    # should use exactly one transfer, not a two-transfer detour.
    assert trips[0].transfers == 1
    assert trips[0].line_combo == (
        ("Tata AFTU", "Ligne 38"),
        ("Tata AFTU", "Ligne 80"),
    )


def test_medina_to_yoff_uses_the_nearby_avenue_and_yoff_village(
    engine: RouteEngine,
) -> None:
    trips = engine.find_trips("medina", "Yoff")

    assert trips
    trip = trips[0]
    assert trip.transfers == 0
    assert trip.line_combo in {
        (("Tata AFTU", "Ligne 3"),),
        (("Tata AFTU", "Ligne 4"),),
    }
    assert trip.legs[0].board_stop == "Avenue Blaise Diagne"
    assert trip.legs[0].alight_stop == "Yoff-Village"

    access = [walk for walk in trip.walks if walk.kind == "access"]
    egress = [walk for walk in trip.walks if walk.kind == "egress"]
    assert access and access[0].distance_m is not None
    assert egress and egress[0].distance_m is not None
    assert access[0].distance_m <= 600
    assert egress[0].distance_m <= 600
    assert all(
        normalize_name(stop) != "grand yoff" for stop in trip.ridden_stops
    )


def test_yoff_never_boards_or_alights_at_grand_yoff(
    engine: RouteEngine,
) -> None:
    trips = engine.find_trips("Yoff", "Palais")

    assert trips
    for trip in trips:
        assert all(
            normalize_name(stop) != "grand yoff"
            for stop in trip.ridden_stops
        )
        assert all(
            normalize_name(leg.board_stop) != "grand yoff"
            and normalize_name(leg.alight_stop) != "grand yoff"
            for leg in trip.legs
        )


@pytest.mark.parametrize(
    ("origin", "destination"),
    [
        ("Ouakam", "Palais"),
        ("Parcelles Assainies", "UCAD"),
        ("Guediawaye", "Palais"),
    ],
)
def test_requested_routes_return_at_least_one_trip(
    engine: RouteEngine,
    origin: str,
    destination: str,
) -> None:
    trips = engine.find_trips(origin, destination)

    assert trips
    print(f"\n{origin} -> {destination}")
    for trip in trips:
        print("  " + _trip_text(trip))

    # These three pairs all have a direct line, so a direct trip must come
    # first even if a transfer would save a few minutes.
    assert trips[0].transfers == 0
    assert all(not trip.detour for trip in trips)

    # Do not return a five-hour route when a reasonable one exists.
    fastest = min(trip.total_minutes for trip in trips)
    assert all(trip.total_minutes <= 3 * fastest for trip in trips)


def test_cli_smoke(capsys) -> None:
    from koumba.cli import main

    assert main(["Rond-Point Sahm", "Malika"]) == 0

    output = capsys.readouterr().out
    assert "Koumba route engine" in output
    assert "Departure match: Sahm" in output
    assert "Trip 1" in output
    assert "Malika" in output
