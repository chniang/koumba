"""Integration checks on the public Dakar dataset."""

from __future__ import annotations

import pytest

from koumba.data import load_data
from koumba.network import Network
from koumba.places import PlaceIndex, normalize_name
from koumba.router import RouteEngine, Trip


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
