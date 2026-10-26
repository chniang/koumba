"""Small synthetic networks make the routing rules easy to see."""

from __future__ import annotations

import math

import pytest

from koumba.network import Line, Network
from koumba.places import PlaceIndex
from koumba.router import NoRouteFoundError, PlaceNotFoundError, RouteEngine

BASE_LAT = 14.7
BASE_LON = -17.45


def point(north_m: float = 0.0, east_m: float = 0.0) -> tuple[float, float]:
    """Turn a simple metre offset into a latitude/longitude pair."""

    lat = BASE_LAT + north_m / 111_320
    lon = BASE_LON + east_m / (111_320 * math.cos(math.radians(BASE_LAT)))
    return (lat, lon)


def make_engine(lines: list[Line], coords: dict[str, tuple[float, float]]) -> RouteEngine:
    network = Network(lines, coords)
    return RouteEngine(network, PlaceIndex.from_network(network))


def test_direct_trip() -> None:
    engine = make_engine(
        [Line("Op A", "Line A", ("Origin", "Mid", "Dest"))],
        {"Origin": point(0), "Mid": point(1000), "Dest": point(2000)},
    )

    trips = engine.find_trips("Origin", "Dest")

    assert trips
    trip = trips[0]
    assert trip.transfers == 0
    assert len(trip.legs) == 1
    assert trip.legs[0].board_stop == "Origin"
    assert trip.legs[0].alight_stop == "Dest"
    assert trip.legs[0].direction == "Dest"
    assert trip.legs[0].num_stops == 2
    assert 7 <= trip.total_minutes <= 9
    assert trip.ridden_stops == ("Origin", "Mid", "Dest")


def test_one_transfer() -> None:
    engine = make_engine(
        [
            Line("Op A", "Line A", ("Origin", "Transfer")),
            Line("Op B", "Line B", ("Transfer", "Dest")),
        ],
        {
            "Origin": point(0),
            "Transfer": point(1000),
            "Dest": point(2000, 1000),
        },
    )

    trips = engine.find_trips("Origin", "Dest")

    assert trips
    trip = trips[0]
    assert trip.transfers == 1
    assert len(trip.legs) == 2
    assert trip.legs[0].alight_stop == "Transfer"
    assert trip.legs[1].board_stop == "Transfer"
    # 4 min + 4.5 min riding, plus the 6 min transfer penalty.
    assert 13 <= trip.total_minutes <= 16


def test_direct_is_preferred_even_when_slightly_slower() -> None:
    engine = make_engine(
        [
            Line("Op D", "Direct", ("Origin", "Mid", "Dest")),
            Line("Op A", "Line A", ("Origin", "Transfer")),
            Line("Op B", "Line B", ("Transfer", "Dest")),
        ],
        {
            "Origin": point(0, 0),
            "Mid": point(1250, 1480),
            "Dest": point(2000, 0),
            "Transfer": point(1000, 0),
        },
    )

    trips = engine.find_trips("Origin", "Dest")

    assert trips
    direct = [trip for trip in trips if trip.transfers == 0]
    assert direct
    assert trips[0].transfers == 0
    assert not trips[0].detour


def test_detour_is_rejected_when_transfer_route_exists() -> None:
    engine = make_engine(
        [
            Line("Op D", "Detour", ("Origin", "Far", "Dest")),
            Line("Op A", "Line A", ("Origin", "Transfer")),
            Line("Op B", "Line B", ("Transfer", "Dest")),
        ],
        {
            "Origin": point(0, 0),
            "Far": point(10000, 0),
            "Dest": point(2000, 0),
            "Transfer": point(1000, 0),
        },
    )

    trips = engine.find_trips("Origin", "Dest")

    assert trips
    # The direct line goes 18 km for a 2 km straight-line trip.  It must be
    # thrown away, leaving the sensible one-transfer route.
    assert all(trip.transfers == 1 for trip in trips)
    assert trips[0].line_combo == (("Op A", "Line A"), ("Op B", "Line B"))


def test_more_than_two_transfers_is_rejected() -> None:
    engine = make_engine(
        [
            Line("Op A", "Line A", ("Origin", "Transfer 1")),
            Line("Op B", "Line B", ("Transfer 1", "Transfer 2")),
            Line("Op C", "Line C", ("Transfer 2", "Transfer 3")),
            Line("Op D", "Line D", ("Transfer 3", "Dest")),
        ],
        {
            "Origin": point(0),
            "Transfer 1": point(1000),
            "Transfer 2": point(2000),
            "Transfer 3": point(3000),
            "Dest": point(4000),
        },
    )

    with pytest.raises(NoRouteFoundError):
        engine.find_trips("Origin", "Dest")


def test_transfer_walk_is_limited_to_300_metres() -> None:
    engine = make_engine(
        [
            Line("Op A", "Line A", ("Origin", "Near")),
            Line("Op B", "Line B", ("Far", "Dest")),
        ],
        {
            "Origin": point(0),
            "Near": point(1000),
            # Near and Far are 1 km apart, so this transfer needs too much
            # walking and must not be used.
            "Far": point(1000, 1000),
            "Dest": point(2000, 1000),
        },
    )

    with pytest.raises(NoRouteFoundError):
        engine.find_trips("Origin", "Dest")


def test_unknown_place() -> None:
    engine = make_engine(
        [Line("Op A", "Line A", ("Origin", "Dest"))],
        {"Origin": point(0), "Dest": point(1000)},
    )

    with pytest.raises(PlaceNotFoundError):
        engine.find_trips("Atlantis", "Dest")
