"""Command line interface: ``python -m koumba.cli "From" "To"``."""

from __future__ import annotations

import argparse
import sys

from .data import describe_data, load_data
from .network import Network
from .places import PlaceIndex
from .router import NoRouteFoundError, PlaceNotFoundError, RouteEngine, Trip


def _minutes(value: float) -> str:
    if value < 1:
        return "under 1 min"
    return f"{round(value)} min"


def _distance(value: float | None, missing: str = "unknown distance") -> str:
    if value is None:
        return missing
    if value >= 1_000:
        return f"{value / 1_000:.1f} km"
    return f"{value:.0f} m"


def _print_trip(number: int, trip: Trip) -> None:
    transfer_word = "transfer" if trip.transfers == 1 else "transfers"
    print()
    print(
        f"Trip {number} - {trip.transfers} {transfer_word} - "
        f"about {_minutes(trip.total_minutes)}"
    )

    access_walks = [walk for walk in trip.walks if walk.after_leg == 0]
    for walk in access_walks:
        print(
            f"  Walk {_minutes(walk.minutes)}: "
            f"{walk.from_place} -> {walk.to_place}"
        )

    for index, leg in enumerate(trip.legs, start=1):
        stop_word = "stop" if leg.num_stops == 1 else "stops"
        print(
            f"  {index}. {leg.operator} {leg.line}: "
            f"{leg.board_stop} -> {leg.alight_stop} "
            f"(towards {leg.direction}), {leg.num_stops} {stop_word}, "
            f"about {_minutes(leg.minutes)}"
        )
        print(
            "     Distance: "
            f"{_distance(leg.distance_m, 'partly unknown')}"
        )

        walks_after = [walk for walk in trip.walks if walk.after_leg == index]
        for walk in walks_after:
            label = "to transfer" if walk.kind == "transfer" else "to destination"
            print(
                f"     Walk {_minutes(walk.minutes)} {label}: "
                f"{walk.from_place} -> {walk.to_place}"
            )

        if index < len(trip.legs):
            next_leg = trip.legs[index]
            print(f"     Transfer at {next_leg.board_stop} (wait included)")

    if trip.total_distance_m is not None:
        detail = _distance(trip.total_distance_m)
        if trip.straight_line_m is not None:
            detail += f" (straight line {_distance(trip.straight_line_m)})"
        print(f"  Total distance: {detail}")
    else:
        print("  Total distance: partly unknown (some stops have no coordinates)")


def _print_matches(label: str, matches) -> None:
    if not matches:
        return
    top = matches[0]
    extra = ""
    if len(matches) > 1:
        extra = f" (+{len(matches) - 1} other candidates)"
    print(f"  {label}: {top.name} (match {top.score:.0f}){extra}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    parser = argparse.ArgumentParser(
        prog="python -m koumba.cli",
        description="Find bus trips in Dakar with Koumba.",
    )
    parser.add_argument("origin", nargs="?", help="departure place")
    parser.add_argument("destination", nargs="?", help="arrival place")
    parser.add_argument(
        "--max-trips",
        type=int,
        default=3,
        help="number of distinct trips to show (default: 3)",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="print a short dataset schema summary and exit",
    )
    args = parser.parse_args(argv)

    data = load_data()
    if args.schema:
        print(describe_data(data))
        return 0

    if not args.origin or not args.destination:
        parser.error("origin and destination are required unless --schema is used")

    network = Network.from_data(data)
    place_index = PlaceIndex.from_data(data)
    engine = RouteEngine(network, place_index)

    print("Koumba route engine")
    print(f"From: {args.origin}")
    print(f"To:   {args.destination}")
    _print_matches("Departure match", place_index.match(args.origin))
    _print_matches("Arrival match", place_index.match(args.destination))

    try:
        trips = engine.find_trips(
            args.origin,
            args.destination,
            max_trips=max(1, args.max_trips),
        )
    except PlaceNotFoundError as exc:
        print()
        print(f"I do not know the place {exc.query!r}.")
        if exc.suggestions:
            suggestions = ", ".join(
                f"{name} ({score:.0f})" for name, score in exc.suggestions
            )
            print(f"Did you mean: {suggestions}?")
        return 1
    except NoRouteFoundError as exc:
        print()
        print(str(exc))
        return 1

    print()
    print(f"Found {len(trips)} trip(s):")
    for number, trip in enumerate(trips, start=1):
        _print_trip(number, trip)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
