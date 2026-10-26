"""Route search with direct trips first, then one or two transfers.

The search is Dijkstra over ``(line, direction, stop position)`` nodes.
The state also remembers how many transfers have been used and which
lines have been used, so we can return distinct line combinations and
never loop back onto a line we already rode.
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass

from .network import (
    WALK_SPEED_M_PER_MIN,
    Edge,
    LineKey,
    Network,
    Node,
    haversine_m,
)
from .places import (
    PlaceCandidate,
    PlaceIndex,
    normalize_name,
    prefer_exact_matches,
    representative_coordinates,
)

MAX_TRANSFERS = 2
ACCESS_WALK_RADIUS_M = 700.0
MAX_NEARBY_STOPS = 8
MAX_LABELS_PER_STATE = 12
MAX_EXPANSIONS = 250_000
DETOUR_LIMIT = 1.8
DOWNTOWN_RADIUS_M = 1_500.0
# A trip three times slower than the best option is not a useful
# alternative.  This also catches routes with missing coordinates that
# escape the distance-based detour check.
MAX_TIME_VS_BEST = 3.0

# These are the downtown / Plateau names.  A trip may pass through them
# when they are close to the start or the destination, but not as a random
# shortcut between two outer neighbourhoods.
DOWNTOWN_STOP_NAMES = {
    normalize_name(name)
    for name in (
        "Palais 1",
        "Palais 2",
        "Palais",
        "Centre-Ville",
        "Plateau",
        "Place Leclerc",
        "Place de l'Indépendance",
        "Médina",
    )
}


class RouteError(Exception):
    """Base class for route errors."""


class PlaceNotFoundError(RouteError):
    def __init__(self, query: str, suggestions: list[tuple[str, float]]) -> None:
        self.query = query
        self.suggestions = suggestions
        super().__init__(f"Unknown place: {query}")


class NoRouteFoundError(RouteError):
    pass


@dataclass(frozen=True)
class Walk:
    kind: str                      # "access", "transfer", or "egress"
    from_place: str
    to_place: str
    minutes: float
    distance_m: float | None = None
    after_leg: int = 0             # 0 = before the first leg


@dataclass(frozen=True)
class Leg:
    operator: str
    line: str
    board_stop: str
    alight_stop: str
    direction: str                 # terminus shown on the bus
    num_stops: int
    minutes: float
    distance_m: float | None = None


@dataclass(frozen=True)
class Trip:
    legs: tuple[Leg, ...]
    walks: tuple[Walk, ...]
    total_minutes: float
    transfers: int
    line_combo: tuple[LineKey, ...]
    origin_name: str
    destination_name: str
    total_distance_m: float | None
    straight_line_m: float | None
    detour: bool
    via_downtown: bool
    ridden_stops: tuple[str, ...]

    @property
    def walked_minutes(self) -> float:
        return sum(walk.minutes for walk in self.walks)


@dataclass(frozen=True)
class AccessPoint:
    stop: str
    walk: Walk


@dataclass(frozen=True)
class _Step:
    kind: str                      # "access", "ride", "transfer", "egress"
    from_stop: str
    to_stop: str
    minutes: float
    line: LineKey
    direction: int
    from_index: int | None
    to_index: int | None
    distance_m: float | None = None
    walk_minutes: float = 0.0
    walk_distance_m: float | None = None
    transfer_penalty: float = 0.0


@dataclass(frozen=True)
class _Label:
    node: Node
    transfers: int
    cost: float
    combo: tuple[LineKey, ...]
    steps: tuple[_Step, ...]
    rode_since_transfer: bool


class RouteEngine:
    def __init__(self, network: Network, place_index: PlaceIndex | None = None) -> None:
        self.network = network
        self.place_index = place_index or PlaceIndex.from_network(network)

    # ------------------------------------------------------------------
    # Endpoint preparation.
    # ------------------------------------------------------------------
    def _access_points(
        self,
        candidate: PlaceCandidate,
        kind: str,
    ) -> list[AccessPoint]:
        """Stops the user can reach from this place, with the walk."""

        points: dict[str, AccessPoint] = {}

        # Exact normalized stop names win: "Malika" and "Sahm" board at the
        # stop itself, without an invented walk.
        for stop in self.network.stops_for_normalized(candidate.normalized):
            points[stop] = AccessPoint(
                stop=stop,
                walk=Walk(
                    kind=kind,
                    from_place=candidate.name,
                    to_place=stop,
                    minutes=0.0,
                    distance_m=0.0,
                    after_leg=0,
                ),
            )

        # A place that is not itself a stop (for example "Palais", which is
        # near Palais 1 and Palais 2) can use the nearest stops.
        if not points and candidate.has_coordinates:
            origin = (float(candidate.lat), float(candidate.lon))
            nearby: list[tuple[float, str]] = []
            for stop in self.network.stop_names:
                stop_coords = self.network.coords(stop)
                if stop_coords is None:
                    continue
                distance = haversine_m(origin, stop_coords)
                if distance <= ACCESS_WALK_RADIUS_M:
                    nearby.append((distance, stop))
            nearby.sort()
            for distance, stop in nearby[:MAX_NEARBY_STOPS]:
                points[stop] = AccessPoint(
                    stop=stop,
                    walk=Walk(
                        kind=kind,
                        from_place=candidate.name,
                        to_place=stop,
                        minutes=distance / WALK_SPEED_M_PER_MIN,
                        distance_m=distance,
                        after_leg=0,
                    ),
                )

        return list(points.values())

    def _endpoint_map(
        self,
        candidates: list[PlaceCandidate],
        kind: str,
    ) -> dict[str, AccessPoint]:
        result: dict[str, AccessPoint] = {}
        for candidate in candidates:
            for point in self._access_points(candidate, kind):
                current = result.get(point.stop)
                if current is None or point.walk.minutes < current.walk.minutes:
                    result[point.stop] = point
        return result

    @staticmethod
    def _is_downtown_endpoint(
        candidates: list[PlaceCandidate],
        coords: tuple[float, float] | None,
        centre: tuple[float, float] | None,
    ) -> bool:
        if any(candidate.normalized in DOWNTOWN_STOP_NAMES for candidate in candidates):
            return True
        if centre is not None and coords is not None:
            return haversine_m(coords, centre) <= DOWNTOWN_RADIUS_M
        return False

    # ------------------------------------------------------------------
    # Public API.
    # ------------------------------------------------------------------
    def find_trips(
        self,
        origin: str,
        destination: str,
        max_trips: int = 3,
        operator: str | None = None,
    ) -> list[Trip]:
        max_trips = max(1, max_trips)
        origin_matches = prefer_exact_matches(
            self.place_index.match(origin),
            origin,
        )
        if not origin_matches:
            raise PlaceNotFoundError(origin, self.place_index.suggestions(origin))

        destination_matches = prefer_exact_matches(
            self.place_index.match(destination),
            destination,
        )
        if not destination_matches:
            raise PlaceNotFoundError(
                destination, self.place_index.suggestions(destination)
            )

        origin_points = self._endpoint_map(origin_matches, "access")
        destination_points = self._endpoint_map(destination_matches, "egress")
        if not origin_points:
            raise PlaceNotFoundError(origin, self.place_index.suggestions(origin))
        if not destination_points:
            raise PlaceNotFoundError(
                destination, self.place_index.suggestions(destination)
            )

        origin_coords = representative_coordinates(origin_matches)
        destination_coords = representative_coordinates(destination_matches)
        centre = self.network.downtown_centre
        origin_is_downtown = self._is_downtown_endpoint(
            origin_matches, origin_coords, centre
        )
        destination_is_downtown = self._is_downtown_endpoint(
            destination_matches, destination_coords, centre
        )

        trips = self._search(
            origin_points=origin_points,
            destination_points=destination_points,
            origin_name=origin,
            destination_name=destination,
            origin_coords=origin_coords,
            destination_coords=destination_coords,
            origin_is_downtown=origin_is_downtown,
            destination_is_downtown=destination_is_downtown,
            max_trips=max_trips,
            operator=operator,
        )
        if not trips:
            raise NoRouteFoundError(
                f"No bus trip found between {origin!r} and {destination!r}."
            )

        # Reject detours when a sane alternative exists.
        non_detour = [trip for trip in trips if not trip.detour]
        if non_detour:
            trips = non_detour

        # Extra guard for this network: do not cross downtown just to
        # connect two outer neighbourhoods.
        non_downtown = [trip for trip in trips if not trip.via_downtown]
        if non_downtown:
            trips = non_downtown

        # Do not offer an absurdly slow "alternative" just because it has
        # fewer transfers.  If every remaining trip is slow, keep them.
        if trips:
            fastest = min(trip.total_minutes for trip in trips)
            sane = [
                trip
                for trip in trips
                if trip.total_minutes <= fastest * MAX_TIME_VS_BEST
            ]
            if sane:
                trips = sane

        trips.sort(
            key=lambda trip: (
                trip.transfers,
                trip.total_minutes,
                trip.total_distance_m
                if trip.total_distance_m is not None
                else float("inf"),
            )
        )
        return trips[:max_trips]

    # ------------------------------------------------------------------
    # Search.
    # ------------------------------------------------------------------
    def _search(
        self,
        origin_points: dict[str, AccessPoint],
        destination_points: dict[str, AccessPoint],
        origin_name: str,
        destination_name: str,
        origin_coords: tuple[float, float] | None,
        destination_coords: tuple[float, float] | None,
        origin_is_downtown: bool,
        destination_is_downtown: bool,
        max_trips: int,
        operator: str | None = None,
    ) -> list[Trip]:
        queue: list[tuple[tuple[int, float], int, _Label]] = []
        counter = itertools.count()
        best: dict[tuple[Node, int], list[_Label]] = defaultdict(list)
        recorded: dict[tuple[LineKey, ...], Trip] = {}
        accepted: list[Trip] = []

        for stop, access in origin_points.items():
            for node in self.network.nodes_at_stop(stop):
                if operator is not None and node.line[0] != operator:
                    continue
                access_step = _Step(
                    kind="access",
                    from_stop=access.walk.from_place,
                    to_stop=stop,
                    minutes=access.walk.minutes,
                    line=node.line,
                    direction=node.direction,
                    from_index=None,
                    to_index=node.index,
                    walk_minutes=access.walk.minutes,
                    walk_distance_m=access.walk.distance_m,
                )
                label = _Label(
                    node=node,
                    transfers=0,
                    cost=access.walk.minutes,
                    combo=(node.line,),
                    steps=(access_step,),
                    rode_since_transfer=False,
                )
                self._push(queue, best, counter, label)

        expansions = 0
        while queue:
            (transfers_so_far, cost_so_far), _, label = heapq.heappop(queue)
            state = (label.node, label.transfers)
            if not any(existing is label for existing in best[state]):
                continue

            expansions += 1
            if expansions > MAX_EXPANSIONS:
                break

            # If the queue cannot beat the worst of our best trips, there
            # is no point in expanding more nodes.
            if len(accepted) >= max_trips:
                worst = accepted[-1]
                if (transfers_so_far, cost_so_far) >= (
                    worst.transfers,
                    worst.total_minutes,
                ):
                    break

            stop = self.network.stop_for(label.node)
            destination = destination_points.get(stop)
            if destination is not None and label.rode_since_transfer:
                trip = self._build_trip(
                    label=label,
                    destination=destination,
                    origin_name=origin_name,
                    destination_name=destination_name,
                    origin_coords=origin_coords,
                    destination_coords=destination_coords,
                    origin_is_downtown=origin_is_downtown,
                    destination_is_downtown=destination_is_downtown,
                )
                current = recorded.get(trip.line_combo)
                if current is None or self._trip_rank(trip) < self._trip_rank(current):
                    recorded[trip.line_combo] = trip

                if not trip.detour and not trip.via_downtown:
                    accepted.append(trip)
                    accepted = self._best_accepted(accepted, max_trips)

            for edge in self.network.adjacency[label.node]:
                if edge.kind == "ride":
                    self._push(
                        queue,
                        best,
                        counter,
                        self._advance_ride(label, edge),
                    )
                else:
                    if label.transfers >= MAX_TRANSFERS:
                        continue
                    if not label.rode_since_transfer:
                        continue
                    if operator is not None and edge.target.line[0] != operator:
                        continue
                    if edge.target.line in label.combo:
                        continue
                    self._push(
                        queue,
                        best,
                        counter,
                        self._advance_transfer(label, edge),
                    )

        return list(recorded.values())

    @staticmethod
    def _trip_rank(trip: Trip) -> tuple[bool, bool, int, float]:
        # Prefer a clean trip over a detour, and a non-downtown trip over a
        # downtown shortcut, even if the clean version is slightly slower.
        return (
            trip.detour,
            trip.via_downtown,
            trip.transfers,
            trip.total_minutes,
        )

    @staticmethod
    def _best_accepted(trips: list[Trip], max_trips: int) -> list[Trip]:
        unique: dict[tuple[LineKey, ...], Trip] = {}
        for trip in trips:
            current = unique.get(trip.line_combo)
            if current is None or (
                trip.transfers,
                trip.total_minutes,
            ) < (current.transfers, current.total_minutes):
                unique[trip.line_combo] = trip

        if not unique:
            return []
        fastest = min(trip.total_minutes for trip in unique.values())
        reasonable = [
            trip
            for trip in unique.values()
            if trip.total_minutes <= fastest * MAX_TIME_VS_BEST
        ]
        ordered = sorted(
            reasonable,
            key=lambda trip: (
                trip.transfers,
                trip.total_minutes,
                trip.total_distance_m
                if trip.total_distance_m is not None
                else float("inf"),
            ),
        )
        return ordered[:max_trips]

    def _push(
        self,
        queue: list[tuple[tuple[int, float], int, _Label]],
        best: dict[tuple[Node, int], list[_Label]],
        counter,
        label: _Label,
    ) -> None:
        state = (label.node, label.transfers)
        current = best[state]

        # A state can be reached with different line combinations.  Keep
        # a small set of them, but never keep the same combination twice.
        for existing in list(current):
            if (
                existing.combo == label.combo
                and existing.rode_since_transfer == label.rode_since_transfer
            ):
                if existing.cost <= label.cost + 1e-9:
                    return
                current.remove(existing)

        current.append(label)
        current.sort(key=lambda item: item.cost)
        if len(current) > MAX_LABELS_PER_STATE:
            del current[MAX_LABELS_PER_STATE:]

        heapq.heappush(
            queue,
            ((label.transfers, label.cost), next(counter), label),
        )

    def _advance_ride(self, label: _Label, edge: Edge) -> _Label:
        from_stop = self.network.stop_for(label.node)
        to_stop = self.network.stop_for(edge.target)
        step = _Step(
            kind="ride",
            from_stop=from_stop,
            to_stop=to_stop,
            minutes=edge.minutes,
            line=label.node.line,
            direction=label.node.direction,
            from_index=label.node.index,
            to_index=edge.target.index,
            distance_m=edge.distance_m,
        )
        return _Label(
            node=edge.target,
            transfers=label.transfers,
            cost=label.cost + edge.minutes,
            combo=label.combo,
            steps=label.steps + (step,),
            rode_since_transfer=True,
        )

    def _advance_transfer(self, label: _Label, edge: Edge) -> _Label:
        from_stop = self.network.stop_for(label.node)
        to_stop = self.network.stop_for(edge.target)
        step = _Step(
            kind="transfer",
            from_stop=from_stop,
            to_stop=to_stop,
            minutes=edge.minutes,
            line=edge.target.line,
            direction=edge.target.direction,
            from_index=label.node.index,
            to_index=edge.target.index,
            walk_minutes=edge.walk_minutes,
            walk_distance_m=edge.walk_distance_m,
            transfer_penalty=edge.transfer_penalty,
        )
        return _Label(
            node=edge.target,
            transfers=label.transfers + 1,
            cost=label.cost + edge.minutes,
            combo=label.combo + (edge.target.line,),
            steps=label.steps + (step,),
            rode_since_transfer=False,
        )

    # ------------------------------------------------------------------
    # Turning a search label into a friendly Trip.
    # ------------------------------------------------------------------
    def _build_trip(
        self,
        label: _Label,
        destination: AccessPoint,
        origin_name: str,
        destination_name: str,
        origin_coords: tuple[float, float] | None,
        destination_coords: tuple[float, float] | None,
        origin_is_downtown: bool,
        destination_is_downtown: bool,
    ) -> Trip:
        alight_stop = self.network.stop_for(label.node)
        egress_step = _Step(
            kind="egress",
            from_stop=alight_stop,
            to_stop=destination.walk.to_place,
            minutes=destination.walk.minutes,
            line=label.node.line,
            direction=label.node.direction,
            from_index=label.node.index,
            to_index=None,
            walk_minutes=destination.walk.minutes,
            walk_distance_m=destination.walk.distance_m,
        )
        steps = label.steps + (egress_step,)

        legs: list[Leg] = []
        walks: list[Walk] = []
        ridden_stops: list[str] = []
        current: dict | None = None

        def finish_leg() -> None:
            nonlocal current
            if current is None:
                return
            line = self.network.lines[current["line"]]
            legs.append(
                Leg(
                    operator=line.company,
                    line=line.name,
                    board_stop=current["board_stop"],
                    alight_stop=current["alight_stop"],
                    direction=line.terminus_for(current["direction"]),
                    num_stops=abs(
                        int(current["end_index"]) - int(current["start_index"])
                    ),
                    minutes=current["minutes"],
                    distance_m=current["distance"],
                )
            )
            current = None

        for step in steps:
            if step.kind == "access":
                if step.walk_minutes > 0 or (step.walk_distance_m or 0) > 0:
                    walks.append(
                        Walk(
                            kind="access",
                            from_place=step.from_stop,
                            to_place=step.to_stop,
                            minutes=step.walk_minutes,
                            distance_m=step.walk_distance_m,
                            after_leg=0,
                        )
                    )
            elif step.kind == "ride":
                if (
                    current is None
                    or current["line"] != step.line
                    or current["direction"] != step.direction
                ):
                    finish_leg()
                    current = {
                        "line": step.line,
                        "direction": step.direction,
                        "board_stop": step.from_stop,
                        "alight_stop": step.to_stop,
                        "start_index": step.from_index,
                        "end_index": step.to_index,
                        "minutes": step.minutes,
                        "distance": step.distance_m,
                    }
                else:
                    current["alight_stop"] = step.to_stop
                    current["end_index"] = step.to_index
                    current["minutes"] += step.minutes
                    if (
                        current["distance"] is None
                        or step.distance_m is None
                    ):
                        current["distance"] = None
                    else:
                        current["distance"] += step.distance_m
                for stop in (step.from_stop, step.to_stop):
                    if not ridden_stops or ridden_stops[-1] != stop:
                        ridden_stops.append(stop)
            elif step.kind == "transfer":
                finish_leg()
                if step.walk_minutes > 0 or (step.walk_distance_m or 0) > 0:
                    walks.append(
                        Walk(
                            kind="transfer",
                            from_place=step.from_stop,
                            to_place=step.to_stop,
                            minutes=step.walk_minutes,
                            distance_m=step.walk_distance_m,
                            after_leg=len(legs),
                        )
                    )
            elif step.kind == "egress":
                finish_leg()
                if step.walk_minutes > 0 or (step.walk_distance_m or 0) > 0:
                    walks.append(
                        Walk(
                            kind="egress",
                            from_place=step.from_stop,
                            to_place=step.to_stop,
                            minutes=step.walk_minutes,
                            distance_m=step.walk_distance_m,
                            after_leg=len(legs),
                        )
                    )

        finish_leg()

        total_minutes = sum(step.minutes for step in steps)

        total_distance: float | None = 0.0
        for step in steps:
            if step.kind == "ride":
                if step.distance_m is None:
                    total_distance = None
                    break
                total_distance += step.distance_m
            else:
                total_distance += step.walk_distance_m or 0.0

        straight_line: float | None = None
        if origin_coords is not None and destination_coords is not None:
            straight_line = haversine_m(origin_coords, destination_coords)

        detour = bool(
            total_distance is not None
            and straight_line is not None
            and total_distance > DETOUR_LIMIT * straight_line
        )
        via_downtown = self._passes_downtown(
            ridden_stops,
            origin_is_downtown=origin_is_downtown,
            destination_is_downtown=destination_is_downtown,
        )

        return Trip(
            legs=tuple(legs),
            walks=tuple(walks),
            total_minutes=total_minutes,
            transfers=label.transfers,
            line_combo=label.combo,
            origin_name=origin_name,
            destination_name=destination_name,
            total_distance_m=total_distance,
            straight_line_m=straight_line,
            detour=detour,
            via_downtown=via_downtown,
            ridden_stops=tuple(ridden_stops),
        )

    def _passes_downtown(
        self,
        ridden_stops: list[str],
        origin_is_downtown: bool,
        destination_is_downtown: bool,
    ) -> bool:
        if origin_is_downtown or destination_is_downtown:
            return False

        if any(normalize_name(stop) in DOWNTOWN_STOP_NAMES for stop in ridden_stops):
            return True

        centre = self.network.downtown_centre
        if centre is None:
            return False
        for stop in ridden_stops:
            coords = self.network.coords(stop)
            if coords is not None and haversine_m(coords, centre) <= DOWNTOWN_RADIUS_M:
                return True
        return False
