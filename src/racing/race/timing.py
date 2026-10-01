"""Live standings and gaps measured from shared, lap-aware track positions."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from math import isfinite

DEFAULT_TIMING_HISTORY_POINTS = 8192


@dataclass(frozen=True, slots=True)
class TimingSample:
    """One car's signed, lap-aware track position at the current physics timestamp."""

    car_id: str
    distance_m: float
    eliminated: bool = False
    discontinuity: bool = False


@dataclass(frozen=True, slots=True)
class TimingStanding:
    """A stable position, leader gap, and interval to the car immediately ahead."""

    rank: int
    car_id: str
    distance_m: float
    gap_seconds: float | None
    eliminated: bool = False
    interval_seconds: float | None = None


@dataclass(slots=True)
class _Crossing:
    distance_m: float
    first_time_seconds: float
    last_time_seconds: float


class _CarHistory:
    """Monotonic distance crossings, retaining both ends of stationary periods."""

    def __init__(self, *, max_points: int) -> None:
        self._max_points = max_points
        self._crossings: list[_Crossing] = []
        self._start = 0

    def update(self, elapsed_seconds: float, sample: TimingSample) -> None:
        previous = self._crossings[-1] if self._crossings else None
        if (
            previous is None
            or sample.discontinuity
            or sample.distance_m < previous.distance_m
            or (elapsed_seconds == previous.last_time_seconds and sample.distance_m != previous.distance_m)
        ):
            self._crossings = [_Crossing(sample.distance_m, elapsed_seconds, elapsed_seconds)]
            self._start = 0
            return
        if sample.distance_m == previous.distance_m:
            previous.last_time_seconds = elapsed_seconds
            return

        self._crossings.append(_Crossing(sample.distance_m, elapsed_seconds, elapsed_seconds))
        if len(self._crossings) - self._start > self._max_points:
            self._start += 1
        # Compact in batches instead of shifting a long list every physics tick.
        # At most twice max_points objects are retained between compactions.
        if self._start >= self._max_points:
            self._crossings = self._crossings[self._start :]
            self._start = 0

    def crossing_time(self, distance_m: float) -> float | None:
        if not self._crossings or distance_m < self._crossings[self._start].distance_m:
            return None
        index = bisect_left(self._crossings, distance_m, lo=self._start, key=lambda crossing: crossing.distance_m)
        if index == len(self._crossings):
            return None
        after = self._crossings[index]
        if distance_m == after.distance_m:
            return after.first_time_seconds
        if index == self._start:
            return None
        before = self._crossings[index - 1]
        fraction = (distance_m - before.distance_m) / (after.distance_m - before.distance_m)
        return before.last_time_seconds + fraction * (after.first_time_seconds - before.last_time_seconds)


class RaceTiming:
    """Track standings, leader gaps, and intervals to the car immediately ahead.

    Call update once per physics step with all cars, using race_track_position_m
    from a shared start line. Positions before the line are negative; completed
    laps continue past the track length. A gap is the current timestamp minus
    the time the leader first crossed the trailing car's current position.
    Linear interpolation connects consecutive observed position increases; speed
    is never used to estimate a gap. Missing history yields None.
    Intervals use the same measurement against the preceding car's own history.

    Each car retains max_history_points distinct position samples. Backward
    movement and explicit discontinuities discard that car's old crossings,
    so timing never interpolates through a reversal or marshal teleport. Reset
    the model between races. Cars absent from an update lose their history.
    """

    def __init__(self, *, max_history_points: int = DEFAULT_TIMING_HISTORY_POINTS) -> None:
        if max_history_points < 2:
            raise ValueError("max_history_points must be at least two")
        self._max_history_points = max_history_points
        self._histories: dict[str, _CarHistory] = {}
        self._retired_at: dict[str, float] = {}
        self._rows: tuple[TimingStanding, ...] = ()
        self._elapsed_seconds: float | None = None

    @property
    def rows(self) -> tuple[TimingStanding, ...]:
        return self._rows

    def reset(self) -> None:
        """Discard standings and crossing history before a new race."""
        self._histories.clear()
        self._retired_at.clear()
        self._rows = ()
        self._elapsed_seconds = None

    def update(self, elapsed_seconds: float, samples: tuple[TimingSample, ...]) -> tuple[TimingStanding, ...]:
        """Rank active cars by track position, then retirees from newest to oldest."""
        if not isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if self._elapsed_seconds is not None and elapsed_seconds < self._elapsed_seconds:
            raise ValueError("elapsed_seconds cannot go backward; reset timing between races")
        car_ids: set[str] = set()
        for sample in samples:
            if not sample.car_id:
                raise ValueError("car_id must be nonempty")
            if sample.car_id in car_ids:
                raise ValueError(f"duplicate timing car_id: {sample.car_id}")
            if not isfinite(sample.distance_m):
                raise ValueError("distance_m must be finite")
            car_ids.add(sample.car_id)

        self._elapsed_seconds = elapsed_seconds
        self._histories = {car_id: history for car_id, history in self._histories.items() if car_id in car_ids}
        self._retired_at = {car_id: time for car_id, time in self._retired_at.items() if car_id in car_ids}
        for sample in samples:
            if sample.eliminated:
                self._retired_at.setdefault(sample.car_id, elapsed_seconds)
            else:
                self._retired_at.pop(sample.car_id, None)
            if sample.car_id not in self._histories:
                self._histories[sample.car_id] = _CarHistory(max_points=self._max_history_points)
            self._histories[sample.car_id].update(elapsed_seconds, sample)

        ordered = sorted(
            samples,
            key=lambda sample: (
                sample.eliminated,
                -self._retired_at[sample.car_id] if sample.eliminated else -sample.distance_m,
                sample.car_id,
            ),
        )
        if not ordered:
            self._rows = ()
            return self._rows
        leader = ordered[0]
        rows: list[TimingStanding] = []
        for rank, sample in enumerate(ordered, start=1):
            rows.append(
                TimingStanding(
                    rank=rank,
                    car_id=sample.car_id,
                    distance_m=sample.distance_m,
                    gap_seconds=self._gap_seconds(elapsed_seconds, sample, leader),
                    eliminated=sample.eliminated,
                    interval_seconds=(
                        None if rank == 1 else self._gap_seconds(elapsed_seconds, sample, ordered[rank - 2])
                    ),
                )
            )
        self._rows = tuple(rows)
        return self._rows

    def gap_to(self, car_id: str, ahead_id: str) -> float | None:
        """Measure against an explicit car when finish order overrides track order."""
        rows = {row.car_id: row for row in self._rows}
        if self._elapsed_seconds is None or car_id not in rows or ahead_id not in rows:
            return None
        return self._gap_seconds(self._elapsed_seconds, rows[car_id], rows[ahead_id])

    def _gap_seconds(
        self, elapsed_seconds: float, sample: TimingSample | TimingStanding, ahead: TimingSample | TimingStanding,
    ) -> float | None:
        if sample.eliminated or ahead.eliminated:
            return None
        if sample.distance_m == ahead.distance_m:
            return 0.0
        crossing = self._histories[ahead.car_id].crossing_time(sample.distance_m)
        return None if crossing is None else max(0.0, elapsed_seconds - crossing)
