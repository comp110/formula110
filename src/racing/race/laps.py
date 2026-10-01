"""Finish-line classification for lap-limited races, shared by both runners."""

from __future__ import annotations

from dataclasses import replace
from math import isfinite

from racing.race.timing import RaceTiming, TimingSample, TimingStanding

DEFAULT_FINISH_TIMEOUT_SECONDS = 20.0


def validate_round_laps(round_laps: int | None) -> None:
    if round_laps is not None and (type(round_laps) is not int or round_laps < 1):
        raise ValueError("round_laps must be a positive integer")


def validate_finish_timeout_seconds(finish_timeout_seconds: float) -> None:
    if not isfinite(finish_timeout_seconds) or finish_timeout_seconds < 0.0:
        raise ValueError("finish_timeout_seconds must be finite and nonnegative")


class LapRace:
    """Lock places at each car's final forward crossing of the shared line.

    Samples start behind the grid's shared line, so its initial crossing does
    not complete a lap. Interpolated crossing times resolve same-tick finishes;
    exact ties use the stable car ID. Eliminations and the winner's configurable
    deadline produce DNFs without taking a finishing place.
    """

    def __init__(
        self,
        *,
        round_laps: int,
        track_length_m: float,
        finish_timeout_seconds: float = DEFAULT_FINISH_TIMEOUT_SECONDS,
    ) -> None:
        validate_round_laps(round_laps)
        validate_finish_timeout_seconds(finish_timeout_seconds)
        if not isfinite(track_length_m) or track_length_m <= 0:
            raise ValueError("track_length_m must be finite and positive")
        self.round_laps = round_laps
        self.track_length_m = track_length_m
        self.finish_distance_m = round_laps * track_length_m
        self.finish_timeout_seconds = finish_timeout_seconds
        self.finish_times: dict[str, float] = {}
        self.dnf: set[str] = set()
        self.complete = False
        self.rows: tuple[TimingStanding, ...] = ()
        self.elapsed_seconds = 0.0
        self._previous: dict[str, TimingSample] = {}
        self._timing = RaceTiming()

    @property
    def deadline_seconds(self) -> float | None:
        if not self.finish_times:
            return None
        return min(self.finish_times.values()) + self.finish_timeout_seconds

    @property
    def clock_text(self) -> str:
        if self.complete:
            return "FINISH"
        if (deadline := self.deadline_seconds) is not None:
            return f"{max(0.0, deadline - self.elapsed_seconds):04.1f}s"
        leader_distance = max((row.distance_m for row in self.rows if not row.eliminated), default=0.0)
        lap = min(self.round_laps, max(0, int(leader_distance // self.track_length_m)) + 1)
        return f"LAP {lap}/{self.round_laps}"

    def finish_position(self, car_id: str) -> int | None:
        ordered = sorted(self.finish_times, key=lambda key: (self.finish_times[key], key))
        return ordered.index(car_id) + 1 if car_id in self.finish_times else None

    def completed_laps(self, car_id: str) -> int:
        if car_id in self.finish_times:
            return self.round_laps
        distance = next(row.distance_m for row in self.rows if row.car_id == car_id)
        return min(self.round_laps, max(0, int(distance // self.track_length_m)))

    def update(self, *, elapsed_seconds: float, samples: tuple[TimingSample, ...]) -> None:
        if self.complete:
            return
        # Validate timestamps and identities through the shared timing model.
        self._timing.update(elapsed_seconds, samples)
        if self._previous and set(self._previous) != {sample.car_id for sample in samples}:
            raise ValueError("lap race entrants cannot change during a round")
        crossings: list[tuple[float, str]] = []
        for sample in samples:
            if sample.car_id in self.finish_times or sample.car_id in self.dnf:
                continue
            if sample.eliminated:
                self.dnf.add(sample.car_id)
                continue
            before = self._previous.get(sample.car_id)
            if (
                before is not None
                and not sample.discontinuity
                and before.distance_m < self.finish_distance_m <= sample.distance_m
            ):
                fraction = (self.finish_distance_m - before.distance_m) / (sample.distance_m - before.distance_m)
                crossing = self.elapsed_seconds + fraction * (elapsed_seconds - self.elapsed_seconds)
                crossings.append((crossing, sample.car_id))
        for crossing, car_id in sorted(crossings):
            deadline = self.deadline_seconds
            if deadline is None or crossing <= deadline:
                self.finish_times[car_id] = crossing
        deadline = self.deadline_seconds
        if deadline is not None and elapsed_seconds >= deadline:
            self.dnf.update(sample.car_id for sample in samples if sample.car_id not in self.finish_times)
        self.complete = bool(samples) and len(self.finish_times) + len(self.dnf) == len(samples)
        self.elapsed_seconds = elapsed_seconds
        self._previous = {sample.car_id: sample for sample in samples}

        ordered = sorted(
            self._timing.rows,
            key=lambda row: (
                0 if row.car_id in self.finish_times else 2 if row.car_id in self.dnf else 1,
                self.finish_times.get(row.car_id, float(row.rank)),
                row.car_id,
            ),
        )
        winner_time = min(self.finish_times.values(), default=0.0)
        self.rows = tuple(
            replace(
                row,
                rank=rank,
                distance_m=self.finish_distance_m if row.car_id in self.finish_times else row.distance_m,
                eliminated=row.car_id in self.dnf,
                gap_seconds=self.finish_times[row.car_id] - winner_time
                if row.car_id in self.finish_times
                else None if self.finish_times else row.gap_seconds,
                interval_seconds=(
                    self.finish_times[row.car_id] - self.finish_times[ordered[rank - 2].car_id]
                    if row.car_id in self.finish_times
                    else self._timing.gap_to(row.car_id, ordered[rank - 2].car_id)
                    if ordered[rank - 2].car_id not in self.finish_times else None
                ) if rank > 1 and row.car_id not in self.dnf else None,
            )
            for rank, row in enumerate(ordered, start=1)
        )
