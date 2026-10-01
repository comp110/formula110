"""Auditable race-interest heuristics, independent of physics and race classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from math import exp, isfinite

from racing.race.heat import HeatRaceSnapshot, HeatStanding


@dataclass(frozen=True, slots=True)
class InterestWeights:
    lead: float = 30.0
    overtakes: float = 30.0
    finish: float = 40.0
    finish_gap_scale_seconds: float = 2.0

    def __post_init__(self) -> None:
        weights = (self.lead, self.overtakes, self.finish)
        if (
            any(not isfinite(value) or value < 0 for value in weights)
            or not isfinite(sum(weights))
            or sum(weights) <= 0
        ):
            raise ValueError("weights must be finite, nonnegative, and have a positive sum")
        if not isfinite(self.finish_gap_scale_seconds) or self.finish_gap_scale_seconds <= 0:
            raise ValueError("finish gap scale must be finite and positive")


DEFAULT_INTEREST_WEIGHTS = InterestWeights()


@dataclass(frozen=True, slots=True)
class InterestScore:
    score: float
    lead_points: float
    overtake_points: float
    finish_points: float
    lead_changes: int
    overtakes: int
    finishers: int
    gap_p2_seconds: float | None
    gap_p3_seconds: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def score_interest(
    *,
    lead_changes: int,
    overtakes: int,
    round_laps: int,
    entrant_count: int,
    finish_times: tuple[float, ...],
    weights: InterestWeights = DEFAULT_INTEREST_WEIGHTS,
) -> InterestScore:
    """Score rates with diminishing returns and the mean P2/P3 gap to P1.

    A lead change is also an overtake: the lead term is an intentional bonus.
    One lead change/lap and one pass/opponent/lap each earn half their term.
    Fewer than three finishers earns zero finish points, never a fabricated gap.
    Compare like-for-like tracks, lap counts, fields and damage settings first.
    """
    if round_laps < 1 or entrant_count < 2 or lead_changes < 0 or overtakes < 0:
        raise ValueError("invalid race length, field size, or event counts")
    if len(finish_times) > entrant_count or any(not isfinite(t) or t < 0 for t in finish_times):
        raise ValueError("invalid finish times")
    times = sorted(finish_times)
    gap2 = times[1] - times[0] if len(times) >= 2 else None
    gap3 = times[2] - times[0] if len(times) >= 3 else None
    lead = lead_changes / (lead_changes + round_laps)
    passing = overtakes / (overtakes + round_laps * (entrant_count - 1))
    finish = 0.0 if gap2 is None or gap3 is None else exp(-(gap2 + gap3) / (2 * weights.finish_gap_scale_seconds))
    scale = 100.0 / (weights.lead + weights.overtakes + weights.finish)
    points = (scale * weights.lead * lead, scale * weights.overtakes * passing, scale * weights.finish * finish)
    return InterestScore(sum(points), *points, lead_changes, overtakes, len(times), gap2, gap3)


@dataclass(frozen=True, slots=True)
class InterestEvent:
    kind: str
    time_seconds: float
    confirmed_at_seconds: float
    overtaker: str
    passed: str


@dataclass(slots=True)
class _Order:
    ahead: str
    candidate: str | None = None
    since: float = 0.0


class RaceInterestTracker:
    """Count sustained position exchanges in unwrapped, lap-aware progress.

    Require a 0.5 m advantage held for 0.3 s. Rebaseline affected pairs after
    teleports and for a 2 s cooldown. Retired/finished cars cannot be passed for
    credit. Initial grid order is a baseline, not an event; lapping is not a
    race-position exchange. Very brief/last-instant swaps are intentionally
    filtered. Keep one tracker per race.
    """

    def __init__(
        self,
        *,
        pass_margin_m: float = 0.5,
        confirmation_seconds: float = 0.3,
        recovery_seconds: float = 2.0,
    ) -> None:
        if any(not isfinite(value) or value < 0 for value in (pass_margin_m, confirmation_seconds, recovery_seconds)):
            raise ValueError("event filters must be finite and nonnegative")
        self.pass_margin_m = pass_margin_m
        self.confirmation_seconds = confirmation_seconds
        self.recovery_seconds = recovery_seconds
        self.events: list[InterestEvent] = []
        self._pairs: dict[tuple[str, str], _Order] = {}
        self._leader: _Order | None = None
        self._blocked_until: dict[str, float] = {}
        self._race_index: int | None = None
        self._time = -1.0

    @property
    def lead_changes(self) -> int:
        return sum(event.kind == "lead_change" for event in self.events)

    @property
    def overtakes(self) -> int:
        return sum(event.kind == "overtake" for event in self.events)

    def _confirm(self, order: _Order, candidate: str, advantage: float, now: float, kind: str) -> None:
        if candidate == order.ahead or advantage < self.pass_margin_m:
            order.candidate = None
            return
        if order.candidate != candidate:
            order.candidate = candidate
            order.since = now
        if now - order.since + 1e-9 >= self.confirmation_seconds:
            self.events.append(InterestEvent(kind, order.since, now, candidate, order.ahead))
            order.ahead = candidate
            order.candidate = None

    def update(self, snapshot: HeatRaceSnapshot) -> None:
        now = snapshot.elapsed_seconds
        if not isfinite(now) or now < self._time:
            raise ValueError("telemetry timestamps must be finite and nondecreasing")
        if self._race_index is not None and self._race_index != snapshot.race_index:
            raise ValueError("use a fresh interest tracker for each race")
        self._race_index = snapshot.race_index
        self._time = now
        for sample in snapshot.samples:
            if sample.discontinuity:
                self._blocked_until[sample.car_id] = now + self.recovery_seconds
        active = {
            sample.car_id: sample.distance_m
            for sample in snapshot.samples
            if not sample.eliminated and sample.car_id not in snapshot.finished_car_ids
        }
        eligible = {
            car_id: distance for car_id, distance in active.items() if now > self._blocked_until.get(car_id, -1.0)
        }
        current_pairs = set(combinations(sorted(eligible), 2))
        self._pairs = {pair: state for pair, state in self._pairs.items() if pair in current_pairs}
        for pair in sorted(current_pairs):
            first, second = pair
            ahead = first if eligible[first] >= eligible[second] else second
            state = self._pairs.setdefault(pair, _Order(ahead))
            self._confirm(state, ahead, abs(eligible[first] - eligible[second]), now, "overtake")

        # Once P1 finishes, the next unfinished car is not a new race leader.
        if snapshot.finished_car_ids or not active:
            self._leader = None
            return
        leader = min(active, key=lambda car_id: (-active[car_id], car_id))
        if self._leader is None or self._leader.ahead not in eligible or leader not in eligible:
            self._leader = _Order(leader)
            return
        self._confirm(self._leader, leader, active[leader] - active[self._leader.ahead], now, "lead_change")

    def score(self, standings: tuple[HeatStanding, ...], *, round_laps: int, weights: InterestWeights) -> InterestScore:
        return score_interest(
            lead_changes=self.lead_changes,
            overtakes=self.overtakes,
            round_laps=round_laps,
            entrant_count=len(standings),
            finish_times=tuple(
                standing.finish_time_seconds
                for standing in standings
                if not standing.dnf and standing.finish_time_seconds is not None
            ),
            weights=weights,
        )
