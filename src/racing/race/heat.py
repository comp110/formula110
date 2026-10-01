"""Heats for two to twenty controllers using shared racing physics and scoring."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from importlib import import_module
from math import isclose
from typing import Any, cast

from racing.graphics.colors import UNC_CAROLINA_BLUE, ColorRGBA
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, PhysicsScene, create_physics_world, create_robot_vehicle
from racing.race.head_to_head import (
    _run_headless_student_runtime_for_duration,  # pyright: ignore[reportPrivateUsage]
    controller_for_copy,
)
from racing.race.laps import (
    DEFAULT_FINISH_TIMEOUT_SECONDS,
    LapRace,
    validate_finish_timeout_seconds,
    validate_round_laps,
)
from racing.race.progress import TrackProgressModel, resolve_track
from racing.race.rules import HeadToHeadRaceRules
from racing.race.runtime import (
    DEFAULT_RACE_RANDOM_SEED,
    RaceCarRuntime,
    RaceRecoveryConfig,
    lap_progress_tracker_for_spawn_pose,
    race_scored_distance_m,
    race_spawn_poses,
    race_track_position_m,
    robot_score_damage,
    seeded_race_start_finish_pose,
)
from racing.race.timing import TimingSample
from racing.student.api import RobotController
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint

HEAT_ENTRANT_COUNTS = tuple(range(2, 21))
HEAT_RESULT_SCHEMA_VERSION = 1
DEFAULT_HEAT_COLORS: tuple[ColorRGBA, ...] = (
    UNC_CAROLINA_BLUE,
    (1.0, 0.78, 0.12, 1.0),
    (0.91, 0.22, 0.52, 1.0),
    (0.22, 0.78, 0.40, 1.0),
    (0.96, 0.42, 0.12, 1.0),
    (0.60, 0.39, 0.93, 1.0),
    (0.18, 0.86, 0.85, 1.0),
    (0.93, 0.93, 0.96, 1.0),
    (0.93, 0.12, 0.16, 1.0),
    (0.64, 0.86, 0.18, 1.0),
    (0.28, 0.36, 0.93, 1.0),
    (0.98, 0.64, 0.72, 1.0),
    (0.08, 0.48, 0.35, 1.0),
    (0.68, 0.40, 0.20, 1.0),
    (0.75, 0.68, 0.98, 1.0),
    (0.55, 0.84, 0.96, 1.0),
    (0.56, 0.12, 0.26, 1.0),
    (0.93, 0.48, 0.83, 1.0),
    (0.62, 0.66, 0.71, 1.0),
    (0.98, 0.89, 0.56, 1.0),
)


@dataclass(frozen=True, slots=True)
class HeatEntrant:
    """One independently controlled car and its display metadata."""

    name: str
    controller: RobotController
    team_color: ColorRGBA


@dataclass(frozen=True, slots=True)
class HeatRaceEntry:
    """Stable entrant identity assigned to its input-order starting-grid slot."""

    entrant_index: int

    @property
    def role(self) -> str:
        return f"heat-{self.entrant_index}"

    @property
    def copy_index(self) -> int:
        return 0

    @property
    def car_id(self) -> str:
        return f"{self.role}:{self.copy_index}"


@dataclass(frozen=True, slots=True)
class HeatStanding:
    """Individual progress, finish classification, and totals for a race or suite."""

    entrant_index: int
    name: str
    distance_m: float
    lap_count: int
    damage: float
    marshal_count: int
    finish_position: int | None = None
    finish_time_seconds: float | None = None
    dnf: bool = False
    placement_total: int | None = None
    dnf_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "entrant_index": self.entrant_index,
            "name": self.name,
            "distance_m": self.distance_m,
            "lap_count": self.lap_count,
            "damage": self.damage,
            "marshal_count": self.marshal_count,
            "finish_position": self.finish_position,
            "finish_time_seconds": self.finish_time_seconds,
            "dnf": self.dnf,
            "placement_total": self.placement_total,
            "dnf_count": self.dnf_count,
        }


@dataclass(frozen=True, slots=True)
class HeatRaceResult:
    """Standings for one completed heat."""

    race_index: int
    standings: tuple[HeatStanding, ...]

    def to_dict(self) -> dict[str, object]:
        return {"race_index": self.race_index, "standings": [standing.to_dict() for standing in self.standings]}


@dataclass(frozen=True, slots=True)
class HeatRaceSnapshot:
    """Read-only telemetry at time zero and after each physics/marshal step."""

    race_index: int
    elapsed_seconds: float
    samples: tuple[TimingSample, ...]
    finished_car_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class HeatResult:
    """Individual standings ranked by distance or summed lap-race placements."""

    entrant_names: tuple[str, ...]
    round_seconds: float
    races: tuple[HeatRaceResult, ...]
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60
    round_laps: int | None = None
    finish_timeout_seconds: float = DEFAULT_FINISH_TIMEOUT_SECONDS

    def __post_init__(self) -> None:
        validate_round_laps(self.round_laps)
        validate_finish_timeout_seconds(self.finish_timeout_seconds)
        entrant_count = len(self.entrant_names)
        if entrant_count not in HEAT_ENTRANT_COUNTS:
            raise ValueError("a heat requires two to twenty entrants")
        expected = set(range(entrant_count))
        for race in self.races:
            if (
                len(race.standings) != entrant_count
                or {standing.entrant_index for standing in race.standings} != expected
            ):
                raise ValueError("each heat result must contain every entrant exactly once")

    @property
    def race_count(self) -> int:
        return len(self.races)

    @property
    def standings(self) -> tuple[HeatStanding, ...]:
        """Sum individual distance/laps/recoveries and average final damage."""
        totals: list[HeatStanding] = []
        for entrant_index, name in enumerate(self.entrant_names):
            results = [
                standing
                for race in self.races
                for standing in race.standings
                if standing.entrant_index == entrant_index
            ]
            totals.append(
                HeatStanding(
                    entrant_index=entrant_index,
                    name=name,
                    distance_m=sum(standing.distance_m for standing in results),
                    lap_count=sum(standing.lap_count for standing in results),
                    damage=sum(standing.damage for standing in results) / len(results) if results else 0.0,
                    marshal_count=sum(standing.marshal_count for standing in results),
                    finish_position=results[0].finish_position if len(results) == 1 else None,
                    finish_time_seconds=results[0].finish_time_seconds if len(results) == 1 else None,
                    dnf=bool(results) and all(standing.dnf for standing in results),
                    placement_total=sum(
                        standing.finish_position
                        if standing.finish_position is not None
                        else len(self.entrant_names) + 1
                        for standing in results
                    )
                    if self.round_laps is not None
                    else None,
                    dnf_count=sum(standing.dnf for standing in results),
                )
            )
        return _sorted_standings(totals, lap_race=self.round_laps is not None)

    @property
    def winner_index(self) -> int | None:
        """Return the leading entrant, or None for a tie or an all-DNF lap heat."""
        standings = self.standings
        if self.round_laps is not None:
            if not self.races or standings[0].dnf or standings[0].placement_total == standings[1].placement_total:
                return None
            return standings[0].entrant_index
        if not self.races or _tied_distance(standings[0].distance_m, standings[1].distance_m):
            return None
        return standings[0].entrant_index

    def to_dict(self) -> dict[str, object]:
        heat_rules = {
            key: value for key, value in self.rules.to_dict().items() if key not in {"scoring", "win_margin_m"}
        }
        return {
            "schema_version": HEAT_RESULT_SCHEMA_VERSION,
            "mode": "heat",
            "entrant_names": list(self.entrant_names),
            "round_seconds": self.round_seconds if self.round_laps is None else None,
            "round_laps": self.round_laps,
            "finish_timeout_seconds": self.finish_timeout_seconds if self.round_laps is not None else None,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "random_seed": self.random_seed,
            "track_id": self.track_id,
            "track_seed": self.track_seed,
            "scoring": "finish-order" if self.round_laps is not None else "total-distance",
            "rules": heat_rules,
            "summary": {
                "race_count": self.race_count,
                "winner_index": self.winner_index,
                "standings": [standing.to_dict() for standing in self.standings],
            },
            "races": [race.to_dict() for race in self.races],
        }


def validate_heat_entrants(entrants: tuple[HeatEntrant, ...]) -> None:
    """Validate the two to twenty independent car slots used by a heat."""
    if len(entrants) not in HEAT_ENTRANT_COUNTS:
        raise ValueError("a heat requires two to twenty entrants")
    for index, entrant in enumerate(entrants, start=1):
        if not entrant.name.strip():
            raise ValueError(f"heat entrant {index} needs a nonempty name")
        if not callable(entrant.controller):
            raise ValueError(f"heat entrant {index} needs a callable controller")


def heat_race_entries(
    *, entrant_count: int, race_index: int, random_seed: int = DEFAULT_RACE_RANDOM_SEED
) -> tuple[HeatRaceEntry, ...]:
    """Keep entrants in input order for every heat, independent of the spawn seed."""
    if entrant_count not in HEAT_ENTRANT_COUNTS:
        raise ValueError("a heat requires two to twenty entrants")
    if race_index < 1:
        raise ValueError("race_index must be at least one")
    return tuple(HeatRaceEntry(entrant_index=index) for index in range(entrant_count))


def heat_race_result_from_runtimes(
    *,
    entrants: tuple[HeatEntrant, ...],
    entries: tuple[HeatRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    race_index: int,
    lap_race: LapRace | None = None,
) -> HeatRaceResult:
    """Collect each entrant's score without combining unrelated cars into teams."""
    validate_heat_entrants(entrants)
    if len(entries) != len(runtimes) or len(entries) != len(entrants):
        raise ValueError("entrants, entries, and runtimes must have the same length")
    if {entry.entrant_index for entry in entries} != set(range(len(entrants))):
        raise ValueError("heat entries must contain every entrant exactly once")
    standings = [
        HeatStanding(
            entrant_index=entry.entrant_index,
            name=entrants[entry.entrant_index].name,
            distance_m=race_scored_distance_m(runtime),
            lap_count=runtime.tracker.lap_count if lap_race is None else lap_race.completed_laps(entry.car_id),
            damage=robot_score_damage(runtime.robot),
            marshal_count=runtime.marshal_count,
            finish_position=None if lap_race is None else lap_race.finish_position(entry.car_id),
            finish_time_seconds=None if lap_race is None else lap_race.finish_times.get(entry.car_id),
            dnf=lap_race is not None and entry.car_id not in lap_race.finish_times,
            dnf_count=int(lap_race is not None and entry.car_id not in lap_race.finish_times),
        )
        for entry, runtime in zip(entries, runtimes, strict=True)
    ]
    return HeatRaceResult(race_index=race_index, standings=_sorted_standings(standings, lap_race=lap_race is not None))


def format_heat_result(result: HeatResult) -> str:
    """Format individual totals, with damage reported independently of scoring."""
    if result.round_laps is not None:
        return "\n".join(
            [
                f"Heat: {len(result.entrant_names)} entrants | {result.race_count} races x {result.round_laps} laps",
                f"Seed: {result.random_seed} | Track: {result.track_id}"
                + (f" (seed {result.track_seed})" if result.track_seed is not None else ""),
                f"Scoring: finish order; DNF after elimination or {result.finish_timeout_seconds:g} seconds after P1.",
                "Across races: lowest placement total wins; each DNF counts as field size + 1.",
                *_lap_result_lines(result, shorten_names=False),
            ]
        )
    name_width = max(len(name) for name in result.entrant_names)
    rows = [
        f"Heat: {len(result.entrant_names)} entrants | {result.race_count} races x {result.round_seconds:g} s",
        f"Seed: {result.random_seed} | Track: {result.track_id}"
        + (f" (seed {result.track_seed})" if result.track_seed is not None else ""),
        "Scoring: total distance after marshal penalties; damage is averaged across races.",
        f"Rank  {'Entrant':<{name_width}}  Distance (m)  Laps  Damage  Marshals",
    ]
    rows.extend(
        f"{place:>4}  {standing.name:<{name_width}}  {standing.distance_m:>12.1f}  "
        f"{standing.lap_count:>4}  {standing.damage * 100.0:>5.1f}%  {standing.marshal_count:>8}"
        for place, standing in _placed_standings(result.standings)
    )
    return "\n".join(rows)


def format_heat_result_banner(result: HeatResult) -> str:
    """Show final rankings and summed damage percentages across all races."""
    if result.round_laps is not None:
        return "\n".join(["HEAT RESULTS", *_lap_result_lines(result, shorten_names=True)])
    return "\n".join(
        ["HEAT RESULTS"]
        + [
            f"{place}. {_banner_name(standing.name)}  {standing.distance_m:.1f} m"
            f"  |  {standing.damage * result.race_count * 100.0:.1f}% total damage"
            for place, standing in _placed_standings(result.standings)
        ]
    )


def run_headless_heat(
    *,
    entrants: tuple[HeatEntrant, ...],
    race_count: int = 1,
    round_seconds: float = 30.0,
    round_laps: int | None = None,
    finish_timeout_seconds: float = DEFAULT_FINISH_TIMEOUT_SECONDS,
    random_seed: int = DEFAULT_RACE_RANDOM_SEED,
    track_id: str = TRACK_ID_MUGELLO_SHORT,
    track_seed: int | None = None,
    rules: HeadToHeadRaceRules | None = None,
    fixed_delta_seconds: float = 1 / 60,
    observer: Callable[[HeatRaceSnapshot], None] | None = None,
) -> HeatResult:
    """Race two to twenty independent controllers through the shared physics loop."""
    validate_heat_entrants(entrants)
    validate_round_laps(round_laps)
    validate_finish_timeout_seconds(finish_timeout_seconds)
    if race_count < 1:
        raise ValueError("race_count must be at least one")
    if round_laps is None and round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    if fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    race_rules = HeadToHeadRaceRules() if rules is None else rules
    resolved_track = resolve_track(track_id, track_seed)
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    try:
        races = tuple(
            _run_headless_heat_race(
                render=base.render,
                entrants=entrants,
                race_index=race_index,
                round_seconds=round_seconds,
                round_laps=round_laps,
                finish_timeout_seconds=finish_timeout_seconds,
                random_seed=random_seed,
                model=resolved_track.model,
                samples=resolved_track.samples,
                rules=race_rules,
                fixed_delta_seconds=fixed_delta_seconds,
                observer=observer,
            )
            for race_index in range(1, race_count + 1)
        )
        return HeatResult(
            entrant_names=tuple(entrant.name for entrant in entrants),
            round_seconds=round_seconds,
            round_laps=round_laps,
            finish_timeout_seconds=finish_timeout_seconds,
            races=races,
            random_seed=random_seed,
            track_id=resolved_track.track_id,
            track_seed=resolved_track.seed,
            rules=race_rules,
            fixed_delta_seconds=fixed_delta_seconds,
        )
    finally:
        base.destroy()


def _run_headless_heat_race(
    *,
    render: Any,
    entrants: tuple[HeatEntrant, ...],
    race_index: int,
    round_seconds: float,
    round_laps: int | None,
    finish_timeout_seconds: float,
    random_seed: int,
    model: TrackProgressModel,
    samples: tuple[TrackPoint, ...],
    rules: HeadToHeadRaceRules,
    fixed_delta_seconds: float,
    observer: Callable[[HeatRaceSnapshot], None] | None = None,
) -> HeatRaceResult:
    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])
    root = render.attachNewNode(f"headless-heat-{race_index}")
    try:
        add_racing_scene_collisions(physics_world=physics_world, render=root, samples=samples)
        entries = heat_race_entries(entrant_count=len(entrants), race_index=race_index, random_seed=random_seed)
        spawn_poses = race_spawn_poses(
            len(entries), model=model, random_seed=random_seed, race_index=race_index, shuffle_grid=False
        )
        controllers = tuple(controller_for_copy(entrants[entry.entrant_index].controller) for entry in entries)
        runtimes: list[RaceCarRuntime] = []
        for entry, pose in zip(entries, spawn_poses, strict=True):
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name=f"headless-heat-{race_index}-{entry.entrant_index}",
                position=pose.position,
                heading_degrees=pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            physics_scene.vehicles.append(robot)
            runtimes.append(
                RaceCarRuntime(robot=robot, tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=pose))
            )
        recovery_config = (
            RaceRecoveryConfig(
                stuck_seconds=rules.marshal_stuck_seconds,
                distance_penalty_m=rules.marshal_penalty_m,
                cooldown_seconds=rules.marshal_cooldown_seconds,
            )
            if rules.marshal_enabled
            else None
        )
        lap_race = (
            None
            if round_laps is None
            else LapRace(
                round_laps=round_laps,
                track_length_m=model.total_length_m,
                finish_timeout_seconds=finish_timeout_seconds,
            )
        )
        start_finish = seeded_race_start_finish_pose(model=model, random_seed=random_seed, race_index=race_index)
        previous_marshals: dict[str, int] = {}

        def race_complete(elapsed_seconds: float) -> bool:
            timing_samples = tuple(
                TimingSample(
                    car_id=entry.car_id,
                    distance_m=race_track_position_m(runtime, start_finish_progress_m=start_finish.progress_distance_m),
                    eliminated=runtime.robot.eliminated,
                    discontinuity=previous_marshals.get(entry.car_id, 0) != runtime.marshal_count,
                )
                for entry, runtime in zip(entries, runtimes, strict=True)
            )
            if lap_race is not None:
                lap_race.update(elapsed_seconds=elapsed_seconds, samples=timing_samples)
            if observer is not None:
                observer(
                    HeatRaceSnapshot(
                        race_index=race_index,
                        elapsed_seconds=elapsed_seconds,
                        samples=timing_samples,
                        finished_car_ids=frozenset() if lap_race is None else frozenset(lap_race.finish_times),
                    )
                )
            previous_marshals.update(
                (entry.car_id, runtime.marshal_count) for entry, runtime in zip(entries, runtimes, strict=True)
            )
            return elapsed_seconds >= round_seconds if lap_race is None else lap_race.complete

        _run_headless_student_runtime_for_duration(
            model=model,
            physics_world=physics_world,
            physics_scene=physics_scene,
            entries=entries,
            controllers=controllers,
            runtimes=tuple(runtimes),
            duration_seconds=round_seconds,
            fixed_delta_seconds=fixed_delta_seconds,
            recovery_config=recovery_config,
            damage_enabled=rules.damage_enabled,
            race_complete=race_complete if lap_race is not None or observer is not None else None,
        )
        return heat_race_result_from_runtimes(
            entrants=entrants, entries=entries, runtimes=tuple(runtimes), race_index=race_index, lap_race=lap_race
        )
    finally:
        root.removeNode()


def _sorted_standings(standings: list[HeatStanding], *, lap_race: bool = False) -> tuple[HeatStanding, ...]:
    if lap_race:
        return tuple(
            sorted(
                standings,
                key=lambda standing: (
                    standing.placement_total
                    if standing.placement_total is not None
                    else standing.finish_position
                    if standing.finish_position is not None
                    else len(standings) + 1,
                    standing.entrant_index,
                ),
            )
        )
    return tuple(sorted(standings, key=lambda standing: (-standing.distance_m, standing.entrant_index)))


def _lap_result_lines(result: HeatResult, *, shorten_names: bool) -> list[str]:
    lines: list[str] = []
    previous_total: int | None = None
    place = 0
    for index, standing in enumerate(result.standings, start=1):
        name = _banner_name(standing.name) if shorten_names else standing.name
        if standing.placement_total != previous_total:
            place = index
        previous_total = standing.placement_total
        if result.race_count == 1:
            label = "DNF" if standing.dnf else str(standing.finish_position)
            detail = "" if standing.finish_time_seconds is None else f"  {standing.finish_time_seconds:.3f}s"
        else:
            label = str(place)
            detail = f"  {standing.placement_total} place total | {standing.dnf_count} DNF"
        lines.append(f"{label}. {name}{detail}  |  {standing.damage * result.race_count * 100.0:.1f}% total damage")
    return lines


def _banner_name(name: str) -> str:
    collapsed = " ".join(name.split())
    return collapsed if len(collapsed) <= 24 else collapsed[:23].rstrip() + "…"


def _tied_distance(first: float, second: float) -> bool:
    return isclose(first, second, rel_tol=0.0, abs_tol=1e-6)


def _placed_standings(standings: tuple[HeatStanding, ...]) -> tuple[tuple[int, HeatStanding], ...]:
    placed: list[tuple[int, HeatStanding]] = []
    place = 0
    previous_distance: float | None = None
    for index, standing in enumerate(standings, start=1):
        if previous_distance is None or not _tied_distance(standing.distance_m, previous_distance):
            place = index
        placed.append((place, standing))
        previous_distance = standing.distance_m
    return tuple(placed)
