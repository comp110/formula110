"""Heats for four or eight controllers using shared racing physics and scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from math import isclose
from random import Random
from typing import Any, cast

from racing.graphics.colors import UNC_CAROLINA_BLUE, ColorRGBA
from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, PhysicsScene, create_physics_world, create_robot_vehicle
from racing.race.head_to_head import (
    _run_headless_student_runtime_for_duration,  # pyright: ignore[reportPrivateUsage]
    controller_for_copy,
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
    robot_score_damage,
)
from racing.student.api import RobotController
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint

HEAT_ENTRANT_COUNTS = (4, 8)
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
)


@dataclass(frozen=True, slots=True)
class HeatEntrant:
    """One independently controlled car and its display metadata."""

    name: str
    controller: RobotController
    team_color: ColorRGBA


@dataclass(frozen=True, slots=True)
class HeatRaceEntry:
    """Stable entrant identity assigned to one shuffled starting-grid slot."""

    entrant_index: int

    @property
    def role(self) -> str:
        return f"heat-{self.entrant_index}"

    @property
    def copy_index(self) -> int:
        return 0


@dataclass(frozen=True, slots=True)
class HeatStanding:
    """Individual distance, laps, damage, and recoveries for a race or suite."""

    entrant_index: int
    name: str
    distance_m: float
    lap_count: int
    damage: float
    marshal_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "entrant_index": self.entrant_index,
            "name": self.name,
            "distance_m": self.distance_m,
            "lap_count": self.lap_count,
            "damage": self.damage,
            "marshal_count": self.marshal_count,
        }


@dataclass(frozen=True, slots=True)
class HeatRaceResult:
    """Standings for one completed heat."""

    race_index: int
    standings: tuple[HeatStanding, ...]

    def to_dict(self) -> dict[str, object]:
        return {"race_index": self.race_index, "standings": [standing.to_dict() for standing in self.standings]}


@dataclass(frozen=True, slots=True)
class HeatResult:
    """Individual standings ranked by summed distance across all races."""

    entrant_names: tuple[str, ...]
    round_seconds: float
    races: tuple[HeatRaceResult, ...]
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60

    def __post_init__(self) -> None:
        entrant_count = len(self.entrant_names)
        if entrant_count not in HEAT_ENTRANT_COUNTS:
            raise ValueError("a heat requires exactly four or eight entrants")
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
                )
            )
        return _sorted_standings(totals)

    @property
    def winner_index(self) -> int | None:
        """Return the leading entrant, or None when the greatest totals tie."""
        standings = self.standings
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
            "round_seconds": self.round_seconds,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "random_seed": self.random_seed,
            "track_id": self.track_id,
            "track_seed": self.track_seed,
            "scoring": "total-distance",
            "rules": heat_rules,
            "summary": {
                "race_count": self.race_count,
                "winner_index": self.winner_index,
                "standings": [standing.to_dict() for standing in self.standings],
            },
            "races": [race.to_dict() for race in self.races],
        }


def validate_heat_entrants(entrants: tuple[HeatEntrant, ...]) -> None:
    """Validate the four or eight independent car slots used by a heat."""
    if len(entrants) not in HEAT_ENTRANT_COUNTS:
        raise ValueError("a heat requires exactly four or eight entrants")
    for index, entrant in enumerate(entrants, start=1):
        if not entrant.name.strip():
            raise ValueError(f"heat entrant {index} needs a nonempty name")
        if not callable(entrant.controller):
            raise ValueError(f"heat entrant {index} needs a callable controller")


def heat_race_entries(
    *, entrant_count: int, race_index: int, random_seed: int = DEFAULT_RACE_RANDOM_SEED
) -> tuple[HeatRaceEntry, ...]:
    """Shuffle stable entrant identities with the head-to-head grid seed formula."""
    if entrant_count not in HEAT_ENTRANT_COUNTS:
        raise ValueError("a heat requires exactly four or eight entrants")
    if race_index < 1:
        raise ValueError("race_index must be at least one")
    entries = [HeatRaceEntry(entrant_index=index) for index in range(entrant_count)]
    Random(random_seed + race_index * 131_071 + 9_173).shuffle(entries)
    return tuple(entries)


def heat_race_result_from_runtimes(
    *,
    entrants: tuple[HeatEntrant, ...],
    entries: tuple[HeatRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    race_index: int,
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
            lap_count=runtime.tracker.lap_count,
            damage=robot_score_damage(runtime.robot),
            marshal_count=runtime.marshal_count,
        )
        for entry, runtime in zip(entries, runtimes, strict=True)
    ]
    return HeatRaceResult(race_index=race_index, standings=_sorted_standings(standings))


def format_heat_result(result: HeatResult) -> str:
    """Format individual totals, with damage reported independently of scoring."""
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
    """Format a compact final ranking for the race viewer."""
    return "\n".join(
        ["HEAT RESULTS"]
        + [
            f"{place}. {_banner_name(standing.name)}  {standing.distance_m:.1f} m"
            for place, standing in _placed_standings(result.standings)
        ]
    )


def run_headless_heat(
    *,
    entrants: tuple[HeatEntrant, ...],
    race_count: int = 1,
    round_seconds: float = 30.0,
    random_seed: int = DEFAULT_RACE_RANDOM_SEED,
    track_id: str = TRACK_ID_MUGELLO_SHORT,
    track_seed: int | None = None,
    rules: HeadToHeadRaceRules | None = None,
    fixed_delta_seconds: float = 1 / 60,
) -> HeatResult:
    """Race four or eight independent controllers through the shared physics loop."""
    validate_heat_entrants(entrants)
    if race_count < 1:
        raise ValueError("race_count must be at least one")
    if round_seconds <= 0.0:
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
                random_seed=random_seed,
                model=resolved_track.model,
                samples=resolved_track.samples,
                rules=race_rules,
                fixed_delta_seconds=fixed_delta_seconds,
            )
            for race_index in range(1, race_count + 1)
        )
        return HeatResult(
            entrant_names=tuple(entrant.name for entrant in entrants),
            round_seconds=round_seconds,
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
    random_seed: int,
    model: TrackProgressModel,
    samples: tuple[TrackPoint, ...],
    rules: HeadToHeadRaceRules,
    fixed_delta_seconds: float,
) -> HeatRaceResult:
    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])
    root = render.attachNewNode(f"headless-heat-{race_index}")
    try:
        add_racing_scene_collisions(physics_world=physics_world, render=root, samples=samples)
        entries = heat_race_entries(entrant_count=len(entrants), race_index=race_index, random_seed=random_seed)
        spawn_poses = race_spawn_poses(len(entries), model=model, random_seed=random_seed, race_index=race_index)
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
        )
        return heat_race_result_from_runtimes(
            entrants=entrants, entries=entries, runtimes=tuple(runtimes), race_index=race_index
        )
    finally:
        root.removeNode()


def _sorted_standings(standings: list[HeatStanding]) -> tuple[HeatStanding, ...]:
    return tuple(sorted(standings, key=lambda standing: (-standing.distance_m, standing.entrant_index)))


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
