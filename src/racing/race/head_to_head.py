"""Head-to-head races between student controllers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from importlib import import_module
from random import Random
from typing import Any, Literal, TypeVar, cast

from racing.graphics.panda_config import configure_headless_panda
from racing.graphics.track_rendering import add_racing_scene_collisions
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.progress import (
    TrackProgressModel,
    TrackProjection,
    project_track_position,
    resolve_track,
)
from racing.race.rules import (
    HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M,
    HeadToHeadRaceRules,
    HeadToHeadScoring,
)
from racing.race.runtime import (
    DEFAULT_RACE_RANDOM_SEED,
    RaceCarRuntime,
    RaceRecoveryConfig,
    lap_progress_tracker_for_spawn_pose,
    maybe_marshal_race_runtimes,
    race_contact_states,
    race_scored_distance_m,
    race_spawn_poses,
    robot_score_damage,
    robot_track_point,
    update_race_runtime_after_step,
)
from racing.race.sensors import build_robot_sensors
from racing.student.api import RobotController, RobotSensors
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint

HEAD_TO_HEAD_DEFAULT_RACE_COUNT = 7
HEAD_TO_HEAD_COPIES_PER_SIDE = 1
HEAD_TO_HEAD_DEFAULT_ROUND_SECONDS = 30.0
HEAD_TO_HEAD_RESULT_SCHEMA_VERSION = 1

HeadToHeadRole = Literal["challenger", "incumbent"]
HeadToHeadOutcome = Literal["challenger", "incumbent", "tie"]
_RaceEntryT = TypeVar("_RaceEntryT")


@dataclass(frozen=True, slots=True)
class HeadToHeadTeamRaceStats:
    """Per-team stats for a head-to-head race."""

    distances_m: tuple[float, ...]
    lap_counts: tuple[int, ...]
    wall_contact_seconds: tuple[float, ...]
    car_contact_seconds: tuple[float, ...]
    damages: tuple[float, ...] = ()
    raw_distances_m: tuple[float, ...] = ()
    max_speeds_mps: tuple[float, ...] = ()
    best_lap_times_seconds: tuple[float | None, ...] = ()
    penalized_distances_m: tuple[float, ...] = ()
    low_progress_seconds: tuple[float, ...] = ()
    off_track_seconds: tuple[float, ...] = ()
    marshal_counts: tuple[int, ...] = ()
    marshal_penalties_m: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.distances_m) == 0:
            raise ValueError("head-to-head team stats require at least one car")
        if len(self.damages) == 0:
            object.__setattr__(self, "damages", tuple(0.0 for _ in self.distances_m))
        if len(self.raw_distances_m) == 0:
            object.__setattr__(self, "raw_distances_m", self.distances_m)
        if len(self.max_speeds_mps) == 0:
            object.__setattr__(self, "max_speeds_mps", tuple(0.0 for _ in self.distances_m))
        if len(self.best_lap_times_seconds) == 0:
            object.__setattr__(self, "best_lap_times_seconds", tuple(None for _ in self.distances_m))
        if len(self.penalized_distances_m) == 0:
            object.__setattr__(self, "penalized_distances_m", tuple(0.0 for _ in self.distances_m))
        if len(self.low_progress_seconds) == 0:
            object.__setattr__(self, "low_progress_seconds", tuple(0.0 for _ in self.distances_m))
        if len(self.off_track_seconds) == 0:
            object.__setattr__(self, "off_track_seconds", tuple(0.0 for _ in self.distances_m))
        if len(self.marshal_counts) == 0:
            object.__setattr__(self, "marshal_counts", tuple(0 for _ in self.distances_m))
        if len(self.marshal_penalties_m) == 0:
            object.__setattr__(self, "marshal_penalties_m", tuple(0.0 for _ in self.distances_m))
        if not (
            len(self.distances_m)
            == len(self.lap_counts)
            == len(self.wall_contact_seconds)
            == len(self.car_contact_seconds)
            == len(self.damages)
            == len(self.raw_distances_m)
            == len(self.max_speeds_mps)
            == len(self.best_lap_times_seconds)
            == len(self.penalized_distances_m)
            == len(self.low_progress_seconds)
            == len(self.off_track_seconds)
            == len(self.marshal_counts)
            == len(self.marshal_penalties_m)
        ):
            raise ValueError("head-to-head team stat fields must have the same length")

    @property
    def best_distance_m(self) -> float:
        """Return the farthest distance reached by any team car."""
        return max(self.distances_m)

    @property
    def team_sum_distance_m(self) -> float:
        """Return the summed scored distance reached by all team cars."""
        return sum(self.distances_m)

    @property
    def average_distance_m(self) -> float:
        """Return the average scored distance reached by the team cars."""
        return sum(self.distances_m) / len(self.distances_m)

    @property
    def best_raw_distance_m(self) -> float:
        """Return the farthest raw distance reached by any team car."""
        return max(self.raw_distances_m)

    @property
    def team_sum_raw_distance_m(self) -> float:
        """Return summed raw distance reached by all team cars."""
        return sum(self.raw_distances_m)

    @property
    def max_speed_mps(self) -> float:
        """Return the fastest speed reached by any team car."""
        return max(self.max_speeds_mps)

    @property
    def best_lap_time_seconds(self) -> float | None:
        """Return the fastest completed lap time, when any team car completed a lap."""
        lap_times = tuple(lap_time for lap_time in self.best_lap_times_seconds if lap_time is not None)
        if len(lap_times) == 0:
            return None
        return min(lap_times)

    @property
    def total_lap_count(self) -> int:
        """Return total completed laps across team cars."""
        return sum(self.lap_counts)

    @property
    def average_damage(self) -> float:
        """Return average final damage across team cars."""
        return sum(self.damages) / len(self.damages)

    @property
    def elimination_count(self) -> int:
        """Return number of team cars fully eliminated by damage."""
        return sum(1 for damage in self.damages if damage >= 1.0)

    @property
    def total_penalized_distance_m(self) -> float:
        """Return summed forward progress excluded while in contact."""
        return sum(self.penalized_distances_m)

    @property
    def total_wall_contact_seconds(self) -> float:
        """Add up how long this team touched walls."""
        return sum(self.wall_contact_seconds)

    @property
    def total_car_contact_seconds(self) -> float:
        """Add up how long this team touched other cars."""
        return sum(self.car_contact_seconds)

    @property
    def total_contact_seconds(self) -> float:
        """Return summed wall and car contact seconds."""
        return self.total_wall_contact_seconds + self.total_car_contact_seconds

    @property
    def total_low_progress_seconds(self) -> float:
        """Return summed stuck or low-progress seconds."""
        return sum(self.low_progress_seconds)

    @property
    def total_off_track_seconds(self) -> float:
        """Return summed time outside the drivable track ribbon."""
        return sum(self.off_track_seconds)

    @property
    def total_marshal_count(self) -> int:
        """Return total deterministic marshal recoveries."""
        return sum(self.marshal_counts)

    @property
    def total_marshal_penalty_m(self) -> float:
        """Return total distance penalty applied for marshal recoveries."""
        return sum(self.marshal_penalties_m)

    def to_dict(self) -> dict[str, object]:
        """Return stable, JSON-compatible per-car values and team summaries."""
        return {
            "distances_m": list(self.distances_m),
            "raw_distances_m": list(self.raw_distances_m),
            "lap_counts": list(self.lap_counts),
            "wall_contact_seconds": list(self.wall_contact_seconds),
            "car_contact_seconds": list(self.car_contact_seconds),
            "damages": list(self.damages),
            "max_speeds_mps": list(self.max_speeds_mps),
            "best_lap_times_seconds": list(self.best_lap_times_seconds),
            "penalized_distances_m": list(self.penalized_distances_m),
            "low_progress_seconds": list(self.low_progress_seconds),
            "off_track_seconds": list(self.off_track_seconds),
            "marshal_counts": list(self.marshal_counts),
            "marshal_penalties_m": list(self.marshal_penalties_m),
            "summary": {
                "best_distance_m": self.best_distance_m,
                "team_sum_distance_m": self.team_sum_distance_m,
                "team_sum_raw_distance_m": self.team_sum_raw_distance_m,
                "total_lap_count": self.total_lap_count,
                "best_lap_time_seconds": self.best_lap_time_seconds,
                "max_speed_mps": self.max_speed_mps,
                "average_damage": self.average_damage,
                "elimination_count": self.elimination_count,
                "total_wall_contact_seconds": self.total_wall_contact_seconds,
                "total_car_contact_seconds": self.total_car_contact_seconds,
                "total_low_progress_seconds": self.total_low_progress_seconds,
                "total_off_track_seconds": self.total_off_track_seconds,
                "total_marshal_count": self.total_marshal_count,
                "total_marshal_penalty_m": self.total_marshal_penalty_m,
            },
        }


@dataclass(frozen=True, slots=True)
class HeadToHeadRaceResult:
    """Result for one head-to-head race."""

    race_index: int
    winner: HeadToHeadOutcome
    challenger: HeadToHeadTeamRaceStats
    incumbent: HeadToHeadTeamRaceStats
    scoring: HeadToHeadScoring = "team-sum"

    @property
    def margin_m(self) -> float:
        """Return challenger scored distance minus incumbent scored distance."""
        return head_to_head_race_margin(challenger=self.challenger, incumbent=self.incumbent, scoring=self.scoring)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible record for this race."""
        return {
            "race_index": self.race_index,
            "winner": self.winner,
            "margin_m": self.margin_m,
            "scoring": self.scoring,
            "challenger": self.challenger.to_dict(),
            "incumbent": self.incumbent.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class HeadToHeadResult:
    """Aggregate result for a head-to-head race suite."""

    challenger_name: str
    incumbent_name: str
    round_seconds: float
    win_margin_m: float
    races: tuple[HeadToHeadRaceResult, ...]
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60

    @property
    def race_count(self) -> int:
        """Return number of completed races."""
        return len(self.races)

    @property
    def challenger_wins(self) -> int:
        """Return number of races won by the challenger."""
        return sum(1 for race in self.races if race.winner == "challenger")

    @property
    def incumbent_wins(self) -> int:
        """Return number of races won by the incumbent."""
        return sum(1 for race in self.races if race.winner == "incumbent")

    @property
    def ties(self) -> int:
        """Return number of tied races."""
        return sum(1 for race in self.races if race.winner == "tie")

    @property
    def majority_wins_needed(self) -> int:
        """Return wins needed to win more than half of the race suite."""
        return self.race_count // 2 + 1

    @property
    def challenger_met_goal(self) -> bool:
        """Return whether the challenger won more than half of the races."""
        return self.challenger_wins >= self.majority_wins_needed

    @property
    def winner(self) -> HeadToHeadOutcome:
        """Return the suite winner by race wins."""
        if self.challenger_wins > self.incumbent_wins:
            return "challenger"
        if self.incumbent_wins > self.challenger_wins:
            return "incumbent"
        return "tie"

    @property
    def aggregate_margin_m(self) -> float:
        """Return summed challenger-vs-incumbent margin over all races."""
        return sum(race.margin_m for race in self.races)

    def to_dict(self) -> dict[str, object]:
        """Return a versioned, JSON-compatible experiment result."""
        return {
            "schema_version": HEAD_TO_HEAD_RESULT_SCHEMA_VERSION,
            "challenger_name": self.challenger_name,
            "incumbent_name": self.incumbent_name,
            "round_seconds": self.round_seconds,
            "fixed_delta_seconds": self.fixed_delta_seconds,
            "random_seed": self.random_seed,
            "track_id": self.track_id,
            "track_seed": self.track_seed,
            "rules": self.rules.to_dict(),
            "summary": {
                "winner": self.winner,
                "race_count": self.race_count,
                "challenger_wins": self.challenger_wins,
                "incumbent_wins": self.incumbent_wins,
                "ties": self.ties,
                "aggregate_margin_m": self.aggregate_margin_m,
            },
            "races": [race.to_dict() for race in self.races],
        }


@dataclass(frozen=True, slots=True)
class HeadToHeadRaceEntry:
    """One car slot in a head-to-head race."""

    role: HeadToHeadRole
    copy_index: int


def head_to_head_race_margin(
    *,
    challenger: HeadToHeadTeamRaceStats,
    incumbent: HeadToHeadTeamRaceStats,
    scoring: HeadToHeadScoring,
) -> float:
    """Compare challenger and incumbent distance using the active scoring rule."""
    if scoring == "best-copy":
        return challenger.best_distance_m - incumbent.best_distance_m
    if scoring == "team-sum":
        return challenger.team_sum_distance_m - incumbent.team_sum_distance_m
    raise ValueError(f"unknown head-to-head scoring rule: {scoring}")


def classify_head_to_head_winner(
    *,
    margin_m: float,
    win_margin_m: float = HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M,
) -> HeadToHeadOutcome:
    """Classify a head-to-head race by challenger distance margin."""
    if win_margin_m < 0.0:
        raise ValueError("win_margin_m cannot be negative")
    if margin_m > win_margin_m:
        return "challenger"
    if margin_m < -win_margin_m:
        return "incumbent"
    return "tie"


def head_to_head_race_entries(
    *,
    race_index: int,
    random_seed: int,
    challenger_copies: int,
    incumbent_copies: int,
) -> tuple[HeadToHeadRaceEntry, ...]:
    """Create the car entries for one repeatable head-to-head race."""
    _validate_head_to_head_copy_counts(challenger_copies=challenger_copies, incumbent_copies=incumbent_copies)
    entries = [
        *(HeadToHeadRaceEntry(role="challenger", copy_index=copy_index) for copy_index in range(challenger_copies)),
        *(HeadToHeadRaceEntry(role="incumbent", copy_index=copy_index) for copy_index in range(incumbent_copies)),
    ]
    rng = Random(random_seed + race_index * 131_071 + 9_173)
    rng.shuffle(entries)
    return tuple(entries)


def run_headless_head_to_head(
    *,
    challenger_controller: RobotController,
    incumbent_controller: RobotController,
    challenger_name: str = "challenger",
    incumbent_name: str = "incumbent",
    race_count: int = HEAD_TO_HEAD_DEFAULT_RACE_COUNT,
    round_seconds: float = HEAD_TO_HEAD_DEFAULT_ROUND_SECONDS,
    random_seed: int = DEFAULT_RACE_RANDOM_SEED,
    track_id: str = TRACK_ID_MUGELLO_SHORT,
    track_seed: int | None = None,
    win_margin_m: float | None = None,
    rules: HeadToHeadRaceRules | None = None,
    copies_per_side: int = HEAD_TO_HEAD_COPIES_PER_SIDE,
    challenger_copies: int | None = None,
    incumbent_copies: int | None = None,
    fixed_delta_seconds: float = 1 / 60,
    sensor_sample_callback: Callable[[HeadToHeadRaceEntry, RobotSensors], None] | None = None,
) -> HeadToHeadResult:
    """Run deterministic headless races between two student controllers."""
    if race_count < 1:
        raise ValueError("race_count must be at least one")
    if round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    if fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    resolved_challenger_copies, resolved_incumbent_copies = _head_to_head_copy_counts(
        copies_per_side=copies_per_side,
        challenger_copies=challenger_copies,
        incumbent_copies=incumbent_copies,
    )
    race_rules = HeadToHeadRaceRules() if rules is None else rules
    if win_margin_m is not None:
        race_rules = replace(race_rules, win_margin_m=win_margin_m)
    resolved_track = resolve_track(track_id, track_seed)

    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    try:
        races = tuple(
            _run_headless_student_race(
                render=base.render,
                challenger_controller=challenger_controller,
                incumbent_controller=incumbent_controller,
                race_index=race_index,
                round_seconds=round_seconds,
                random_seed=random_seed,
                model=resolved_track.model,
                samples=resolved_track.samples,
                rules=race_rules,
                challenger_copies=resolved_challenger_copies,
                incumbent_copies=resolved_incumbent_copies,
                fixed_delta_seconds=fixed_delta_seconds,
                sensor_sample_callback=sensor_sample_callback,
            )
            for race_index in range(1, race_count + 1)
        )
        return HeadToHeadResult(
            challenger_name=challenger_name,
            incumbent_name=incumbent_name,
            round_seconds=round_seconds,
            win_margin_m=race_rules.win_margin_m,
            races=races,
            random_seed=random_seed,
            track_id=resolved_track.track_id,
            track_seed=resolved_track.seed,
            rules=race_rules,
            fixed_delta_seconds=fixed_delta_seconds,
        )
    finally:
        base.destroy()


def format_head_to_head_result(result: HeadToHeadResult) -> str:
    """Format race results for terminal output."""
    lines = [
        _head_to_head_winner_line(result),
        _head_to_head_record_line(result),
        _head_to_head_metadata_line(result),
        "",
        *_head_to_head_result_table_lines(result),
        "",
        "Per-race results:",
    ]
    for race in result.races:
        lines.append(_head_to_head_race_summary_line(result=result, race=race))
    return "\n".join(lines)


def format_head_to_head_result_banner(result: HeadToHeadResult) -> str:
    """Show the winner, distance, and summed damage across team cars and races."""
    winner_line = (
        "RESULT: TIE" if result.winner == "tie" else f"WINNER: {_head_to_head_result_role_name(result, result.winner)}"
    )
    lines = [winner_line]
    for role in ("challenger", "incumbent"):
        damage = sum(sum(_head_to_head_race_team_stats(race, role).damages) for race in result.races)
        lines.append(
            f"{_head_to_head_result_role_name(result, role)}: "
            f"{_format_distance_m(_aggregate_role_scored_distance_m(result, role))}"
            f"  |  {damage * 100.0:.1f}% total damage"
        )
    return "\n".join(lines)


def _head_to_head_winner_line(result: HeadToHeadResult) -> str:
    if result.winner == "tie":
        return "Winner: tie"
    return f"Winner: {_head_to_head_result_role_name(result, result.winner)}"


def _head_to_head_record_line(result: HeadToHeadResult) -> str:
    return (
        f"Record: {_head_to_head_result_role_name(result, 'challenger')} {result.challenger_wins}, "
        f"{_head_to_head_result_role_name(result, 'incumbent')} {result.incumbent_wins}, "
        f"ties {result.ties}"
    )


def _head_to_head_metadata_line(result: HeadToHeadResult) -> str:
    track_description = (
        result.track_id if result.track_seed is None else f"{result.track_id} (seed {result.track_seed})"
    )
    return (
        f"Races: {result.race_count} | Round: {result.round_seconds:.1f}s | "
        f"{_head_to_head_result_copy_summary(result)} | Scoring: {result.rules.scoring} | "
        f"Seed: {result.random_seed} | Track: {track_description} | {_head_to_head_marshal_summary(result.rules)}"
    )


def _head_to_head_marshal_summary(rules: HeadToHeadRaceRules) -> str:
    if not rules.marshal_enabled:
        return "Marshal: off"
    return (
        f"Marshal: on ({rules.marshal_stuck_seconds:.1f}s stuck, "
        f"{rules.marshal_penalty_m:.1f}m penalty, {rules.marshal_cooldown_seconds:.1f}s cooldown)"
    )


def _head_to_head_result_table_lines(result: HeadToHeadResult) -> list[str]:
    first_role, second_role = _head_to_head_result_table_roles(result)
    headers = (
        _head_to_head_result_role_name(result, first_role),
        _head_to_head_result_role_name(result, second_role),
    )
    rows: list[tuple[str, tuple[str, str]]] = [
        (
            "Race wins",
            (
                str(_head_to_head_role_wins(result, first_role)),
                str(_head_to_head_role_wins(result, second_role)),
            ),
        ),
        (
            "Laps completed",
            (
                str(_aggregate_role_lap_count(result, first_role)),
                str(_aggregate_role_lap_count(result, second_role)),
            ),
        ),
        (
            "Scored distance",
            (
                _format_distance_m(_aggregate_role_scored_distance_m(result, first_role)),
                _format_distance_m(_aggregate_role_scored_distance_m(result, second_role)),
            ),
        ),
        (
            "Raw distance",
            (
                _format_distance_m(_aggregate_role_raw_distance_m(result, first_role)),
                _format_distance_m(_aggregate_role_raw_distance_m(result, second_role)),
            ),
        ),
        (
            "Best lap",
            (
                _format_optional_seconds(_aggregate_role_best_lap_seconds(result, first_role)),
                _format_optional_seconds(_aggregate_role_best_lap_seconds(result, second_role)),
            ),
        ),
        (
            "Max speed",
            (
                _format_speed_mps(_aggregate_role_max_speed_mps(result, first_role)),
                _format_speed_mps(_aggregate_role_max_speed_mps(result, second_role)),
            ),
        ),
        (
            "Average damage",
            (
                _format_percent(_aggregate_role_average_damage(result, first_role)),
                _format_percent(_aggregate_role_average_damage(result, second_role)),
            ),
        ),
        (
            "Eliminations",
            (
                str(_aggregate_role_elimination_count(result, first_role)),
                str(_aggregate_role_elimination_count(result, second_role)),
            ),
        ),
        (
            "Marshal resets",
            (
                str(_aggregate_role_marshal_count(result, first_role)),
                str(_aggregate_role_marshal_count(result, second_role)),
            ),
        ),
        (
            "Marshal penalty",
            (
                _format_distance_m(_aggregate_role_marshal_penalty_m(result, first_role)),
                _format_distance_m(_aggregate_role_marshal_penalty_m(result, second_role)),
            ),
        ),
        (
            "Wall contact time",
            (
                _format_seconds(_aggregate_role_wall_contact_seconds(result, first_role)),
                _format_seconds(_aggregate_role_wall_contact_seconds(result, second_role)),
            ),
        ),
        (
            "Car contact time",
            (
                _format_seconds(_aggregate_role_car_contact_seconds(result, first_role)),
                _format_seconds(_aggregate_role_car_contact_seconds(result, second_role)),
            ),
        ),
        (
            "Low-progress time",
            (
                _format_seconds(_aggregate_role_low_progress_seconds(result, first_role)),
                _format_seconds(_aggregate_role_low_progress_seconds(result, second_role)),
            ),
        ),
        (
            "Off-track time",
            (
                _format_seconds(_aggregate_role_off_track_seconds(result, first_role)),
                _format_seconds(_aggregate_role_off_track_seconds(result, second_role)),
            ),
        ),
    ]
    metric_width = max(len("Metric"), *(len(metric) for metric, _ in rows))
    column_widths = (
        max(len(headers[0]), *(len(values[0]) for _, values in rows)),
        max(len(headers[1]), *(len(values[1]) for _, values in rows)),
    )
    lines = [
        _format_head_to_head_table_row("Metric", headers, metric_width=metric_width, column_widths=column_widths),
        _format_head_to_head_table_separator(metric_width=metric_width, column_widths=column_widths),
    ]
    lines.extend(
        _format_head_to_head_table_row(metric, values, metric_width=metric_width, column_widths=column_widths)
        for metric, values in rows
    )
    return lines


def _head_to_head_result_table_roles(result: HeadToHeadResult) -> tuple[HeadToHeadRole, HeadToHeadRole]:
    if result.winner == "incumbent":
        return ("incumbent", "challenger")
    return ("challenger", "incumbent")


def _format_head_to_head_table_row(
    metric: str,
    values: tuple[str, str],
    *,
    metric_width: int,
    column_widths: tuple[int, int],
) -> str:
    return f"{metric:<{metric_width}}  {values[0]:>{column_widths[0]}}  {values[1]:>{column_widths[1]}}"


def _format_head_to_head_table_separator(*, metric_width: int, column_widths: tuple[int, int]) -> str:
    return f"{'-' * metric_width}  {'-' * column_widths[0]}  {'-' * column_widths[1]}"


def _head_to_head_race_summary_line(*, result: HeadToHeadResult, race: HeadToHeadRaceResult) -> str:
    race_winner = "tie" if race.winner == "tie" else _head_to_head_result_role_name(result, race.winner)
    challenger_scored_m = _head_to_head_team_scored_distance_m(race.challenger, race.scoring)
    incumbent_scored_m = _head_to_head_team_scored_distance_m(race.incumbent, race.scoring)
    return (
        f"race {race.race_index:02d}: winner {race_winner}, margin {race.margin_m:+.2f}m "
        f"({_head_to_head_result_role_name(result, 'challenger')} {challenger_scored_m:.1f}m, "
        f"{_head_to_head_result_role_name(result, 'incumbent')} {incumbent_scored_m:.1f}m; "
        f"car contact C {race.challenger.total_car_contact_seconds:.2f}s/"
        f"I {race.incumbent.total_car_contact_seconds:.2f}s; "
        f"marshal C {race.challenger.total_marshal_count}:{race.challenger.total_marshal_penalty_m:.1f}m/"
        f"I {race.incumbent.total_marshal_count}:{race.incumbent.total_marshal_penalty_m:.1f}m)"
    )


def _head_to_head_result_role_name(result: HeadToHeadResult, role: HeadToHeadRole) -> str:
    if role == "challenger":
        return result.challenger_name
    return result.incumbent_name


def _head_to_head_role_wins(result: HeadToHeadResult, role: HeadToHeadRole) -> int:
    if role == "challenger":
        return result.challenger_wins
    return result.incumbent_wins


def _head_to_head_race_team_stats(race: HeadToHeadRaceResult, role: HeadToHeadRole) -> HeadToHeadTeamRaceStats:
    if role == "challenger":
        return race.challenger
    return race.incumbent


def _head_to_head_team_scored_distance_m(stats: HeadToHeadTeamRaceStats, scoring: HeadToHeadScoring) -> float:
    if scoring == "best-copy":
        return stats.best_distance_m
    if scoring == "team-sum":
        return stats.team_sum_distance_m
    raise ValueError(f"unknown head-to-head scoring rule: {scoring}")


def _head_to_head_team_raw_distance_m(stats: HeadToHeadTeamRaceStats, scoring: HeadToHeadScoring) -> float:
    if scoring == "best-copy":
        return stats.best_raw_distance_m
    if scoring == "team-sum":
        return stats.team_sum_raw_distance_m
    raise ValueError(f"unknown head-to-head scoring rule: {scoring}")


def _aggregate_role_lap_count(result: HeadToHeadResult, role: HeadToHeadRole) -> int:
    return sum(_head_to_head_race_team_stats(race, role).total_lap_count for race in result.races)


def _aggregate_role_scored_distance_m(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(
        _head_to_head_team_scored_distance_m(_head_to_head_race_team_stats(race, role), race.scoring)
        for race in result.races
    )


def _aggregate_role_raw_distance_m(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(
        _head_to_head_team_raw_distance_m(_head_to_head_race_team_stats(race, role), race.scoring)
        for race in result.races
    )


def _aggregate_role_best_lap_seconds(result: HeadToHeadResult, role: HeadToHeadRole) -> float | None:
    lap_times: list[float] = []
    for race in result.races:
        lap_time = _head_to_head_race_team_stats(race, role).best_lap_time_seconds
        if lap_time is not None:
            lap_times.append(lap_time)
    if len(lap_times) == 0:
        return None
    return min(lap_times)


def _aggregate_role_max_speed_mps(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return max((_head_to_head_race_team_stats(race, role).max_speed_mps for race in result.races), default=0.0)


def _aggregate_role_average_damage(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    damage_total = 0.0
    damage_count = 0
    for race in result.races:
        stats = _head_to_head_race_team_stats(race, role)
        damage_total += sum(stats.damages)
        damage_count += len(stats.damages)
    if damage_count == 0:
        return 0.0
    return damage_total / damage_count


def _aggregate_role_elimination_count(result: HeadToHeadResult, role: HeadToHeadRole) -> int:
    return sum(_head_to_head_race_team_stats(race, role).elimination_count for race in result.races)


def _aggregate_role_marshal_count(result: HeadToHeadResult, role: HeadToHeadRole) -> int:
    return sum(_head_to_head_race_team_stats(race, role).total_marshal_count for race in result.races)


def _aggregate_role_marshal_penalty_m(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(_head_to_head_race_team_stats(race, role).total_marshal_penalty_m for race in result.races)


def _aggregate_role_wall_contact_seconds(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(_head_to_head_race_team_stats(race, role).total_wall_contact_seconds for race in result.races)


def _aggregate_role_car_contact_seconds(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(_head_to_head_race_team_stats(race, role).total_car_contact_seconds for race in result.races)


def _aggregate_role_low_progress_seconds(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(_head_to_head_race_team_stats(race, role).total_low_progress_seconds for race in result.races)


def _aggregate_role_off_track_seconds(result: HeadToHeadResult, role: HeadToHeadRole) -> float:
    return sum(_head_to_head_race_team_stats(race, role).total_off_track_seconds for race in result.races)


def _format_distance_m(distance_m: float) -> str:
    return f"{distance_m:.1f} m"


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.2f} s"


def _format_optional_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    return _format_seconds(seconds)


def _format_speed_mps(speed_mps: float) -> str:
    return f"{speed_mps:.1f} m/s"


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def head_to_head_team_stats_from_runtimes(
    *,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    role: HeadToHeadRole,
) -> HeadToHeadTeamRaceStats:
    """Collect scored distance, damage, contact, and lap stats for one team."""
    if len(entries) != len(runtimes):
        raise ValueError("entries and runtimes must have the same length")
    role_runtimes = tuple(runtime for entry, runtime in zip(entries, runtimes, strict=True) if entry.role == role)
    if len(role_runtimes) == 0:
        raise ValueError(f"expected at least one race result for {role}")
    return HeadToHeadTeamRaceStats(
        distances_m=tuple(race_scored_distance_m(runtime) for runtime in role_runtimes),
        lap_counts=tuple(runtime.tracker.lap_count for runtime in role_runtimes),
        wall_contact_seconds=tuple(runtime.tracker.wall_contact_seconds for runtime in role_runtimes),
        car_contact_seconds=tuple(runtime.tracker.car_contact_seconds for runtime in role_runtimes),
        damages=tuple(robot_score_damage(runtime.robot) for runtime in role_runtimes),
        raw_distances_m=tuple(runtime.tracker.best_distance_m for runtime in role_runtimes),
        max_speeds_mps=tuple(runtime.max_speed_mps for runtime in role_runtimes),
        best_lap_times_seconds=tuple(
            min(runtime.tracker.lap_times_seconds) if runtime.tracker.lap_times_seconds else None
            for runtime in role_runtimes
        ),
        penalized_distances_m=tuple(runtime.tracker.penalized_distance_m for runtime in role_runtimes),
        low_progress_seconds=tuple(runtime.low_progress_seconds for runtime in role_runtimes),
        off_track_seconds=tuple(runtime.off_track_seconds for runtime in role_runtimes),
        marshal_counts=tuple(runtime.marshal_count for runtime in role_runtimes),
        marshal_penalties_m=tuple(runtime.marshal_penalty_m for runtime in role_runtimes),
    )


def controller_for_copy(controller: RobotController) -> RobotController:
    """Give each copied car its own controller instance when possible."""
    copy_for_car = getattr(controller, "copy_for_car", None)
    if callable(copy_for_car):
        return cast(RobotController, copy_for_car())
    return controller


def _run_headless_student_race(
    *,
    render: Any,
    challenger_controller: RobotController,
    incumbent_controller: RobotController,
    race_index: int,
    round_seconds: float,
    random_seed: int,
    model: TrackProgressModel,
    samples: tuple[TrackPoint, ...],
    rules: HeadToHeadRaceRules,
    challenger_copies: int,
    incumbent_copies: int,
    fixed_delta_seconds: float,
    sensor_sample_callback: Callable[[HeadToHeadRaceEntry, RobotSensors], None] | None = None,
) -> HeadToHeadRaceResult:
    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])
    root = render.attachNewNode(f"headless-h2h-{race_index}")
    add_racing_scene_collisions(physics_world=physics_world, render=root, samples=samples)
    entries = head_to_head_race_entries(
        race_index=race_index,
        random_seed=random_seed,
        challenger_copies=challenger_copies,
        incumbent_copies=incumbent_copies,
    )
    spawn_poses = race_spawn_poses(
        len(entries),
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=random_seed,
        race_index=race_index,
    )
    controllers = tuple(
        controller_for_copy(challenger_controller if entry.role == "challenger" else incumbent_controller)
        for entry in entries
    )
    runtimes: list[RaceCarRuntime] = []
    try:
        for index, (entry, pose) in enumerate(zip(entries, spawn_poses, strict=True)):
            robot = create_robot_vehicle(
                world=physics_world,
                render=root,
                name=f"headless-h2h-{race_index}-{entry.role}-{entry.copy_index}-{index}",
                position=pose.position,
                heading_degrees=pose.heading_degrees,
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            )
            physics_scene.vehicles.append(robot)
            runtimes.append(
                RaceCarRuntime(robot=robot, tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=pose))
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
            recovery_config=_head_to_head_recovery_config(rules),
            damage_enabled=rules.damage_enabled,
            sensor_sample_callback=sensor_sample_callback,
        )
        challenger_stats = head_to_head_team_stats_from_runtimes(
            entries=entries,
            runtimes=tuple(runtimes),
            role="challenger",
        )
        incumbent_stats = head_to_head_team_stats_from_runtimes(
            entries=entries,
            runtimes=tuple(runtimes),
            role="incumbent",
        )
        winner = classify_head_to_head_winner(
            margin_m=head_to_head_race_margin(
                challenger=challenger_stats,
                incumbent=incumbent_stats,
                scoring=rules.scoring,
            ),
            win_margin_m=rules.win_margin_m,
        )
        return HeadToHeadRaceResult(
            race_index=race_index,
            winner=winner,
            challenger=challenger_stats,
            incumbent=incumbent_stats,
            scoring=rules.scoring,
        )
    finally:
        root.removeNode()


def _run_headless_student_runtime_for_duration(
    *,
    model: Any,
    physics_world: Any,
    physics_scene: PhysicsScene,
    entries: tuple[_RaceEntryT, ...],
    controllers: tuple[RobotController, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    duration_seconds: float,
    fixed_delta_seconds: float,
    recovery_config: RaceRecoveryConfig | None,
    damage_enabled: bool = True,
    sensor_sample_callback: Callable[[_RaceEntryT, RobotSensors], None] | None = None,
    race_complete: Callable[[float], bool] | None = None,
) -> None:
    elapsed_seconds = 0.0
    while True:
        complete = race_complete(elapsed_seconds) if race_complete is not None else elapsed_seconds >= duration_seconds
        if complete:
            break
        projections = _run_headless_student_runtime_step(
            model=model,
            physics_world=physics_world,
            physics_scene=physics_scene,
            entries=entries,
            controllers=controllers,
            runtimes=runtimes,
            elapsed_seconds=elapsed_seconds,
            fixed_delta_seconds=fixed_delta_seconds,
            damage_enabled=damage_enabled,
            sensor_sample_callback=sensor_sample_callback,
        )
        if recovery_config is not None:
            maybe_marshal_race_runtimes(
                model=model,
                runtimes=runtimes,
                projections=projections,
                recovery_config=recovery_config,
                delta_seconds=fixed_delta_seconds,
            )
        elapsed_seconds += fixed_delta_seconds


def _run_headless_student_runtime_step(
    *,
    model: Any,
    physics_world: Any,
    physics_scene: PhysicsScene,
    entries: tuple[_RaceEntryT, ...],
    controllers: tuple[RobotController, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    elapsed_seconds: float,
    fixed_delta_seconds: float,
    damage_enabled: bool = True,
    sensor_sample_callback: Callable[[_RaceEntryT, RobotSensors], None] | None = None,
) -> tuple[TrackProjection, ...]:
    if not (len(entries) == len(controllers) == len(runtimes)):
        raise ValueError("entries, controllers, and runtimes must have the same length")
    for entry, controller, runtime in zip(entries, controllers, runtimes, strict=True):
        if bool(getattr(runtime.robot, "eliminated", False)):
            continue
        sensors, runtime.sensor_state = build_robot_sensors(
            physics_world=physics_world,
            robot=runtime.robot,
            track_model=model,
            time_s=elapsed_seconds,
            dt_s=fixed_delta_seconds,
            previous_state=runtime.sensor_state,
            other_robot_node_names=_other_runtime_node_names(runtime=runtime, runtimes=runtimes),
            other_robots=_other_runtime_robots(runtime=runtime, runtimes=runtimes),
        )
        if sensor_sample_callback is not None:
            sensor_sample_callback(entry, sensors)
        apply_robot_vehicle_command(robot=runtime.robot, command=controller(sensors))

    physics_scene.step(fixed_delta_seconds)
    next_elapsed_seconds = elapsed_seconds + fixed_delta_seconds
    contact_states = race_contact_states(physics_world=physics_world, runtimes=runtimes)
    if damage_enabled:
        apply_wall_impact_damage(
            physics_world=physics_world,
            robots=tuple(runtime.robot for runtime in runtimes),
            fixed_time_step=physics_scene.fixed_time_step,
        )
    projections: list[TrackProjection] = []
    for runtime, contact_state in zip(runtimes, contact_states, strict=True):
        projection = project_track_position(model, robot_track_point(runtime.robot))
        projections.append(projection)
        update_race_runtime_after_step(
            runtime=runtime,
            projection=projection,
            contact_state=contact_state,
            elapsed_seconds=next_elapsed_seconds,
            delta_seconds=fixed_delta_seconds,
        )
    return tuple(projections)


def _head_to_head_copy_counts(
    *,
    copies_per_side: int = HEAD_TO_HEAD_COPIES_PER_SIDE,
    challenger_copies: int | None = None,
    incumbent_copies: int | None = None,
) -> tuple[int, int]:
    if copies_per_side < 1:
        raise ValueError("copies_per_side must be at least one")
    resolved_challenger_copies = copies_per_side if challenger_copies is None else challenger_copies
    resolved_incumbent_copies = copies_per_side if incumbent_copies is None else incumbent_copies
    _validate_head_to_head_copy_counts(
        challenger_copies=resolved_challenger_copies,
        incumbent_copies=resolved_incumbent_copies,
    )
    return resolved_challenger_copies, resolved_incumbent_copies


def _validate_head_to_head_copy_counts(*, challenger_copies: int, incumbent_copies: int) -> None:
    if challenger_copies < 1:
        raise ValueError("challenger_copies must be at least one")
    if incumbent_copies < 1:
        raise ValueError("incumbent_copies must be at least one")


def _head_to_head_result_copy_summary(result: HeadToHeadResult) -> str:
    challenger_copies, incumbent_copies = _head_to_head_result_copy_counts(result)
    if challenger_copies == incumbent_copies:
        return f"{_copy_count_text(challenger_copies)} each"
    return f"challenger {_copy_count_text(challenger_copies)}, incumbent {_copy_count_text(incumbent_copies)}"


def _head_to_head_result_copy_counts(result: HeadToHeadResult) -> tuple[int, int]:
    if len(result.races) == 0:
        return HEAD_TO_HEAD_COPIES_PER_SIDE, HEAD_TO_HEAD_COPIES_PER_SIDE
    return len(result.races[0].challenger.distances_m), len(result.races[0].incumbent.distances_m)


def _copy_count_text(count: int) -> str:
    noun = "copy" if count == 1 else "copies"
    return f"{count} {noun}"


def _head_to_head_recovery_config(rules: HeadToHeadRaceRules) -> RaceRecoveryConfig | None:
    if not rules.marshal_enabled:
        return None
    return RaceRecoveryConfig(
        stuck_seconds=rules.marshal_stuck_seconds,
        distance_penalty_m=rules.marshal_penalty_m,
        cooldown_seconds=rules.marshal_cooldown_seconds,
    )


def _other_runtime_node_names(*, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]) -> frozenset[str]:
    return frozenset(
        _node_name(other_runtime.robot.chassis_np.node())
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _other_runtime_robots(*, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]) -> tuple[Any, ...]:
    return tuple(
        other_runtime.robot
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""
