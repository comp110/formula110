from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG, vehicle_collision_bounds
from racing.race import runtime as race_runtime
from racing.race.progress import (
    LapProgressTracker,
    TrackProjection,
    build_track_progress_model,
    track_pose_at_distance,
)
from racing.race.runtime import (
    RaceCarRuntime,
    RaceContactState,
    RaceRecoveryConfig,
    maybe_marshal_race_runtimes,
    race_scored_distance_m,
    race_spawn_poses,
    seeded_race_start_finish_pose,
    start_finish_pose_for_progress,
    update_race_runtime_after_step,
)
from racing.track.spatial import track_left_vector
from racing.track.world import TrackPoint


def square_track_model_points() -> tuple[TrackPoint, ...]:
    return (
        TrackPoint(0.0, 0.0),
        TrackPoint(0.0, 10.0),
        TrackPoint(10.0, 10.0),
        TrackPoint(10.0, 0.0),
    )


def test_start_finish_pose_is_two_car_lengths_ahead_of_start_progress() -> None:
    model = build_track_progress_model(square_track_model_points())
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    pose = start_finish_pose_for_progress(
        model=model,
        start_progress_distance_m=3.0,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )

    assert pose.progress_distance_m == pytest.approx(3.0 + car_length_m * 2.0)
    assert pose.position.x == pytest.approx(0.0)
    assert pose.position.z == pytest.approx(7.6)
    assert pose.heading_degrees == pytest.approx(0.0)


def test_single_car_spawn_position_is_deterministic_for_seed() -> None:
    model = build_track_progress_model(square_track_model_points())

    first = race_spawn_poses(1, model=model, random_seed=271, race_index=1)[0]
    repeated = race_spawn_poses(1, model=model, random_seed=271, race_index=1)[0]
    different = race_spawn_poses(1, model=model, random_seed=272, race_index=1)[0]

    assert first == repeated
    assert first.progress_distance_m != different.progress_distance_m


@pytest.mark.parametrize("car_count", [4, 8])
@pytest.mark.parametrize("seed", [1, 110, 271])
@pytest.mark.parametrize("race_index", [1, 2])
def test_ordered_grid_runs_from_pole_to_back_across_track_wrap(car_count: int, seed: int, race_index: int) -> None:
    # Allow the entire eight-car grid to fit before the start line.
    model = build_track_progress_model(
        tuple(TrackPoint(point.x * 4, point.z * 4) for point in square_track_model_points())
    )
    poses = race_spawn_poses(car_count, model=model, random_seed=seed, race_index=race_index, shuffle_grid=False)
    start_finish = seeded_race_start_finish_pose(model=model, random_seed=seed, race_index=race_index)
    distances_to_line = [
        (start_finish.progress_distance_m - pose.progress_distance_m) % model.total_length_m for pose in poses
    ]

    assert distances_to_line == sorted(distances_to_line)
    assert len(set(distances_to_line)) == car_count
    assert set(poses) == set(race_spawn_poses(car_count, model=model, random_seed=seed, race_index=race_index))


def test_seeded_start_finish_pose_matches_front_spawn_progress() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_pose = race_spawn_poses(
        1,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )[0]
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0
    expected_pose = track_pose_at_distance(model, spawn_pose.progress_distance_m + car_length_m * 2.0)

    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )

    assert start_finish_pose.progress_distance_m == pytest.approx(expected_pose.progress_distance_m)
    assert start_finish_pose.position.x == pytest.approx(expected_pose.position.x)
    assert start_finish_pose.position.z == pytest.approx(expected_pose.position.z)
    assert start_finish_pose.heading_degrees == pytest.approx(expected_pose.heading_degrees)


def test_seeded_start_finish_pose_uses_front_slot_for_shuffled_grid() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_poses = race_spawn_poses(
        4,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    distances_ahead = tuple(
        (start_finish_pose.progress_distance_m - spawn_pose.progress_distance_m) % model.total_length_m
        for spawn_pose in spawn_poses
    )

    assert min(distances_ahead) == pytest.approx(car_length_m * 2.0)


def test_two_car_grid_starts_inside_car_behind_outside_car() -> None:
    model = build_track_progress_model(square_track_model_points())
    spawn_poses = race_spawn_poses(
        2,
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )
    start_finish_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=110,
        race_index=2,
    )

    def lateral_offset(spawn_pose_index: int) -> float:
        spawn_pose = spawn_poses[spawn_pose_index]
        center_pose = track_pose_at_distance(model, spawn_pose.progress_distance_m)
        left_x, left_z = track_left_vector(spawn_pose.heading_degrees)
        return (spawn_pose.position[0] - center_pose.position.x) * left_x + (
            spawn_pose.position[2] - center_pose.position.z
        ) * left_z

    inside_index = min(range(2), key=lateral_offset)
    outside_index = max(range(2), key=lateral_offset)
    distances_to_finish = tuple(
        (start_finish_pose.progress_distance_m - spawn_pose.progress_distance_m) % model.total_length_m
        for spawn_pose in spawn_poses
    )
    car_length_m = vehicle_collision_bounds(FORMULA_VEHICLE_PHYSICS_CONFIG).half_length * 2.0

    assert lateral_offset(inside_index) < 0.0
    assert lateral_offset(outside_index) > 0.0
    assert distances_to_finish[inside_index] - distances_to_finish[outside_index] == pytest.approx(car_length_m * 2.25)


def test_race_scored_distance_ignores_damage_and_applies_marshal_penalty() -> None:
    runtime = RaceCarRuntime(
        robot=cast(Any, SimpleNamespace(damage=1.0, eliminated=True)),
        tracker=LapProgressTracker(total_length_m=100.0, best_distance_m=42.0),
        marshal_penalty_m=5.0,
    )

    assert race_scored_distance_m(runtime) == 37.0


def test_race_progress_counts_while_car_is_touching_wall_and_another_car() -> None:
    runtime = RaceCarRuntime(
        robot=cast(
            Any,
            SimpleNamespace(
                vehicle=SimpleNamespace(getCurrentSpeedKmHour=lambda: 18.0),
                damage=0.0,
                eliminated=False,
            ),
        ),
        tracker=LapProgressTracker(total_length_m=100.0, starting_progress_distance_m=10.0),
    )
    projection = TrackProjection(
        position=TrackPoint(0.0, 15.0),
        nearest_center=TrackPoint(0.0, 15.0),
        progress_distance_m=15.0,
        lap_progress=0.15,
        signed_distance_to_center_m=0.0,
        heading_degrees=0.0,
    )

    update_race_runtime_after_step(
        runtime=runtime,
        projection=projection,
        contact_state=RaceContactState(wall_contact=1.0, car_contact=1.0),
        elapsed_seconds=1.0,
        delta_seconds=1.0,
    )

    assert runtime.tracker.best_distance_m == 5.0
    assert runtime.tracker.last_counted_progress_delta_m == 5.0
    assert runtime.tracker.penalized_distance_m == 0.0
    assert runtime.tracker.wall_contact_seconds == 1.0
    assert runtime.tracker.car_contact_seconds == 1.0


def _marshal_test_runtime(speed_mph: float) -> RaceCarRuntime:
    return RaceCarRuntime(
        robot=cast(
            Any,
            SimpleNamespace(
                vehicle=SimpleNamespace(getCurrentSpeedKmHour=lambda: speed_mph * 1.609344),
                config=FORMULA_VEHICLE_PHYSICS_CONFIG,
                damage=0.0,
                eliminated=False,
            ),
        ),
        tracker=LapProgressTracker(total_length_m=100.0, starting_progress_distance_m=0.0),
    )


def _marshal_test_projection(*, off_track: bool = False) -> TrackProjection:
    return TrackProjection(
        position=TrackPoint(100.0 if off_track else 0.0, 0.0),
        nearest_center=TrackPoint(0.0, 0.0),
        progress_distance_m=0.0,
        lap_progress=0.0,
        signed_distance_to_center_m=100.0 if off_track else 0.0,
        heading_degrees=0.0,
    )


@pytest.mark.parametrize(
    ("speed_mph", "expected_stuck_seconds"),
    [(0.0, 1.0), (2.99, 1.0), (3.0, 1.0), (-3.0, 1.0), (3.01, 0.0), (-3.01, 0.0), (40.0, 0.0)],
)
def test_car_contact_only_counts_as_stuck_at_three_mph_or_less(speed_mph: float, expected_stuck_seconds: float) -> None:
    runtime = _marshal_test_runtime(speed_mph)

    update_race_runtime_after_step(
        runtime=runtime,
        projection=_marshal_test_projection(),
        contact_state=RaceContactState(car_contact=1.0),
        elapsed_seconds=1.0,
        delta_seconds=1.0,
    )

    assert runtime.stuck_seconds == expected_stuck_seconds
    assert runtime.low_progress_seconds == expected_stuck_seconds
    assert runtime.tracker.car_contact_seconds == 1.0


def test_car_contact_above_three_mph_decays_existing_stuck_time_without_marshalling() -> None:
    runtime = _marshal_test_runtime(3.01)
    runtime.stuck_seconds = 1.9
    runtime.low_progress_seconds = 1.9
    projection = _marshal_test_projection()
    config = RaceRecoveryConfig(stuck_seconds=2.0, distance_penalty_m=5.0, cooldown_seconds=2.0)

    for tick in range(30):
        update_race_runtime_after_step(
            runtime=runtime,
            projection=projection,
            contact_state=RaceContactState(car_contact=1.0),
            elapsed_seconds=(tick + 1) * 0.1,
            delta_seconds=0.1,
        )
        assert (
            maybe_marshal_race_runtimes(
                runtimes=(runtime,), projections=(projection,), recovery_config=config, delta_seconds=0.1
            )
            == 0
        )

    assert runtime.stuck_seconds == 0.0
    assert runtime.low_progress_seconds == 1.9
    assert runtime.marshal_count == 0
    assert runtime.tracker.car_contact_seconds == pytest.approx(3.0)


@pytest.mark.parametrize("reason", ["stationary", "slow_contact", "wall", "off_track"])
def test_marshal_still_recovers_stationary_slow_contact_wall_and_off_track_cars(
    reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    speed_mph = 0.0 if reason == "stationary" else 3.0 if reason == "slow_contact" else 40.0
    runtime = _marshal_test_runtime(speed_mph)
    projection = _marshal_test_projection(off_track=reason == "off_track")
    config = RaceRecoveryConfig(stuck_seconds=2.0, distance_penalty_m=5.0, cooldown_seconds=2.0)
    reset = Mock()
    monkeypatch.setattr(race_runtime, "reset_robot_vehicle", reset)

    update_race_runtime_after_step(
        runtime=runtime,
        projection=projection,
        contact_state=RaceContactState(
            wall_contact=1.0 if reason == "wall" else 0.0,
            car_contact=0.0 if reason == "stationary" else 1.0,
        ),
        elapsed_seconds=2.0,
        delta_seconds=2.0,
    )
    marshalled = maybe_marshal_race_runtimes(
        runtimes=(runtime,), projections=(projection,), recovery_config=config, delta_seconds=2.0
    )

    assert marshalled == 1
    assert runtime.marshal_count == 1
    assert runtime.marshal_penalty_m == 5.0
    assert runtime.stuck_seconds == 0.0
    reset.assert_called_once()
