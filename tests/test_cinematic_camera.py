from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from itertools import pairwise
from math import atan2, cos, degrees, dist, isfinite, radians, sin, tan
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from racing.game.app import _cinematic_race_cars  # pyright: ignore[reportPrivateUsage]
from racing.game.cli import build_argument_parser
from racing.game.config import CameraView
from racing.graphics.camera import (
    CameraRig,
    TrackCameraFrame,
    apply_camera_view,
    starting_grid_camera_pose,
    update_camera_cycle,
)
from racing.graphics.cinematic import (
    ANGLE_SHOT_SECONDS,
    MAX_ELEVATION_SPEED_DEG_S,
    MAX_ORBIT_ACCEL_DEG_S2,
    MAX_ORBIT_SPEED_DEG_S,
    MAX_ZOOM_OUT_MPS,
    MIN_SHOT_SECONDS,
    SUBJECT_TRANSITION_SECONDS,
    CinematicAngle,
    CinematicCar,
    CinematicDirector,
    CinematicPose,
)
from racing.graphics.track_rendering import (
    START_FINISH_BANNER_CENTER_Y,
    START_FINISH_BANNER_HEIGHT,
    START_FINISH_BANNER_OVERHANG,
    START_FINISH_BANNER_THICKNESS,
    start_finish_banner_side_distances,
    start_finish_render_pose,
)
from racing.race.heat import HeatRaceEntry
from racing.race.progress import LapProgressTracker, build_track_progress_model, resolve_track
from racing.race.runtime import RaceCarRuntime, race_spawn_poses, seeded_race_start_finish_pose
from racing.race.start import GRID_CAMERA_TRANSITION_SECONDS
from racing.race.timing import TimingStanding
from racing.track.world import TrackPoint


def car(rank: int, distance: float, speed: float = 20.0, *, car_id: str | None = None) -> CinematicCar:
    return CinematicCar(car_id or str(rank), rank, distance, (0.0, 0.0, distance), speed)


def advance(director: CinematicDirector, cars: tuple[CinematicCar, ...], seconds: float) -> None:
    for _ in range(round(seconds * 60)):
        director.update(cars, delta_seconds=1 / 60)


def screen_position(pose: CinematicPose, position: tuple[float, float, float], aspect: float) -> tuple[float, float]:
    """Project onto the real lens's horizontal and vertical view limits."""
    core = cast(Any, import_module("panda3d.core"))
    lens = core.PerspectiveLens()
    lens.setAspectRatio(aspect)
    lens.setFov(pose.fov)
    forward = core.Vec3(*pose.look_at) - core.Vec3(*pose.position)
    forward.normalize()
    right = forward.cross(core.Vec3(0, 1, 0))
    right.normalize()
    up = right.cross(forward)
    ray = core.Vec3(*position) - core.Vec3(*pose.position)
    depth = float(ray.dot(forward))
    assert depth > 0
    return (
        float(ray.dot(right)) / (depth * tan(radians(float(lens.getHfov()) / 2))),
        float(ray.dot(up)) / (depth * tan(radians(float(lens.getVfov()) / 2))),
    )


@pytest.mark.parametrize("aspect", [16 / 9, 4 / 3, 9 / 16])
def test_grid_intro_frames_the_entire_track_then_settles_with_two_lights_left(aspect: float) -> None:
    frame = TrackCameraFrame(center_x=50, center_z=-20, width=500, length=300)
    destination = CinematicPose((70, 8, 45), (60, 1, 60), 48)
    opening = starting_grid_camera_pose(destination, frame=frame, aspect_ratio=aspect, elapsed_seconds=0)
    assert opening.position[0] == pytest.approx(opening.look_at[0], abs=0.011)
    assert opening.position[2] == pytest.approx(opening.look_at[2], abs=0.011)
    for x in (frame.center_x - frame.width / 2, frame.center_x + frame.width / 2):
        for z in (frame.center_z - frame.length / 2, frame.center_z + frame.length / 2):
            screen_x, screen_y = screen_position(opening, (x, 0, z), aspect)
            assert abs(screen_x) < 0.65
            assert -0.50 < screen_y < 0.80
    positions = [starting_grid_camera_pose(
        destination, frame=frame, aspect_ratio=aspect, elapsed_seconds=i / 60,
    ) for i in range(round(GRID_CAMERA_TRANSITION_SECONDS * 60) + 1)]
    assert positions[-1] == destination
    assert all(a.position[1] > b.position[1] for a, b in pairwise(positions))
    assert dist(positions[0].position, positions[1].position) < 1.0
    assert dist(positions[-2].position, positions[-1].position) < 0.1


@pytest.mark.parametrize("command", [[], ["h2h"], ["heat", "--module", "controllers.demo"]])
def test_cinematic_cli_option(command: list[str]) -> None:
    args = build_argument_parser().parse_args([*command, "--camera", "cinematic"])
    assert CameraView(args.camera) is CameraView.CINEMATIC


@pytest.mark.parametrize("include_split", [False, True])
def test_v_cycles_cinematic_once_and_resets_director(include_split: bool) -> None:
    rig = CameraRig(view=CameraView.TOP_DOWN, selected_car_id="3")
    visited: list[CameraView] = []
    for _ in range(8 if include_split else 7):
        rig.cinematic.update((car(1, 100),), delta_seconds=0)
        update_camera_cycle(rig, cycle_key_down=True, include_split=include_split)
        visited.append(rig.view)
        assert rig.cinematic.pose is None
        update_camera_cycle(rig, cycle_key_down=True, include_split=include_split)
        assert rig.view is visited[-1]
        update_camera_cycle(rig, cycle_key_down=False, include_split=include_split)
    assert visited.count(CameraView.CINEMATIC) == 1
    assert rig.view is CameraView.TOP_DOWN
    assert rig.selected_car_id == "3"


def test_prefers_a_closing_podium_battle_to_a_runaway_leader() -> None:
    director = CinematicDirector()
    director.update((car(1, 200), car(2, 100, 18), car(3, 88, 24)), delta_seconds=0)
    assert director.subject_ids == ("2", "3")
    assert director.angle is CinematicAngle.CHASE


def test_closing_speed_can_outweigh_a_closer_but_static_gap() -> None:
    director = CinematicDirector()
    director.update((car(1, 100, 20), car(2, 96, 20), car(3, 81, 30)), delta_seconds=0)
    assert director.subject_ids == ("2", "3")


def test_challenge_for_third_counts_but_a_battle_for_fifth_does_not() -> None:
    director = CinematicDirector()
    cars = (car(1, 300), car(2, 200), car(3, 100), car(4, 97, 24), car(5, 50), car(6, 49, 35))
    director.update(cars, delta_seconds=0)
    assert director.subject_ids == ("3", "4")
    assert director.angle is CinematicAngle.SIDE
    director = CinematicDirector()
    director.update(
        tuple(replace(c, distance_m=150, position=(0, 0, 150)) if c.rank == 3 else c for c in cars), delta_seconds=0
    )
    assert director.subject_ids == ("1",)


def test_physical_proximity_does_not_mistake_lapped_cars_for_a_battle() -> None:
    director = CinematicDirector()
    director.update((car(1, 1000), replace(car(2, 100), position=(1, 0, 1000))), delta_seconds=0)
    assert director.subject_ids == ("1",)


def test_track_gap_alone_does_not_frame_cars_on_distant_sections() -> None:
    director = CinematicDirector()
    director.update((car(1, 100), replace(car(2, 99), position=(100, 0, 99))), delta_seconds=0)
    assert director.subject_ids == ("1",)


def test_holds_story_before_switching_to_a_more_urgent_battle() -> None:
    director = CinematicDirector()
    director.update((car(1, 100), car(2, 97), car(3, 70)), delta_seconds=0)
    assert director.subject_ids == ("1", "2")
    cars = (car(1, 100), car(2, 75), car(3, 73, 30))
    advance(director, cars, MIN_SHOT_SECONDS - 0.5)
    assert director.subject_ids == ("1", "2")
    advance(director, cars, 1.0)
    assert director.subject_ids == ("2", "3")


def test_small_score_fluctuations_do_not_ping_pong_between_pairs() -> None:
    director = CinematicDirector()
    director.update((car(1, 100), car(2, 90), car(3, 79)), delta_seconds=0)
    for frame in range(900):
        director.update((car(1, 100), car(2, 90), car(3, 80 + frame % 2)), delta_seconds=1 / 60)
        assert director.subject_ids == ("1", "2")


def test_distant_subject_handoff_eases_out_and_settles_on_the_battle() -> None:
    director = CinematicDirector()
    director.update((car(1, 200), car(2, 100), car(3, 50)), delta_seconds=0)
    advance(director, (car(1, 200), car(2, 100), car(3, 50)), MIN_SHOT_SECONDS)
    before = after = director.pose
    for _ in range(120):
        before = director.pose
        after = director.update((car(1, 200), car(2, 100), car(3, 99)), delta_seconds=1 / 60)
        if director.subject_ids == ("2", "3"):
            break
    assert before is not None and after is not None
    assert director.subject_ids == ("2", "3")
    assert dist(before.position, after.position) < 1.0
    assert dist(before.look_at, after.look_at) < 0.1
    advance(director, (car(1, 200), car(2, 100), car(3, 99)), SUBJECT_TRANSITION_SECONDS + 2.0)
    assert director.pose is not None
    assert director.pose.look_at[2] == pytest.approx(99.5, abs=0.1)


def test_a_pass_keeps_the_same_battle_and_continuous_camera() -> None:
    director = CinematicDirector()
    before = director.update((car(1, 100), car(2, 99)), delta_seconds=0)
    after = director.update((car(2, 100, car_id="1"), car(1, 101, car_id="2")), delta_seconds=1 / 60)
    assert director.subject_ids == ("1", "2")
    assert before is not None and after is not None
    assert dist(before.position, after.position) < 1.0
    assert director.shot_seconds > 0.0


def test_an_intervening_car_does_not_interrupt_the_shot() -> None:
    director = CinematicDirector()
    director.update((car(1, 120), car(2, 100, 18), car(3, 98, 25), car(4, 85)), delta_seconds=0)
    assert director.subject_ids == ("2", "3")
    director.update(
        (car(1, 120), car(4, 100, car_id="2"), car(2, 105, car_id="3"), car(3, 103, car_id="4")), delta_seconds=1 / 60
    )
    assert director.subject_ids == ("2", "3")


def test_gap_threshold_has_hysteresis() -> None:
    director = CinematicDirector()
    director.update((car(1, 100), car(2, 70, 30)), delta_seconds=0)
    assert director.subject_ids == ("1", "2")
    for frame in range(120):
        director.update((car(1, 100), car(2, 58 + (1 if frame % 2 else -1), 30)), delta_seconds=1 / 60)
        assert director.subject_ids == ("1", "2")


def test_nearby_marshal_recovery_is_smoothed() -> None:
    director = CinematicDirector()
    before = director.update((car(1, 100),), delta_seconds=0)
    after = director.update((replace(car(1, 110), recovery_count=1),), delta_seconds=1 / 60)
    assert before is not None and after is not None
    assert dist(before.position, after.position) < 0.1


def test_retirement_retargets_without_waiting_and_all_retired_holds_pose() -> None:
    director = CinematicDirector()
    first = car(1, 100)
    second = car(2, 99)
    director.update((first, second), delta_seconds=0)
    pose = director.update((replace(first, eliminated=True), second), delta_seconds=1 / 60)
    assert director.subject_ids == ("2",)
    assert director.update((replace(second, eliminated=True),), delta_seconds=1 / 60) == pose


def test_marshal_recovery_reframes_instead_of_flying_across_track() -> None:
    director = CinematicDirector()
    director.update((car(1, 100),), delta_seconds=0)
    moved = replace(car(1, 100), position=(500, 0, 500), recovery_count=1)
    pose = director.update((moved,), delta_seconds=1 / 60)
    assert pose is not None
    assert dist(pose.look_at, moved.position) < 1.0


def test_zero_and_negative_delta_hold_the_pose_and_shot_timer() -> None:
    director = CinematicDirector()
    pose = director.update((car(1, 100),), delta_seconds=0)
    for dt in (0.0, -1.0):
        assert director.update((car(1, 110),), delta_seconds=dt) == pose
        assert director.shot_seconds == 0.0


def test_smoothing_is_frame_rate_independent_for_a_fixed_destination() -> None:
    poses: list[CinematicPose | None] = []
    for fps in (30, 60, 120):
        director = CinematicDirector()
        director.update((car(1, 100),), delta_seconds=0)
        for _ in range(fps):
            director.update((car(1, 110),), delta_seconds=1 / fps)
        poses.append(director.pose)
    assert all(p is not None for p in poses)
    assert poses[0] is not None
    for pose in poses[1:]:
        assert pose is not None
        assert pose.position == pytest.approx(poses[0].position)
        assert pose.look_at == pytest.approx(poses[0].look_at)


def test_angle_change_glides_without_cutting_or_zoom_jump() -> None:
    director = CinematicDirector()
    director.update((car(1, 100), car(2, 85)), delta_seconds=0)
    cars = (car(1, 100), car(2, 97))
    advance(director, cars, ANGLE_SHOT_SECONDS - 0.1)
    before = director.pose
    assert before is not None
    for _ in range(30):
        after = director.update(cars, delta_seconds=1 / 60)
        assert after is not None
        assert dist(before.position, after.position) < 1.0
        assert abs(before.fov - after.fov) < 0.1
        before = after
    assert director.angle is CinematicAngle.SIDE


def test_quiet_shot_does_not_change_angle_just_for_variety() -> None:
    director = CinematicDirector()
    first = director.update((car(1, 100),), delta_seconds=0)
    advance(director, (car(1, 100),), 40.0)
    assert director.pose == first
    assert director.angle is CinematicAngle.CHASE


def test_corner_heading_does_not_orbit_an_established_shot() -> None:
    director = CinematicDirector()
    first = director.update((car(1, 100),), delta_seconds=0)
    advance(director, (replace(car(1, 100), heading_degrees=170),), 6.0)
    assert director.pose == first


def test_brief_interest_spike_does_not_switch_a_settled_shot() -> None:
    director = CinematicDirector()
    cars = (car(1, 200), car(2, 100), car(3, 50))
    advance(director, cars, 10.0)
    advance(director, (car(1, 200), car(2, 100), car(3, 99)), 0.8)
    advance(director, cars, 1.0)
    assert director.subject_ids == ("1",)


@pytest.mark.parametrize("fps", [30, 60, 120])
def test_camera_motion_limits_survive_large_angle_changes_and_slow_frames(fps: int) -> None:
    director = CinematicDirector()
    director.update((car(1, 100), car(2, 85)), delta_seconds=0)
    advance(director, (car(1, 100), car(2, 85)), 15.0)
    old_pose = director.pose
    old_rate = 0.0
    assert old_pose is not None
    for frame in range(8 * fps):
        dt = 0.25 if frame == fps * 2 else 1 / fps
        pose = director.update(
            (replace(car(1, 100), heading_degrees=160), car(2, 99)),
            delta_seconds=dt,
        )
        assert pose is not None
        old_offset = tuple(p - c for p, c in zip(old_pose.position, old_pose.look_at, strict=True))
        offset = tuple(p - c for p, c in zip(pose.position, pose.look_at, strict=True))
        old_angle = degrees(atan2(old_offset[0], old_offset[2]))
        angle = degrees(atan2(offset[0], offset[2]))
        rate = ((angle - old_angle + 180) % 360 - 180) / dt
        assert abs(rate) <= MAX_ORBIT_SPEED_DEG_S + 1e-7
        if frame not in (fps * 2, fps * 2 + 1):
            assert abs(rate - old_rate) / dt <= MAX_ORBIT_ACCEL_DEG_S2 + 1e-7
        old_elevation = degrees(atan2(old_offset[1], (old_offset[0] ** 2 + old_offset[2] ** 2) ** 0.5))
        elevation = degrees(atan2(offset[1], (offset[0] ** 2 + offset[2] ** 2) ** 0.5))
        assert abs(elevation - old_elevation) / dt <= MAX_ELEVATION_SPEED_DEG_S + 1e-7
        assert (
            abs(dist(pose.position, pose.look_at) - dist(old_pose.position, old_pose.look_at)) / dt <= MAX_ZOOM_OUT_MPS
        )
        assert pose.fov == old_pose.fov
        old_pose, old_rate = pose, rate
    assert director.angle is CinematicAngle.SIDE


def test_heading_wrap_takes_the_short_path() -> None:
    director = CinematicDirector()
    before = director.update((replace(car(1, 100), heading_degrees=179),), delta_seconds=0)
    after = director.update((replace(car(1, 100), heading_degrees=-179),), delta_seconds=1 / 60)
    assert before is not None and after is not None
    assert dist(before.position, after.position) < 0.1


def test_corner_uses_a_wide_aerial_composition() -> None:
    model = build_track_progress_model((TrackPoint(0, 0), TrackPoint(0, 10), TrackPoint(10, 10), TrackPoint(10, 0)))
    director = CinematicDirector()
    director.update((car(1, 0),), delta_seconds=0, track_model=model)
    assert director.angle is CinematicAngle.AERIAL


def test_side_camera_rises_to_see_a_car_hugging_the_near_barrier() -> None:
    model = build_track_progress_model((TrackPoint(0, 0), TrackPoint(0, 500), TrackPoint(100, 500), TrackPoint(100, 0)))
    cars = (
        replace(car(1, 100), position=(4.0, 0.1, 100), track_distance_m=100),
        replace(car(2, 99), position=(3.6, 0.1, 99), track_distance_m=99),
    )
    pose = CinematicDirector().update(cars, delta_seconds=0, track_model=model)
    assert pose is not None
    assert (pose.position[1] - pose.look_at[1]) / dist(pose.position, pose.look_at) > 0.85


@pytest.mark.parametrize("aspect", [0.6, 1.0, 16 / 9, 32 / 9])
def test_frames_both_subjects_and_widens_for_large_gaps(aspect: float) -> None:
    close = CinematicDirector().update((car(1, 100), car(2, 99)), delta_seconds=0, aspect_ratio=aspect)
    wide = CinematicDirector().update((car(1, 100), car(2, 65, 28)), delta_seconds=0, aspect_ratio=aspect)
    assert close is not None and wide is not None
    assert all(isfinite(v) for v in (*wide.position, *wide.look_at, wide.fov))
    assert dist(wide.position, wide.look_at) > dist(close.position, close.look_at)
    assert wide.look_at[2] == pytest.approx(82.5)
    for position in ((0, 0, 100), (0, 0, 65)):
        x, y = screen_position(wide, position, aspect)
        assert abs(x) < 0.85 and abs(y) < 0.85


def test_closing_battle_tightens_smoothly_while_both_moving_cars_stay_visible() -> None:
    director = CinematicDirector()
    first = director.update((car(1, 100), car(2, 68, 22)), delta_seconds=0)
    assert first is not None
    previous_distance = dist(first.position, first.look_at)
    first_distance = previous_distance
    for frame in range(1, 1201):
        time = frame / 60
        gap = max(2.0, 32 - 2 * time)
        cars = (car(1, 100 + 20 * time), car(2, 100 + 20 * time - gap, 22 if gap > 2 else 20))
        pose = director.update(cars, delta_seconds=1 / 60)
        assert pose is not None
        distance = dist(pose.position, pose.look_at)
        assert (previous_distance - distance) * 60 <= 3.0 + 1e-6
        for subject in cars:
            x, y = screen_position(pose, subject.position, 16 / 9)
            assert abs(x) < 0.95 and abs(y) < 0.95
        previous_distance = distance
    assert previous_distance < first_distance * 0.7


@pytest.mark.parametrize("count", [2, 4, 8, 9, 10])
@pytest.mark.parametrize("aspect", [4 / 3, 16 / 9, 32 / 9])
@pytest.mark.parametrize("track_id", ["mugello-short", "bahrain"])
@pytest.mark.parametrize("seed", [1, 100, 110])
def test_opening_grid_frames_cars_and_complete_banner_at_top(
    count: int, aspect: float, track_id: str, seed: int,
) -> None:
    model = resolve_track(track_id).model
    spawns = race_spawn_poses(count, model=model, random_seed=seed, shuffle_grid=False)
    cars = tuple(
        CinematicCar(
            str(index), index + 1, -index * 6.0, spawn.position, speed_mps=0, track_distance_m=spawn.progress_distance_m
        )
        for index, spawn in enumerate(spawns)
    )
    director = CinematicDirector()
    pose = director.update(cars, delta_seconds=0, aspect_ratio=aspect, track_model=model, grid=True)
    assert pose is not None
    assert len(director.subject_ids) == count
    assert director.angle is CinematicAngle.GRID
    assert 1.0 < pose.position[1] < 16.0
    for subject in cars:
        x, y = screen_position(pose, subject.position, aspect)
        assert abs(x) < 0.62 and -0.50 < y < 0.95
    for spawn in spawns:
        heading = radians(spawn.heading_degrees)
        for forward, side in ((-1.3, -1.0), (-1.3, 1.0), (1.3, -1.0), (1.3, 1.0)):
            for height in (0.0, 1.0):
                corner = (
                    spawn.position[0] + sin(heading) * forward + cos(heading) * side,
                    spawn.position[1] + height,
                    spawn.position[2] + cos(heading) * forward - sin(heading) * side,
                )
                x, y = screen_position(pose, corner, aspect)
                assert abs(x) < 0.621 and -0.501 < y < 0.901
    start = seeded_race_start_finish_pose(model=model, random_seed=seed)
    banner = start_finish_render_pose(samples=model.points, position=start.position)
    negative, positive = start_finish_banner_side_distances(samples=model.points, position=banner.position)
    heading = radians(banner.heading_degrees)
    banner_top = -1.0
    for side in (-negative - START_FINISH_BANNER_OVERHANG / 2, positive + START_FINISH_BANNER_OVERHANG / 2):
        for depth in (-START_FINISH_BANNER_THICKNESS / 2, START_FINISH_BANNER_THICKNESS / 2):
            for height in (-START_FINISH_BANNER_HEIGHT / 2, START_FINISH_BANNER_HEIGHT / 2):
                corner = (
                    banner.position.x - cos(heading) * side + sin(heading) * depth,
                    START_FINISH_BANNER_CENTER_Y + height,
                    banner.position.z + sin(heading) * side + cos(heading) * depth,
                )
                x, y = screen_position(pose, corner, aspect)
                assert abs(x) < 0.621 and -0.50 < y < 0.98
                assert x * aspect > -aspect + 0.65  # Clear the expanded timing tower.
                banner_top = max(banner_top, y)
    assert banner_top == pytest.approx(0.97, abs=0.002)
    after = director.update(cars, delta_seconds=1 / 60, aspect_ratio=aspect, track_model=model, grid=False)
    assert after is not None
    assert len(director.subject_ids) <= 2
    assert dist(after.position, pose.position) < 0.1
    assert abs(after.fov - pose.fov) < 0.2


def test_live_samples_use_tower_order_distance_and_recovery_identity() -> None:
    runtime = RaceCarRuntime(
        robot=cast(Any, SimpleNamespace(chassis_np=Mock(getPos=lambda: (3, 0, 5)), eliminated=False)),
        tracker=LapProgressTracker(total_length_m=100, last_progress_distance_m=5),
        recent_progress_mps=22,
        marshal_count=2,
    )
    samples = _cinematic_race_cars(
        entries=(HeatRaceEntry(0),), runtimes=(runtime,), standings=(TimingStanding(3, "heat-0:0", 205, None),)
    )
    assert samples == (CinematicCar("heat-0:0", 3, 205, (3, 0, 5), 22, 5, recovery_count=2),)


def test_grid_samples_use_spawn_progress_before_the_first_physics_tick() -> None:
    runtime = RaceCarRuntime(
        robot=cast(Any, SimpleNamespace(chassis_np=Mock(getPos=lambda: (3, 0, 5)), eliminated=False)),
        tracker=LapProgressTracker(total_length_m=100, starting_progress_distance_m=72),
    )
    samples = _cinematic_race_cars(
        entries=(HeatRaceEntry(0),), runtimes=(runtime,), standings=(TimingStanding(1, "heat-0:0", 0, None),)
    )
    assert samples[0].track_distance_m == 72


def test_cinematic_renderer_uses_battle_even_with_an_existing_manual_selection() -> None:
    ursina = cast(Any, SimpleNamespace(scene=object(), camera=Mock(aspect_ratio=16 / 9)))
    rig = CameraRig(view=CameraView.CINEMATIC, selected_car_id="1")
    apply_camera_view(
        ursina=ursina, view=rig.view, target=Mock(), rig=rig, cinematic_cars=(car(1, 200), car(2, 100), car(3, 98))
    )
    assert rig.cinematic.subject_ids == ("2", "3")
    assert ursina.camera.orthographic is False
    assert ursina.camera.position == rig.cinematic.pose.position  # pyright: ignore[reportOptionalMemberAccess]
    ursina.camera.setR.assert_called_with(0.0)
    rig.select_follow_car("3")
    assert rig.view is CameraView.HELICOPTER
    assert rig.cinematic.pose is None


def test_solo_renderer_builds_a_cinematic_subject_from_the_car() -> None:
    ursina = cast(Any, SimpleNamespace(scene=object(), camera=Mock(aspect_ratio=16 / 9)))
    target = Mock(getPos=lambda: (3, 0, 10), getH=lambda: 0)
    rig = CameraRig(view=CameraView.CINEMATIC)
    apply_camera_view(ursina=ursina, view=rig.view, target=target, rig=rig)
    assert rig.cinematic.subject_ids == ("solo",)
    assert rig.cinematic.pose is not None
    assert rig.cinematic.pose.look_at == (3, 0.6, 10)
