"""Validate the intro's real Bullet stunt run without a graphics context."""

from __future__ import annotations

import math
from importlib import import_module
from itertools import pairwise
from typing import Any, cast

import pytest

from racing.intro import build_parser, pose_at
from racing.intro.driving import (
    DURATION,
    START_HEADING,
    STEP,
    DrivingLoop,
    angle_difference,
    simulate_loop,
    simulate_sequence,
)
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG


@pytest.fixture(scope="module", params=[-1, 1], ids=["away", "toward"])
def driving(request: pytest.FixtureRequest) -> DrivingLoop:
    return simulate_loop(request.param)


@pytest.mark.parametrize("seconds", [0, 1, 7.012, 10.5, 13.6, 16, 32, 39.99])
def test_lighting_and_physics_repeat_after_one_loop(driving: DrivingLoop, seconds: float) -> None:
    first, repeat = pose_at(seconds), pose_at(seconds + DURATION)
    assert repeat.logo_heading == pytest.approx(first.logo_heading)
    assert repeat.phase == pytest.approx(first.phase)
    a, b = driving.sample(seconds), driving.sample(seconds + DURATION)
    assert a.stage == b.stage
    for original, repeated in zip((a.chassis, *a.wheels), (b.chassis, *b.wheels), strict=True):
        assert repeated.position == pytest.approx(original.position)
        assert repeated.rotation == pytest.approx(original.rotation)


def test_car_brakes_parks_donuts_and_accelerates_right(driving: DrivingLoop) -> None:
    stages = [f.stage for i, f in enumerate(driving.frames) if i == 0 or driving.frames[i - 1].stage != f.stage]
    assert stages == ["waiting", "approach", "parked", "donut", "exit", "waiting"]
    parked = [f for f in driving.frames if f.stage == "parked"]
    assert len(parked) * STEP >= 1.6
    assert max(abs(f.chassis.position[0]) for f in parked) < 0.1
    assert max(f.speed for f in parked) < 0.15
    donut = [f for f in driving.frames if f.stage == "donut"]
    turn = sum(angle_difference(b.heading, a.heading) for a, b in pairwise(donut))
    assert turn * driving.turn_direction > 315
    assert max(f.skid for f in donut) > 0.5
    # Rear torque should break traction into a compact donut, rather than
    # carrying the car around a wide, mostly gripping circle.
    heading = math.radians(START_HEADING)
    for axis_x, axis_z in ((math.sin(heading), math.cos(heading)), (math.cos(heading), -math.sin(heading))):
        coordinates = [f.chassis.position[0] * axis_x + f.chassis.position[2] * axis_z for f in donut]
        assert max(coordinates) - min(coordinates) < 5.3
    assert math.dist(donut[0].chassis.position, donut[len(donut) // 2].chassis.position) > 2
    exit_frames = [f for f in driving.frames if f.stage == "exit"]
    assert all(f.throttle == 1 for f in exit_frames)
    assert exit_frames[120].speed > exit_frames[0].speed * 2
    assert exit_frames[-1].chassis.position[0] > 19
    assert not driving.sample(17).visible


def test_tires_stay_on_floor_throughout_the_stunt(driving: DrivingLoop) -> None:
    radius = FORMULA_VEHICLE_PHYSICS_CONFIG.wheel_radius
    for frame in driving.frames:
        if abs(frame.chassis.position[0]) > 20:
            continue
        for index, wheel in enumerate(frame.wheels):
            visible_radius = radius * (1.08 if index < 2 else 1.18)
            assert wheel.position[1] == pytest.approx(visible_radius, abs=0.003)
        assert frame.chassis.position[1] > 0.1


def test_wheel_tops_rotate_forward_during_approach(driving: DrivingLoop) -> None:
    core = cast(Any, import_module("panda3d.core"))
    a, b = driving.sample(7), driving.sample(7 + STEP)
    forward = core.Quat(*a.chassis.rotation).xform(core.Vec3(0, 0, 1))
    for first, second in zip(a.wheels, b.wheels, strict=True):
        before, after = core.Quat(*first.rotation), core.Quat(*second.rotation)
        local_top = before.conjugate().xform(core.Vec3(0, 1, 0))
        moved_top = after.xform(local_top)
        assert float(moved_top.dot(forward)) > 0.03


def test_front_wheels_steer_into_the_physical_donut(driving: DrivingLoop) -> None:
    core = cast(Any, import_module("panda3d.core"))
    first, second = driving.sample(12.5), driving.sample(12.5 + STEP)
    assert angle_difference(second.heading, first.heading) * driving.turn_direction > 0
    for index, wheel in enumerate(first.wheels):
        # The game's corrected Bullet axle is -X; up cross axle is forward.
        axle = core.Quat(*wheel.rotation).xform(core.Vec3(1, 0, 0))
        forward = core.Vec3(0, 1, 0).cross(axle)
        heading = math.degrees(math.atan2(float(forward.x), float(forward.z)))
        steer = angle_difference(heading, first.heading)
        assert steer == pytest.approx(25 * driving.turn_direction if index < 2 else 0, abs=0.1)


def test_run_is_deterministic_and_reset_is_offstage(driving: DrivingLoop) -> None:
    assert simulate_loop(driving.turn_direction).frames == driving.frames
    assert not driving.sample(DURATION - STEP).visible
    assert driving.sample(DURATION).chassis.position[0] < -9
    assert driving.sample(DURATION).stage == "waiting"


def test_reset_teleports_offstage_without_interpolating_across_the_screen(driving: DrivingLoop) -> None:
    reset_tick = next(
        i
        for i in range(1, len(driving.frames))
        if driving.frames[i - 1].stage == "exit" and driving.frames[i].stage == "waiting"
    )
    reset_time = reset_tick * STEP
    for fraction in (0.99, 0.5, 0.01):
        before = driving.sample(reset_time - fraction * STEP)
        assert before.visible
        assert before.chassis.position[0] > 19
    after = driving.sample(reset_time + 0.001)
    assert not after.visible
    assert after.chassis.position == driving.frames[0].chassis.position
    # Exercise consecutive loops, including fractional render times at the seam.
    for seconds in (reset_time, 17, 25, 39.999, 40, 40.001, 42.99):
        for loop in range(3):
            frame = driving.sample(seconds + loop * DURATION)
            assert not frame.visible
            assert frame.chassis.position[0] < -9
    assert driving.sample(47).visible


def test_interpolation_handles_quaternion_sign_and_remains_normalized(driving: DrivingLoop) -> None:
    for seconds in (7.012, 13.678, 14.234, 15.678):
        frame = driving.sample(seconds)
        for transform in (frame.chassis, *frame.wheels):
            assert sum(x * x for x in transform.rotation) == pytest.approx(1)
        displacement = math.dist(frame.chassis.position, driving.sample(seconds + 0.001).chassis.position)
        assert displacement == pytest.approx(frame.speed * 0.001, rel=0.05, abs=0.0002)


def test_lighting_scales_with_loop_duration() -> None:
    assert pose_at(14, 40) == pose_at(28, 80)


def test_donut_alternates_direction_and_both_runs_reset_offscreen() -> None:
    sequence = simulate_sequence()
    for loop in range(4):
        direction = -1 if loop % 2 == 0 else 1
        turning = sequence.sample(loop * DURATION + 12.8)
        assert turning.stage == "donut"
        assert turning.steer == direction
        assert turning.chassis.position[2] * direction < -3
        parked = sequence.sample(loop * DURATION + 10.5)
        assert abs(parked.chassis.position[0]) < 0.1
        assert not sequence.sample(loop * DURATION + 17).visible
    assert sequence.sample(12.8) == sequence.sample(92.8)
    assert sequence.sample(52.8) == sequence.sample(132.8)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--duration", "0"),
        ("--duration", "nan"),
        ("--duration", "inf"),
        ("--time", "nan"),
        ("--size", "0x900"),
        ("--size", "wide"),
    ],
)
def test_cli_rejects_invalid_render_configuration(flag: str, value: str) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([flag, value])


def test_timeline_rejects_nonfinite_time() -> None:
    with pytest.raises(ValueError):
        pose_at(math.nan)
