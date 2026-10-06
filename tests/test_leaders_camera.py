from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from itertools import product
from math import radians, tan
from typing import Any, cast

import pytest

from racing.game.cli import build_argument_parser
from racing.game.config import CameraView
from racing.graphics.camera import CameraRig
from racing.graphics.cinematic import CinematicCar, CinematicPose
from racing.graphics.leaders import FinishCameraSequence, LeadersCamera, leading_battle_ids
from racing.race.laps import LapRace
from racing.race.timing import TimingSample, TimingStanding


def _standings(third_gap: float | None = 2.0) -> tuple[TimingStanding, ...]:
    return tuple(TimingStanding(i, str(i), 100 - i * 10, gap)
                 for i, gap in enumerate((0.0, 1.0, third_gap, 4.0), start=1))


def _project(pose: CinematicPose, point: tuple[float, float, float], aspect: float) -> tuple[float, float]:
    core = cast(Any, import_module('panda3d.core'))
    forward = core.Vec3(*pose.look_at) - core.Vec3(*pose.position)
    forward.normalize()
    right = forward.cross(core.Vec3(0, 1, 0))
    right.normalize()
    up = right.cross(forward)
    ray = core.Vec3(*point) - core.Vec3(*pose.position)
    depth = float(ray.dot(forward))
    assert depth > 0
    return (float(ray.dot(right)) / (depth * tan(radians(pose.fov / 2))),
            float(ray.dot(up)) / (depth * tan(radians(pose.fov / 2)) / aspect))


@pytest.mark.parametrize(('gap', 'expected'), [(2.9, ('1', '2', '3')), (3.0, ('1', '2', '3')),
                                              (3.01, ('1', '2')), (None, ('1', '2'))])
def test_p3_must_be_within_three_seconds_of_leader(gap: float | None, expected: tuple[str, ...]) -> None:
    assert leading_battle_ids(_standings(gap)) == expected


def test_targets_follow_current_order_and_skip_retirements() -> None:
    rows = _standings()
    assert leading_battle_ids(tuple(reversed(rows))) == ('1', '2', '3')
    passed = tuple(replace(row, rank=rank) for row, rank in zip(rows, (3, 1, 2, 4), strict=True))
    assert leading_battle_ids(passed) == ('2', '3', '1')
    retired = tuple(replace(row, eliminated=row.car_id != '2') for row in rows)
    assert leading_battle_ids(retired) == ('2',)
    assert leading_battle_ids(()) == ()


@pytest.mark.parametrize('aspect', [4 / 3, 16 / 9, 21 / 9])
@pytest.mark.parametrize('heading', [0.0, 90.0, 179.0, 270.0])
@pytest.mark.parametrize('spread', [5.0, 40.0, 300.0])
def test_car_bodies_fit_clear_of_tower_and_cards_even_after_sudden_motion(
    aspect: float, heading: float, spread: float,
) -> None:
    director = LeadersCamera()
    cars = tuple(CinematicCar(str(i + 1), i + 1, 100 - i * 10, (i * 4.0, 0.5, i * 5.0),
                              heading_degrees=heading) for i in range(3))
    left = -1 + 0.75 / aspect
    director.update(cars, _standings(), delta_seconds=0, aspect_ratio=aspect, left_edge=left)
    cars = tuple(replace(car, position=(i * spread, 0.5, -i * spread)) for i, car in enumerate(cars))
    pose = director.update(cars, _standings(), delta_seconds=1 / 60, aspect_ratio=aspect, left_edge=left)
    assert pose is not None
    assert director.subject_ids == ('1', '2', '3')
    for car in cars:
        for dx, dy, dz in product((-2.5, 2.5), repeat=3):
            x, y = _project(pose, (car.position[0] + dx, car.position[1] + dy, car.position[2] + dz), aspect)
            assert left <= x <= 0.88
            assert -0.80 <= y <= 0.65


def test_finishing_camera_stays_fixed_while_cars_drive_away_and_includes_finish_line() -> None:
    director = LeadersCamera()
    cars = (CinematicCar('1', 1, 100, (0, 0.5, 1)), CinematicCar('2', 2, 90, (5, 0.5, -10)))
    finish_line = ((-10.0, 0.1, 0.0), (10.0, 0.1, 0.0))
    pose = director.update(cars, _standings(), delta_seconds=0, aspect_ratio=16 / 9,
                           finishing=True, finish_points=finish_line)
    assert pose is not None
    for point in finish_line:
        x, y = _project(pose, point, 16 / 9)
        assert -0.6 < x < 0.88 and -0.8 < y < 0.65
    moved = tuple(replace(car, position=(200, 0.5, 200)) for car in cars)
    for _ in range(120):
        assert director.update(moved, _standings(), delta_seconds=1 / 60, aspect_ratio=16 / 9,
                               finishing=True, finish_points=finish_line) == pose
    rig = CameraRig(view=CameraView.LEADERS, leaders=director)
    rig.reset_follow_history()
    assert rig.leaders.finish_pose is None


def test_finish_sequence_waits_for_second_final_lap_crossing_then_holds_one_second() -> None:
    race = LapRace(round_laps=2, track_length_m=100)
    sequence = FinishCameraSequence()
    for elapsed, first, second in ((0, 0, -5), (10, 101, 99), (11, 110, 101), (20, 201, 199)):
        race.update(elapsed_seconds=elapsed, samples=(TimingSample('1', first), TimingSample('2', second)))
        sequence.advance(len(race.finish_times), 0.1)
        assert not sequence.pulling_back
        assert sequence.second_finish_elapsed is None
    race.update(elapsed_seconds=21, samples=(TimingSample('1', 210), TimingSample('2', 201)))
    sequence.advance(len(race.finish_times), 0.1)
    assert race.complete
    assert sequence.second_finish_elapsed == 0
    sequence.advance(2, 0.99)
    assert not sequence.pulling_back
    sequence.advance(2, 0.01)
    assert sequence.pulling_back and not sequence.complete
    sequence.advance(2, 2.99)
    assert not sequence.complete
    sequence.advance(2, 0.01)
    assert sequence.complete


@pytest.mark.parametrize('command', [[], ['h2h'], ['heat', '--module', 'driver.py', '--module', 'other.py']])
def test_leaders_camera_is_available_from_cli(command: list[str]) -> None:
    args = build_argument_parser().parse_args([*command, '--camera', 'leaders'])
    assert CameraView(args.camera) is CameraView.LEADERS
