from __future__ import annotations

from math import dist, hypot
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from racing.game.cli import build_argument_parser
from racing.game.config import CameraView
from racing.graphics.camera import (
    HELICOPTER_CAMERA_MAX_LAG_M,
    HELICOPTER_CAMERA_OFFSET,
    CameraRig,
    apply_camera_view,
    apply_helicopter_camera_view,
    update_camera_cycle,
)


def _scene() -> tuple[SimpleNamespace, Mock, CameraRig]:
    return (
        SimpleNamespace(scene=object(), camera=Mock(aspect_ratio=16 / 9)),
        Mock(getPos=Mock(return_value=(0.0, 0.0, 0.0))),
        CameraRig(view=CameraView.HELICOPTER),
    )


@pytest.mark.parametrize("command", [[], ["h2h"], ["heat", *(["--module", "driver.py"] * 4)]])
def test_helicopter_camera_is_available_in_all_viewers(command: list[str]) -> None:
    args = build_argument_parser().parse_args([*command, "--camera", "helicopter"])
    assert CameraView(args.camera) is CameraView.HELICOPTER


def test_helicopter_pans_faster_than_it_translates_and_ignores_chassis_rotation() -> None:
    ursina, target, rig = _scene()
    apply_camera_view(ursina=ursina, view=rig.view, target=target, rig=rig)
    initial_position = ursina.camera.position
    initial_aim = ursina.camera.look_at.call_args.args[0]
    assert initial_position == HELICOPTER_CAMERA_OFFSET
    assert ursina.camera.orthographic is False

    target.getPos.return_value = (10.0, 0.0, 0.0)
    apply_camera_view(ursina=ursina, view=rig.view, target=target, rig=rig, delta_seconds=0.1)
    translation = ursina.camera.position[0] - initial_position[0]
    pan = ursina.camera.look_at.call_args.args[0][0] - initial_aim[0]
    assert 0.0 < translation < pan < 10.0
    target.getQuat.assert_not_called()
    ursina.camera.setR.assert_called_with(0.0)


def test_helicopter_smoothing_is_consistent_across_frame_rates() -> None:
    positions: list[tuple[float, float, float]] = []
    aims: list[tuple[float, float, float]] = []
    for fps in (30, 60, 120):
        ursina, target, rig = _scene()
        apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig)
        target.getPos.return_value = (10.0, 0.0, 10.0)
        for _ in range(fps):
            apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=1 / fps)
        positions.append(ursina.camera.position)
        aims.append(ursina.camera.look_at.call_args.args[0])
    for position, aim in zip(positions[1:], aims[1:], strict=True):
        assert position == pytest.approx(positions[0])
        assert aim == pytest.approx(aims[0])


def test_fast_car_cannot_pull_helicopter_overhead_or_too_far_away() -> None:
    ursina, target, rig = _scene()
    apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig)
    offset_x, _, offset_z = HELICOPTER_CAMERA_OFFSET
    for frame in range(1, 601):
        x, z = frame * 0.3, frame * -0.4
        target.getPos.return_value = (x, 0.0, z)
        apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=1 / 60)
        position = ursina.camera.position
        assert hypot(position[0] - x - offset_x, position[2] - z - offset_z) <= HELICOPTER_CAMERA_MAX_LAG_M + 1e-9
        assert hypot(position[0] - x, position[2] - z) >= 18.0 - 1e-9


@pytest.mark.parametrize("reset", ["target", "teleport", "race"])
def test_helicopter_reframes_immediately_after_target_changes_and_resets(reset: str) -> None:
    ursina, target, rig = _scene()
    apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig)
    destination = (100.0, 0.0, 100.0) if reset == "teleport" else (10.0, 0.0, 10.0)
    if reset == "target":
        target = Mock(getPos=Mock(return_value=destination))
    else:
        target.getPos.return_value = destination
    if reset == "race":
        rig.reset_follow_history()
    apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=1 / 60)
    assert ursina.camera.position == tuple(a + b for a, b in zip(destination, HELICOPTER_CAMERA_OFFSET, strict=True))
    assert ursina.camera.look_at.call_args.args[0] == (destination[0], 0.5, destination[2])


def test_zero_delta_holds_the_camera_and_a_stopped_car_eventually_settles() -> None:
    ursina, target, rig = _scene()
    apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig)
    initial_position = ursina.camera.position
    initial_aim = ursina.camera.look_at.call_args.args[0]
    target.getPos.return_value = (10.0, 0.0, 0.0)
    apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=0.0)
    assert ursina.camera.position == initial_position
    assert ursina.camera.look_at.call_args.args[0] == initial_aim

    for _ in range(600):
        apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=1 / 30)
    desired = (initial_position[0] + 10.0, initial_position[1], initial_position[2])
    assert dist(ursina.camera.position, desired) < 0.01


@pytest.mark.parametrize("include_split", [False, True])
def test_camera_cycle_includes_helicopter_once_and_clears_motion(include_split: bool) -> None:
    rig = CameraRig(view=CameraView.TOP_DOWN, selected_car_id="heat-2:0")
    visited: list[CameraView] = []
    for _ in range(8 if include_split else 7):
        rig.helicopter_position = (1.0, 2.0, 3.0)
        rig.helicopter_look_at = (4.0, 5.0, 6.0)
        rig.helicopter_target_position = (7.0, 8.0, 9.0)
        update_camera_cycle(rig, cycle_key_down=True, include_split=include_split)
        visited.append(rig.view)
        assert rig.helicopter_position is None
        assert rig.helicopter_look_at is None
        assert rig.helicopter_target_position is None
        update_camera_cycle(rig, cycle_key_down=True, include_split=include_split)
        assert rig.view is visited[-1]
        update_camera_cycle(rig, cycle_key_down=False, include_split=include_split)
    assert visited.count(CameraView.HELICOPTER) == 1
    assert rig.view is CameraView.TOP_DOWN
    assert rig.selected_car_id == "heat-2:0"
