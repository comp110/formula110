from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from racing.game.config import CameraView
from racing.graphics.camera import (
    FORMULA_FOLLOW_CAMERA_SETTINGS,
    CameraRig,
    apply_camera_view,
    apply_follow_camera_view,
    perspective_near_clip_for_height,
)
from racing.graphics.cinematic import CinematicCar
from racing.graphics.track_rendering import TRACK_KERB_LOW_HEIGHT, TRACK_SURFACE_Y


@pytest.mark.parametrize("height", [0.0, 1.0, 1.66, 1.9])
def test_close_camera_preserves_nearby_bodywork(height: float) -> None:
    assert perspective_near_clip_for_height(height) == pytest.approx(0.1)


@pytest.mark.parametrize("height", [10.0, 35.0, 100.0, 200.0, 300.0])
def test_distant_camera_resolves_the_shallow_kerb_above_the_ground(height: float) -> None:
    near = perspective_near_clip_for_height(height)
    far = 10000.0
    # A conservative oblique view: depth separation is smaller than the actual
    # kerb height. Keep it well above a single unit of a 24-bit depth buffer.
    ground_distance = height / 0.8
    kerb_distance = ground_distance - TRACK_KERB_LOW_HEIGHT * 0.8

    def depth(distance: float) -> float:
        return far / (far - near) - far * near / ((far - near) * distance)

    depth_steps = (depth(ground_distance) - depth(kerb_distance)) * (2**24 - 1)
    assert depth_steps > 16.0
    assert 0.1 < near <= 10.0
    assert near < (height - TRACK_SURFACE_Y) * 0.1


def test_far_cinematic_then_follow_restores_close_near_plane() -> None:
    core = cast(Any, import_module("panda3d.core"))
    lens = core.PerspectiveLens()
    lens.setNearFar(0.1, 10000.0)
    camera = Mock(aspect_ratio=16 / 9, perspective_lens=lens)
    ursina = SimpleNamespace(scene=object(), camera=camera, Vec3=core.Vec3)
    target = Mock(getPos=Mock(return_value=(0.0, 0.0, 0.0)), getH=Mock(return_value=0.0))
    rig = CameraRig(view=CameraView.CINEMATIC)
    apply_camera_view(
        ursina=ursina,
        view=rig.view,
        target=target,
        rig=rig,
        cinematic_cars=(
            CinematicCar("leader", 1, 100, (0, 0, 100), speed_mps=30),
            CinematicCar("challenger", 2, 70, (0, 0, 70), speed_mps=40),
        ),
    )
    assert lens.getNear() > 3.0
    assert lens.getFar() == pytest.approx(10000.0)
    apply_camera_view(
        ursina=ursina, view=CameraView.FOLLOW, target=target, rig=rig, follow_settings=FORMULA_FOLLOW_CAMERA_SETTINGS
    )
    assert lens.getNear() == pytest.approx(0.1)
    assert lens.getFar() == pytest.approx(10000.0)


def test_helicopter_also_uses_altitude_to_preserve_depth_precision() -> None:
    core = cast(Any, import_module("panda3d.core"))
    lens = core.PerspectiveLens()
    ursina = SimpleNamespace(scene=object(), camera=Mock(aspect_ratio=16 / 9, perspective_lens=lens))
    apply_camera_view(ursina=ursina, view=CameraView.HELICOPTER, target=Mock(getPos=Mock(return_value=(0.0, 0.0, 0.0))))
    assert lens.getNear() > 1.0


def test_both_split_follow_lenses_reset_a_copied_distant_near_plane() -> None:
    core = cast(Any, import_module("panda3d.core"))
    original = core.PerspectiveLens()
    original.setNearFar(5.0, 10000.0)
    lenses = (original, original.makeCopy())
    ursina = SimpleNamespace(scene=object(), Vec3=core.Vec3)
    target = Mock(getPos=Mock(return_value=(0.0, 0.0, 0.0)), getH=Mock(return_value=0.0))
    for lens in lenses:
        apply_follow_camera_view(
            ursina=ursina, camera=Mock(), lens=lens, target=target, follow_settings=FORMULA_FOLLOW_CAMERA_SETTINGS
        )
        assert lens.getNear() == pytest.approx(0.1)
