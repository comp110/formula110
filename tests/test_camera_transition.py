from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from types import SimpleNamespace
from typing import Any, cast

import pytest

from racing.game.config import CameraView
from racing.graphics.camera import CameraRig, select_camera_view_from_key
from racing.graphics.camera_transition import CameraViewTransition, blend_orientation


@pytest.mark.parametrize(("key", "view"), [
    ("q", CameraView.CINEMATIC), ("w", CameraView.TOP_DOWN), ("e", CameraView.THREE_QUARTER),
    ("r", CameraView.HELICOPTER), ("t", CameraView.DRONE), ("y", CameraView.SPLIT_FOLLOW),
    ("u", CameraView.FOLLOW), ("a", CameraView.LEADERS),
])
def test_requested_shortcuts_preserve_selection(key: str, view: CameraView) -> None:
    rig = CameraRig(view=CameraView.FOLLOW, selected_car_id="heat-2:0")
    select_camera_view_from_key(rig, key, include_split=True)
    assert rig.view is view
    assert rig.selected_car_id == "heat-2:0"


@dataclass
class _Region:
    dimensions: tuple[float, float, float, float] = (0, 1, 0, 1)
    active: bool = True

    def setDimensions(self, left: float, right: float, bottom: float, top: float) -> None:
        self.dimensions = left, right, bottom, top

    def getLeft(self) -> float:
        return self.dimensions[0]

    def getRight(self) -> float:
        return self.dimensions[1]

    def getBottom(self) -> float:
        return self.dimensions[2]

    def getTop(self) -> float:
        return self.dimensions[3]

    def setActive(self, value: bool) -> None:
        self.active = value

    def isActive(self) -> bool:
        return self.active


def _fixture() -> SimpleNamespace:
    core = cast(Any, import_module("panda3d.core"))
    scene = core.NodePath("scene")
    camera = type("SceneCamera", (core.NodePath,), {})("camera-transform")
    camera.reparentTo(scene)
    camera.setPos(0, 0, -50)
    camera.display_region = _Region()
    lens = core.OrthographicLens()
    lens.setCoordinateSystem(core.CSYupRight)
    lens.setFilmSize(100, 60)
    node = camera.attachNewNode(core.Camera("camera"))
    node.node().setLens(lens)
    target = scene.attachNewNode("car")
    window = SimpleNamespace(getXSize=lambda: 1600, getYSize=lambda: 960)
    ursina = SimpleNamespace(scene=scene, camera=camera,
                             application=SimpleNamespace(base=SimpleNamespace(cam=node, win=window)))
    transition = CameraViewTransition(ursina)
    transition.begin(CameraView.TOP_DOWN)
    transition.apply(0, (target,))
    perspective = core.PerspectiveLens()
    perspective.setCoordinateSystem(core.CSYupRight)
    perspective.setNearFar(0.1, 1000)
    perspective.setFov(64)
    perspective.setAspectRatio(5 / 3)
    return SimpleNamespace(core=core, scene=scene, camera=camera, node=node, target=target,
                           transition=transition, perspective=perspective)


def _destination(fixture: SimpleNamespace, x: float = 20) -> None:
    fixture.camera.setPos(x, 0, -10)
    fixture.node.node().setLens(fixture.perspective)


def test_directed_transition_starts_continuously_and_finishes_exactly_at_three_seconds() -> None:
    f = _fixture()
    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f)
    f.transition.apply(0, (f.target,))
    assert tuple(f.camera.getPos()) == pytest.approx((0, 0, -50))
    assert f.transition.active

    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f)
    f.transition.apply(1.5, (f.target,))
    assert tuple(f.camera.getPos()) == pytest.approx((10, 0, -30))
    matrix = f.node.node().getLens().getProjectionMat()
    assert all(isfinite(matrix.getCell(r, c)) for r in range(4) for c in range(4))
    assert f.core.Mat4(matrix).invertInPlace()

    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f, x=24)  # Destination keeps tracking a moving car.
    f.transition.apply(1.5, (f.target,))
    assert not f.transition.active
    assert tuple(f.camera.getPos()) == pytest.approx((24, 0, -10))
    assert f.node.node().getLens() == f.perspective


def test_rapid_switch_starts_from_displayed_pose_and_same_view_does_not_restart() -> None:
    f = _fixture()
    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f)
    f.transition.apply(0.2, (f.target,))
    displayed = tuple(f.camera.getPos())
    f.transition.begin(CameraView.FOLLOW, duration_seconds=3.0)
    _destination(f, 60)
    f.transition.apply(0, (f.target,))
    assert tuple(f.camera.getPos()) == pytest.approx(displayed)
    for _ in range(180):
        f.transition.begin(CameraView.FOLLOW, duration_seconds=3.0)
        _destination(f, 60)
        f.transition.apply(1 / 60, (f.target,))
    assert not f.transition.active
    assert tuple(f.camera.getPos()) == pytest.approx((60, 0, -10))


def test_split_regions_cut_open_and_closed_immediately() -> None:
    f = _fixture()
    right_transform = f.scene.attachNewNode("right-transform")
    right_transform.setPos(30, 0, -10)
    right_camera = right_transform.attachNewNode(f.core.Camera("right-camera"))
    right_camera.node().setLens(f.perspective.makeCopy())
    right_region = _Region((0.5, 1, 0, 1))
    split = SimpleNamespace(right_transform=right_transform, right_camera=right_camera, right_region=right_region)
    f.transition.begin(CameraView.SPLIT_FOLLOW, split=True)
    _destination(f)
    f.camera.display_region.setDimensions(0, 0.5, 0, 1)
    f.perspective.setAspectRatio(5 / 6)
    right_camera.node().getLens().setAspectRatio(5 / 6)
    f.transition.apply(0, (f.target, f.target), split)
    assert f.camera.display_region.getRight() == 0.5
    assert right_region.getLeft() == 0.5
    assert not f.transition.active
    f.transition.begin(CameraView.DRONE)
    _destination(f)
    f.camera.display_region.setDimensions(0, 1, 0, 1)
    right_region.setActive(False)
    f.transition.apply(0, (f.target,), split)
    assert f.camera.display_region.getRight() == 1
    assert not right_region.isActive()
    assert not f.transition.active
    assert f.node.node().getLens() == f.perspective


def test_reset_clears_animation_and_restores_real_lens() -> None:
    f = _fixture()
    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f)
    f.transition.apply(0.1, (f.target,))
    f.transition.reset()
    assert not f.transition.active
    assert f.node.node().getLens() == f.perspective
    f.transition.begin(CameraView.TOP_DOWN)
    f.transition.apply(0, (f.target,))
    assert not f.transition.active


def test_orientation_handles_opposite_quaternion_signs() -> None:
    assert blend_orientation((1, 0, 0, 0), (-1, 0, 0, 0), 0.5) == pytest.approx((1, 0, 0, 0))
    q = blend_orientation((0.01, 0, 0.99995, 0), (-0.01, 0, 0.99995, 0), 0.5)
    assert q == pytest.approx((0, 0, 1, 0))


def test_manual_view_switch_cuts_immediately_and_cancels_directed_transition() -> None:
    f = _fixture()
    f.transition.begin(CameraView.DRONE, duration_seconds=3.0)
    _destination(f)
    f.transition.apply(1, (f.target,))
    assert f.transition.active
    f.transition.begin(CameraView.FOLLOW)
    _destination(f, 60)
    f.transition.apply(0, (f.target,))
    assert not f.transition.active
    assert f.transition.progress == 1
    assert tuple(f.camera.getPos()) == pytest.approx((60, 0, -10))
    assert f.node.node().getLens() == f.perspective
