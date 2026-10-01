"""One-second transitions between scene cameras and viewport layouts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from math import sqrt
from typing import Any, cast

from racing.game.config import CameraView
from racing.graphics.split_screen import SplitScreenCameras

CAMERA_TRANSITION_SECONDS = 1.0


def blend_orientation(
    start: tuple[float, ...], end: tuple[float, ...], amount: float,
) -> tuple[float, ...]:
    """Interpolate normalized quaternions along the shorter rotation."""
    if sum(a * b for a, b in zip(start, end, strict=True)) < 0:
        end = tuple(-value for value in end)
    values = tuple(a + (b - a) * amount for a, b in zip(start, end, strict=True))
    length = sqrt(sum(value * value for value in values))
    return tuple(value / length for value in values)


@dataclass(frozen=True, slots=True)
class _Frame:
    position: tuple[float, ...]
    orientation: tuple[float, ...]
    projection: tuple[float, ...]
    dimensions: tuple[float, float, float, float]
    aspect: float
    active: bool
    lens: Any


@dataclass(slots=True)
class _Pane:
    transform: Any
    camera: Any
    region: Any
    blend_lens: Any


class CameraViewTransition:
    """Blend rendered poses without feeding them back into camera directors.

    Projection matrices are normalized at the subject before interpolation,
    allowing perspective and orthographic views to zoom continuously. Split
    regions expand/contract with the same easing as the camera movement.
    """

    def __init__(self, ursina: Any) -> None:
        self._ursina = ursina
        self._core = cast(Any, import_module("panda3d.core"))
        self._panes = {"main": _Pane(
            ursina.camera, ursina.application.base.cam, ursina.camera.display_region, self._core.MatrixLens(),
        )}
        self._key: tuple[CameraView, bool] | None = None
        self._elapsed = CAMERA_TRANSITION_SECONDS
        self._sources: dict[str, _Frame] = {}
        self._destinations: dict[str, _Frame] = {}
        self._displayed: dict[str, _Frame] = {}

    @property
    def active(self) -> bool:
        return self._elapsed < CAMERA_TRANSITION_SECONDS

    @property
    def progress(self) -> float:
        return self._elapsed / CAMERA_TRANSITION_SECONDS

    def begin(self, view: CameraView, *, split: bool = False) -> None:
        """Call before calculating this frame's destination camera poses."""
        key = (view, split)
        if self._key is not None and key != self._key and self._displayed:
            # A rapid second switch starts from the currently displayed pose.
            self._sources = self._displayed.copy()
            self._elapsed = 0.0
        self._key = key
        self._restore_destination_lenses()

    def reset(self) -> None:
        """Skip animation when resetting a race behind its starting screen."""
        self._restore_destination_lenses()
        self._key = None
        self._elapsed = CAMERA_TRANSITION_SECONDS
        self._sources.clear()
        self._destinations.clear()
        self._displayed.clear()

    def apply(
        self, delta_seconds: float, targets: tuple[Any, ...], split_cameras: SplitScreenCameras | None = None,
    ) -> None:
        """Call after calculating poses, before projecting scene labels."""
        if split_cameras is not None and "right" not in self._panes:
            self._panes["right"] = _Pane(
                split_cameras.right_transform, split_cameras.right_camera,
                split_cameras.right_region, self._core.MatrixLens(),
            )
        targets_by_pane = {"main": targets[0], "right": targets[-1]}
        self._destinations = {
            name: self._capture(pane, targets_by_pane[name]) for name, pane in self._panes.items()
        }
        if self.active:
            self._elapsed = min(CAMERA_TRANSITION_SECONDS, self._elapsed + max(0.0, delta_seconds))
            if self._elapsed >= CAMERA_TRANSITION_SECONDS - 1e-9:
                self._elapsed = CAMERA_TRANSITION_SECONDS
            amount = self.progress**2 * (3 - 2 * self.progress)
            if self.active:
                for name, pane in self._panes.items():
                    end = self._destinations[name]
                    start = self._sources.get(name)
                    if name == "right":
                        if start is None or not start.active:
                            start = self._collapsed_right(self._sources["main"])
                        if not end.active:
                            end = self._collapsed_right(self._destinations["main"])
                    assert start is not None
                    self._blend(pane, start, end, amount)
        self._displayed = {
            name: self._capture(pane, targets_by_pane[name]) for name, pane in self._panes.items()
        }

    @staticmethod
    def _collapsed_right(frame: _Frame) -> _Frame:
        _, right, bottom, top = frame.dimensions
        return replace(frame, dimensions=(right, right, bottom, top), active=False)

    def _restore_destination_lenses(self) -> None:
        for name, frame in self._destinations.items():
            pane = self._panes[name]
            pane.camera.node().setLens(frame.lens)
            pane.region.setDimensions(*frame.dimensions)
            pane.region.setActive(frame.active)

    def _capture(self, pane: _Pane, target: Any) -> _Frame:
        lens = pane.camera.node().getLens()
        matrix = lens.getProjectionMat()
        target_position = pane.camera.getRelativePoint(target, self._core.Point3(0, 0, 0))
        projected = matrix.xform(self._core.Vec4(*target_position, 1))
        reference_w = max(1e-6, abs(float(projected[3])))
        dimensions = (float(pane.region.getLeft()), float(pane.region.getRight()),
                      float(pane.region.getBottom()), float(pane.region.getTop()))
        return _Frame(
            tuple(float(value) for value in pane.transform.getPos(self._ursina.scene)),
            tuple(float(value) for value in pane.transform.getQuat(self._ursina.scene)),
            tuple(float(matrix.getCell(row, column)) / reference_w for row in range(4) for column in range(4)),
            dimensions, self._aspect(dimensions),
            bool(pane.region.isActive()), lens,
        )

    def _aspect(self, dimensions: tuple[float, ...]) -> float:
        window = self._ursina.application.base.win
        return max(1e-6, float(window.getXSize()) * (dimensions[1] - dimensions[0])) / max(
            1e-6, float(window.getYSize()) * (dimensions[3] - dimensions[2]),
        )

    def _blend(self, pane: _Pane, start: _Frame, end: _Frame, amount: float) -> None:
        def interpolate(first: tuple[float, ...], last: tuple[float, ...]) -> tuple[float, ...]:
            return tuple(a + (b - a) * amount for a, b in zip(first, last, strict=True))

        position = interpolate(start.position, end.position)
        orientation = blend_orientation(start.orientation, end.orientation, amount)
        pane.transform.setPos(self._ursina.scene, self._core.Vec3(*position))
        pane.transform.setQuat(self._ursina.scene, self._core.Quat(*orientation))
        dimensions = interpolate(start.dimensions, end.dimensions)
        aspect = self._aspect(dimensions)
        # Preserve vertical framing as a pane grows; cropping its horizontal
        # field avoids squeezing cars into a narrow, opening viewport.
        projections = tuple(
            tuple(value * frame.aspect / aspect if index % 4 == 0 else value
                  for index, value in enumerate(frame.projection))
            for frame in (start, end)
        )
        pane.blend_lens.setUserMat(self._core.Mat4(*interpolate(*projections)))
        pane.camera.node().setLens(pane.blend_lens)
        pane.region.setDimensions(*dimensions)
        pane.region.setActive((start.active or end.active) and dimensions[1] - dimensions[0] > 1e-6)
