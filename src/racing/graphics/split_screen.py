"""Two scene cameras sharing one window and its existing HUD layers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast


class SplitScreenCameras:
    """Split the scene into two views while leaving every UI region intact."""

    def __init__(self, *, ursina: Any) -> None:
        self._ursina = ursina
        self._window = ursina.application.base.win
        self._main_region = ursina.camera.display_region
        self._main_lens = ursina.camera.perspective_lens
        self._original_dimensions = (
            float(self._main_region.getLeft()),
            float(self._main_region.getRight()),
            float(self._main_region.getBottom()),
            float(self._main_region.getTop()),
        )
        self._original_main_clear_color = self._main_region.getClearColor()
        self._original_main_clear_color_active = self._main_region.getClearColorActive()
        self._original_main_clear_depth_active = self._main_region.getClearDepthActive()
        self.active = False
        self._closed = False

        core = cast(Any, import_module("panda3d.core"))
        main_camera = ursina.application.base.cam
        self.right_transform = ursina.Entity(
            parent=ursina.scene,
            name="head-to-head-right-camera-transform",
            add_to_scene_entities=False,
        )
        self.right_lens = self._main_lens.makeCopy()
        camera_node = core.Camera("head-to-head-right-camera")
        camera_node.setLens(self.right_lens)
        main_scene = main_camera.node().getScene()
        # Ursina installs lights on render, above scene; both cameras must
        # traverse the same root to inherit that lighting state.
        camera_node.setScene(main_camera.getTop() if main_scene.isEmpty() else main_scene)
        camera_node.setCameraMask(main_camera.node().getCameraMask())
        camera_node.setInitialState(main_camera.node().getInitialState())
        self.right_camera = self.right_transform.attachNewNode(camera_node)
        self.right_camera.setTransform(main_camera.getTransform(ursina.camera))

        left, right, bottom, top = self._original_dimensions
        self.right_region = self._window.makeDisplayRegion((left + right) / 2.0, right, bottom, top)
        self.right_region.setSort(self._main_region.getSort())
        self.right_region.setCamera(self.right_camera)
        self.right_region.setClearDepthActive(True)
        self.right_region.setClearColorActive(True)
        self.right_region.setClearColor(self._window.getClearColor())
        self.right_region.setActive(False)
        # Window clears may be disabled by the existing rendering pipeline.
        # Each scene region must clear its own pixels, including the restored
        # full-width primary region after leaving split view.
        self._main_region.setClearColor(self._window.getClearColor())
        self._main_region.setClearColorActive(True)
        self._main_region.setClearDepthActive(True)

    def set_active(self, active: bool) -> None:
        """Enable the two scene regions or restore the original single view."""
        if self._closed:
            return
        if active != self.active:
            left, right, bottom, top = self._original_dimensions
            if active:
                self._main_region.setDimensions(left, (left + right) / 2.0, bottom, top)
            else:
                self._main_region.setDimensions(left, right, bottom, top)
            self.right_region.setActive(active)
            self.active = active
        self.update_aspect_ratio()

    def update_aspect_ratio(self) -> None:
        """Apply the viewport aspect after resize events update Panda's main lens."""
        if self._closed:
            return
        width = int(self._window.getXSize())
        height = int(self._window.getYSize())
        if width <= 0 or height <= 0:
            return
        left, right, bottom, top = self._original_dimensions
        viewport_width = width * (right - left) / (2.0 if self.active else 1.0)
        viewport_height = height * (top - bottom)
        if viewport_width <= 0.0 or viewport_height <= 0.0:
            return
        aspect_ratio = viewport_width / viewport_height
        self._main_lens.setAspectRatio(aspect_ratio)
        self.right_lens.setAspectRatio(aspect_ratio)

    def cleanup(self) -> None:
        """Restore the main region and release the secondary camera."""
        if self._closed:
            return
        self.set_active(False)
        self._main_region.setClearColor(self._original_main_clear_color)
        self._main_region.setClearColorActive(self._original_main_clear_color_active)
        self._main_region.setClearDepthActive(self._original_main_clear_depth_active)
        self._window.removeDisplayRegion(self.right_region)
        self._ursina.destroy(self.right_transform)
        self._closed = True
