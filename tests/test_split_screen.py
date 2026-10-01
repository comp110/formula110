from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from racing.game import cli
from racing.game.app import build_scene
from racing.game.config import CameraView, GameConfig, HeadToHeadViewerConfig
from racing.graphics.camera import (
    FORMULA_FOLLOW_CAMERA_SETTINGS,
    CameraRig,
    apply_follow_camera_view,
    next_camera_view,
    update_camera_cycle,
)
from racing.graphics.split_screen import SplitScreenCameras, split_follow_targets
from racing.race.timing import TimingStanding


def test_split_camera_cycles_on_a_new_key_press_and_resets_follow_history() -> None:
    rig = CameraRig(view=CameraView.FOLLOW, follow_target_id=123, follow_direction_initialized=True)
    rig.follow_forward_samples.append((0.1, 1.0, 0.0))

    update_camera_cycle(rig, cycle_key_down=True, include_split=True)

    assert rig.view is CameraView.SPLIT_FOLLOW
    assert rig.follow_target_id is None
    assert not rig.follow_direction_initialized
    assert rig.follow_forward_samples == []
    update_camera_cycle(rig, cycle_key_down=True, include_split=True)
    assert rig.view is CameraView.SPLIT_FOLLOW
    update_camera_cycle(rig, cycle_key_down=False, include_split=True)
    update_camera_cycle(rig, cycle_key_down=True, include_split=True)
    assert rig.view is CameraView.TOP_DOWN
    assert next_camera_view(CameraView.FOLLOW) is CameraView.TOP_DOWN


def test_split_camera_cli_dispatches_to_watched_head_to_head(monkeypatch: pytest.MonkeyPatch) -> None:
    viewer = Mock()
    create_viewer = Mock(return_value=viewer)
    monkeypatch.setattr(cli, "create_head_to_head_viewer_app", create_viewer)

    cli.main(["h2h", "--watch", "--challenger-keyboard", "--incumbent-keyboard", "--camera", "split_follow"])

    config = create_viewer.call_args.args[0]
    assert isinstance(config, HeadToHeadViewerConfig)
    assert config.camera_view is CameraView.SPLIT_FOLLOW
    viewer.run.assert_called_once_with()


def test_split_camera_requires_a_race_viewer() -> None:
    with pytest.raises(SystemExit):
        cli.build_argument_parser().parse_args(["--camera", "split_follow"])
    with pytest.raises(ValueError, match="requires the head-to-head viewer"):
        build_scene(GameConfig(camera_view=CameraView.SPLIT_FOLLOW))


def _standings(*ids: str, out: tuple[str, ...] = ()) -> tuple[TimingStanding, ...]:
    return tuple(TimingStanding(index, car_id, 500 - index * 110, None, eliminated=car_id in out)
                 for index, car_id in enumerate(ids, start=1))


@pytest.mark.parametrize("ids", [("heat-2:0", "heat-0:0", "heat-1:0"),
                                ("incumbent:1", "challenger:2", "incumbent:0")])
def test_split_camera_uses_race_order_and_preserves_selected_copy(ids: tuple[str, str, str]) -> None:
    standings = _standings(*ids)
    assert split_follow_targets(standings) == ids[:2]
    assert split_follow_targets(standings, ids[1]) == ids[1:]


def test_split_focus_stays_with_car_and_trailer_updates_after_overtakes() -> None:
    assert split_follow_targets(_standings("a", "b", "c", "d"), "b") == ("b", "c")
    assert split_follow_targets(_standings("a", "b", "d", "c"), "b") == ("b", "d")
    assert split_follow_targets(_standings("a", "d", "b", "c"), "b") == ("b", "c")


def test_split_skips_retired_and_dnf_competitors_and_falls_back_to_car_ahead() -> None:
    assert split_follow_targets(_standings("a", "b", "c", out=("b",)), "a") == ("a", "c")
    assert split_follow_targets(_standings("a", "b", "c"), "c") == ("c", "b")
    assert split_follow_targets(_standings("a", "b", "c", out=("b",)), "c") == ("c", "a")


def test_split_auto_uses_active_leader_and_handles_one_remaining_car() -> None:
    standings = _standings("a", "b", "c", out=("a",))
    assert split_follow_targets(standings) == ("b", "c")
    assert split_follow_targets(standings, "missing") == ("b", "c")
    assert split_follow_targets(_standings("a", "b", out=("b",))) == ("a", None)
    assert split_follow_targets(_standings("a", "b", out=("a", "b"))) == ("a", None)


def test_split_manual_retired_focus_stays_selected_until_auto() -> None:
    standings = _standings("a", "b", "c", out=("c",))
    assert split_follow_targets(standings, "c") == ("c", "b")
    assert split_follow_targets(standings) == ("a", "b")


def test_follow_cameras_keep_independent_target_history_and_poses() -> None:
    core = cast(Any, import_module("panda3d.core"))
    ursina = SimpleNamespace(scene=object(), Vec3=core.Vec3)
    targets = (
        Mock(
            getPos=Mock(return_value=(10.0, 0.0, 20.0)),
            getQuat=Mock(return_value=Mock(xform=Mock(return_value=(0.0, 0.0, 1.0)))),
        ),
        Mock(
            getPos=Mock(return_value=(100.0, 0.0, 200.0)),
            getQuat=Mock(return_value=Mock(xform=Mock(return_value=(1.0, 0.0, 0.0)))),
        ),
    )
    rigs = (CameraRig(), CameraRig())
    cameras = (Mock(), Mock())
    lenses = (core.PerspectiveLens(), core.PerspectiveLens())
    for _ in range(2):
        for target, rig, camera, lens in zip(targets, rigs, cameras, lenses, strict=True):
            apply_follow_camera_view(
                ursina=ursina,
                camera=camera,
                lens=lens,
                target=target,
                rig=rig,
                delta_seconds=1.0 / 60.0,
                follow_settings=FORMULA_FOLLOW_CAMERA_SETTINGS,
            )

    assert cameras[0].position == pytest.approx((10.0, 1.66, 16.0))
    assert cameras[1].position == pytest.approx((96.0, 1.66, 200.0))
    assert rigs[0].follow_heading_degrees == pytest.approx(0.0)
    assert rigs[1].follow_heading_degrees == pytest.approx(90.0)
    assert rigs[0].follow_target_id == id(targets[0])
    assert rigs[1].follow_target_id == id(targets[1])
    rigs[0].reset_follow_history()
    assert rigs[1].follow_direction_initialized
    assert rigs[1].follow_target_id == id(targets[1])


def _split_camera_fixture(
    *, explicit_scene: bool = False, clear_color_active: bool = False, clear_depth_active: bool = False
) -> SimpleNamespace:
    core = cast(Any, import_module("panda3d.core"))
    render = core.NodePath("render")
    scene = render.attachNewNode("scene")
    camera_type = type("SceneCamera", (core.NodePath,), {})
    camera = camera_type("main-camera-transform")
    camera.reparentTo(scene)
    lens = core.PerspectiveLens()
    lens.setAspectRatio(16 / 9)
    lens.setNearFar(0.1, 500.0)
    camera.perspective_lens = lens
    main_region = Mock(
        getLeft=Mock(return_value=0.0),
        getRight=Mock(return_value=1.0),
        getBottom=Mock(return_value=0.0),
        getTop=Mock(return_value=1.0),
        getSort=Mock(return_value=0),
        getClearColor=Mock(return_value=core.Vec4(0.7, 0.6, 0.5, 1.0)),
        getClearColorActive=Mock(return_value=clear_color_active),
        getClearDepthActive=Mock(return_value=clear_depth_active),
    )
    camera.display_region = main_region
    camera_node = core.Camera("main-camera")
    camera_node.setLens(lens)
    if explicit_scene:
        camera_node.setScene(scene)
    main_camera = camera.attachNewNode(camera_node)
    main_camera.setHpr(180.0, 0.0, 0.0)
    right_region = Mock()
    hud_region = Mock()
    window = Mock(
        getXSize=Mock(return_value=1600),
        getYSize=Mock(return_value=900),
        getClearColor=Mock(return_value=core.Vec4(0.1, 0.2, 0.3, 1.0)),
        makeDisplayRegion=Mock(return_value=right_region),
        getDisplayRegions=Mock(return_value=(main_region, hud_region, right_region)),
    )
    ursina = SimpleNamespace(
        scene=scene,
        camera=camera,
        Entity=Mock(return_value=scene.attachNewNode("right-transform")),
        application=SimpleNamespace(base=SimpleNamespace(win=window, cam=main_camera)),
        destroy=Mock(),
    )
    return SimpleNamespace(
        cameras=SplitScreenCameras(ursina=ursina),
        ursina=ursina,
        window=window,
        main_region=main_region,
        right_region=right_region,
        hud_region=hud_region,
        main_camera=main_camera,
        render=render,
    )


def test_split_regions_resize_restore_and_leave_hud_untouched() -> None:
    fixture = _split_camera_fixture()
    cameras = fixture.cameras
    lens = fixture.ursina.camera.perspective_lens
    fixture.main_region.setDimensions.assert_not_called()
    assert not cameras.active
    assert cameras.right_lens is not lens
    assert cameras.right_lens.getNear() == pytest.approx(lens.getNear())
    assert cameras.right_lens.getFar() == pytest.approx(lens.getFar())
    assert cameras.right_camera.getTransform(cameras.right_transform) == fixture.main_camera.getTransform(
        fixture.ursina.camera
    )

    cameras.set_active(True)
    fixture.main_region.setDimensions.assert_called_once_with(0.0, 0.5, 0.0, 1.0)
    fixture.window.makeDisplayRegion.assert_called_once_with(0.5, 1.0, 0.0, 1.0)
    fixture.right_region.setActive.assert_called_with(True)
    assert lens.getAspectRatio() == pytest.approx(8 / 9)
    assert cameras.right_lens.getAspectRatio() == pytest.approx(8 / 9)

    fixture.window.getXSize.return_value = 1920
    fixture.window.getYSize.return_value = 1080
    lens.setAspectRatio(16 / 9)
    cameras.update_aspect_ratio()
    assert lens.getAspectRatio() == pytest.approx(8 / 9)
    fixture.window.getYSize.return_value = 1200
    cameras.update_aspect_ratio()
    assert lens.getAspectRatio() == pytest.approx(0.8)
    assert cameras.right_lens.getAspectRatio() == pytest.approx(0.8)

    cameras.set_active(False)
    fixture.main_region.setDimensions.assert_called_with(0.0, 1.0, 0.0, 1.0)
    fixture.right_region.setActive.assert_called_with(False)
    assert lens.getAspectRatio() == pytest.approx(1.6)
    assert fixture.hud_region.mock_calls == []
    cameras.cleanup()
    cameras.cleanup()
    fixture.window.removeDisplayRegion.assert_called_once_with(fixture.right_region)
    fixture.ursina.destroy.assert_called_once_with(cameras.right_transform)


@pytest.mark.parametrize("explicit_scene", [False, True])
def test_split_camera_preserves_scene_root_for_inherited_lighting(explicit_scene: bool) -> None:
    fixture = _split_camera_fixture(explicit_scene=explicit_scene)
    expected_scene = fixture.ursina.scene if explicit_scene else fixture.render

    assert fixture.cameras.right_camera.node().getScene() == expected_scene
    fixture.cameras.cleanup()


@pytest.mark.parametrize(("clear_color_active", "clear_depth_active"), [(False, True), (True, False)])
def test_split_regions_clear_each_frame_and_restore_original_clear_state_on_cleanup(
    clear_color_active: bool, clear_depth_active: bool
) -> None:
    fixture = _split_camera_fixture(clear_color_active=clear_color_active, clear_depth_active=clear_depth_active)
    cameras = fixture.cameras
    cameras.set_active(True)

    for region in (fixture.main_region, fixture.right_region):
        region.setClearColorActive.assert_called_with(True)
        region.setClearDepthActive.assert_called_with(True)
        region.setClearColor.assert_called_with(fixture.window.getClearColor.return_value)

    cameras.set_active(False)
    fixture.main_region.setClearColorActive.assert_called_with(True)
    fixture.main_region.setClearDepthActive.assert_called_with(True)
    fixture.main_region.setClearColor.assert_called_with(fixture.window.getClearColor.return_value)

    cameras.cleanup()
    fixture.main_region.setClearColorActive.assert_called_with(clear_color_active)
    fixture.main_region.setClearDepthActive.assert_called_with(clear_depth_active)
    fixture.main_region.setClearColor.assert_called_with(fixture.main_region.getClearColor.return_value)
