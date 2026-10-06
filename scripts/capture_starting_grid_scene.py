"""Capture and verify the input-gated lineup and cinematic countdown."""

from __future__ import annotations

import argparse
from dataclasses import replace
from importlib import import_module
from math import isclose
from pathlib import Path
from typing import Any, cast

from racing.game.app import create_head_to_head_viewer_app, create_heat_viewer_app
from racing.game.config import (
    CameraView,
    HeadToHeadViewerConfig,
    HeatViewerConfig,
    RacingAudioConfig,
    parse_window_size,
)
from racing.graphics.timing_tower import TimingTowerRow, format_timing_gap
from racing.physics import FORMULA_VEHICLE_PHYSICS_CONFIG
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatEntrant
from racing.race.progress import track_pose_at_distance
from racing.race.runtime import reset_robot_vehicle, seeded_race_start_finish_pose
from racing.race.timing import TimingSample
from racing.student.api import RobotCommand, RobotSensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/starting-grid"))
    parser.add_argument("--size", type=parse_window_size, default=(1280, 720))
    parser.add_argument("--h2h", action="store_true")
    parser.add_argument("--lap-results", action="store_true", help="verify recorded lap finishes and DNFs")
    parser.add_argument("--split-check", action="store_true", help="verify focus/trailing split cameras and selection")
    parser.add_argument("--camera-cuts", "--camera-transitions", dest="camera_cuts", action="store_true",
                        help="verify immediate cuts between all camera views")
    parser.add_argument("--leaders-check", action="store_true", help="verify battle framing and finishing camera")
    parser.add_argument("--audio-check", action="store_true", help="verify M toggles audio without an on-screen label")
    parser.add_argument("--track", default="mugello-short")
    args = parser.parse_args()
    if args.h2h and (args.lap_results or args.leaders_check):
        parser.error("lap finishing checks require a heat")
    args.output.mkdir(parents=True, exist_ok=True)
    calls: list[RobotSensors] = []

    def controller(sensors: RobotSensors) -> RobotCommand:
        calls.append(sensors)
        return RobotCommand(throttle=0.0 if args.camera_cuts or args.split_check or args.leaders_check else 0.5)

    names = ("Ada L. & Grace H.", "Jean-Luc P.", "Élodie D. & Alex M.", "Sam R.", "Taylor B. & Jordan C.",
             "Morgan T.", "Jamie K. & Casey W.", "Robin S.", "Christopher A. & Alexandria B.", "Avery J.")
    if args.leaders_check:
        names = names[:3]
    if args.audio_check:
        # Exercise the real audio runtime and mute state without playing sound during capture.
        cast(Any, import_module("panda3d.core")).loadPrcFileData("", "audio-library-name null")
    common: dict[str, Any] = dict(
        title="Race Night — Grand Final", size=args.size, starting_grid=True, camera_view=CameraView.CINEMATIC,
        round_seconds=0.1, race_count=2, window_type="offscreen", vsync=False,
        audio=RacingAudioConfig(enabled=args.audio_check, muted=True, music_enabled=False), track_id=args.track,
    )
    if args.lap_results:
        common.update(round_laps=1, race_count=1)
    if args.split_check:
        common.update(camera_view=CameraView.SPLIT_FOLLOW, round_seconds=30.0, race_count=1)
    if args.leaders_check:
        common.update(round_laps=2, race_count=1, camera_view=CameraView.LEADERS)
    if args.camera_cuts:
        common.update(round_seconds=60.0, race_count=1)
    app = cast(Any, create_head_to_head_viewer_app(HeadToHeadViewerConfig(
        **common, grid_names=names[:2], challenger_name="Blue Lightning", incumbent_name="Apex Hunters",
        challenger_controller=controller, incumbent_controller=controller,
    )) if args.h2h else create_heat_viewer_app(HeatViewerConfig(
        **common, grid_names=names,
        entrants=tuple(
            HeatEntrant(f"Car {index + 1}", controller, color)
            for index, color in enumerate(DEFAULT_HEAT_COLORS[:len(names)])
        ),
    )))
    try:
        ursina = cast(Any, import_module("ursina"))
        for _ in range(3):
            app.step()
        loop = next(entity for entity in ursina.scene.entities if entity.name == "head_to_head_viewer_loop")

        def advance(seconds: float) -> None:
            for _ in range(round(seconds * 60)):
                ursina.time.dt = 1 / 60
                loop.update()

        def capture(name: str) -> None:
            assert app.aspect2d.find("**/audio-hud-label").isEmpty()
            assert app.aspect2d.find("**/audio-hud-background").isEmpty()
            assert app.aspect2d.find("**/head-to-head-split-left").isEmpty()
            assert app.aspect2d.find("**/head-to-head-split-right").isEmpty()
            for _ in range(3):
                app.graphicsEngine.renderFrame()
            path = args.output / f"{name}.png"
            assert app.screenshot(namePrefix=str(path.resolve()), defaultFilename=False) is not None
            print(path)

        if args.audio_check:
            audio = app.racing_audio_runtime
            assert audio.enabled and audio.muted
            ursina.held_keys["m"] = 1
            advance(2 / 60)
            assert not audio.muted  # Holding M must toggle only once.
            ursina.held_keys["m"] = 0
            advance(1 / 60)
            ursina.held_keys["m"] = 1
            advance(1 / 60)
            assert audio.muted
            ursina.held_keys["m"] = 0
            advance(1 / 60)
            print("Verified M still toggles enabled audio once per press without any audio overlay.")

        def verify_split_cards() -> None:
            ranks = {row.car_id: row.rank for row in app.racing_timing.rows}
            aspect = args.size[0] / args.size[1]
            for side, markers, car_id in zip(
                (-1, 1), app.racing_split_position_markers, app.racing_split_target_ids, strict=True,
            ):
                assert set(markers.positions) == {car_id}, (side, car_id, markers.positions)
                widgets = markers._markers[car_id]
                assert widgets.rank.node().getText() == f"P{ranks[car_id]}"
                assert not widgets.pointer.isHidden()
                assert isclose(widgets.pointer.getColor()[3], 0.36, abs_tol=0.003)
                core = cast(Any, import_module("panda3d.core"))
                reader = core.GeomVertexReader(widgets.vertices, "vertex")
                for _ in range(3):
                    point = reader.getData3f()
                    assert 0 <= side * point[0] <= aspect
                    assert -1 <= point[1] <= 1

        def capture_results() -> None:
            assert hasattr(app, "racing_result")
            count = len(calls)
            position = tuple(ursina.camera.position)
            overlay = app.aspect2d.find("**/starting-grid")
            assert overlay.getColorScale()[3] == 0
            capture("08-results-start")
            advance(0.75)
            assert isclose(overlay.getColorScale()[3], 0.5, abs_tol=1e-6)
            capture("09-results-fade")
            advance(1.0)
            assert overlay.getColorScale()[3] == 1
            assert len(calls) == count
            assert tuple(ursina.camera.position) == position
            for key in ("space", "left mouse down", "1", "v"):
                loop.input(key)
            advance(1.0)
            assert len(calls) == count
            assert tuple(ursina.camera.position) == position
            capture("10-results")
            texts = [path.node().getText() for path in overlay.findAllMatches("**/+TextNode")]
            assert "Race Night — Grand Final" in texts
            assert any("RESULTS" in value for value in texts)
            assert all(name in texts for name in (names[:2] if args.h2h else names))
            if args.lap_results:
                assert "1:23.125 · WINNER" in texts
                assert "1:23.819 · +0.694s" in texts
                assert sum(value.startswith("DNF") for value in texts) == 2
            print("Verified results fade, participant names, recorded timings, and frozen simulation/camera.")

        if args.leaders_check:
            loop.input("space")
            advance(9.3)
            race = app.racing_lap_race
            model = app.racing_track.model
            line = seeded_race_start_finish_pose(model=model, config=FORMULA_VEHICLE_PHYSICS_CONFIG,
                                                 random_seed=110, race_index=1)
            ids = tuple(entry.car_id for entry in app.racing_entries)
            length = model.total_length_m

            def stage(distances: tuple[float, ...]) -> None:
                previous = {row.car_id: row.distance_m for row in app.racing_timing.rows}
                for car_id, runtime, distance in zip(ids, app.racing_runtimes, distances, strict=True):
                    pose = track_pose_at_distance(model, line.progress_distance_m + distance)
                    runtime.tracker.unwrapped_progress_distance_m += distance - previous[car_id]
                    runtime.tracker.last_progress_distance_m = pose.progress_distance_m
                    reset_robot_vehicle(runtime.robot, position=(pose.position.x, 0.6, pose.position.z),
                                        heading_degrees=pose.heading_degrees)
                advance(1 / 60)

            stage((length + 2, length - 12, length - 24))
            assert not race.finish_times  # Crossing an earlier lap must not lock the camera.
            assert app.racing_camera_rig.leaders.finish_pose is None
            stage((2 * length - 2, 2 * length - 14, 2 * length - 26))
            race.rows = tuple(replace(row, gap_seconds=(0.0, 1.0, 2.9)[i]) for i, row in enumerate(race.rows))
            ursina.time.dt = 0
            loop.update()
            assert len(app.racing_camera_rig.leaders.subject_ids) == 3
            capture("leaders-01-three-cars")
            race.rows = tuple(replace(row, gap_seconds=(0.0, 1.0, 3.1)[i]) for i, row in enumerate(race.rows))
            loop.update()
            assert len(app.racing_camera_rig.leaders.subject_ids) == 2
            capture("leaders-02-two-cars")
            stage((2 * length + 2, 2 * length - 14, 2 * length - 26))
            assert len(race.finish_times) == 1
            frozen = (tuple(ursina.camera.position), tuple(ursina.camera.rotation), ursina.camera.fov)
            capture("leaders-03-winner-finish")
            stage((2 * length + 45, 2 * length - 10, 2 * length - 26))
            advance(0.5)
            assert tuple(ursina.camera.position) == frozen[0]
            assert all(isclose(a, b, abs_tol=1e-4) for a, b in zip(frozen[1], ursina.camera.rotation, strict=True))
            assert ursina.camera.fov == frozen[2]
            assert not app.racing_camera_transition.active
            app.racing_runtimes[2].robot.eliminated = True
            stage((2 * length + 45, 2 * length + 2, 2 * length - 26))
            assert len(race.finish_times) == 2 and race.complete
            assert not hasattr(app, "racing_result")
            frozen_time = race.elapsed_seconds
            call_count = len(calls)
            advance(59 / 60)
            assert not app.racing_camera_transition.active
            assert tuple(ursina.camera.position) == frozen[0]
            assert all(isclose(a, b, abs_tol=1e-4) for a, b in zip(frozen[1], ursina.camera.rotation, strict=True))
            assert ursina.camera.fov == frozen[2]
            capture("leaders-04-second-finish-hold")
            advance(1 / 60)
            assert app.racing_camera_transition.active
            assert app.racing_camera_transition.progress == 0
            capture("leaders-05-pullback-start")
            advance(1.5)
            assert isclose(app.racing_camera_transition.progress, 0.5)
            assert not hasattr(app, "racing_result")
            capture("leaders-06-pullback-mid")
            advance(1.5)
            assert not app.racing_camera_transition.active
            assert ursina.camera.orthographic
            assert abs(float(ursina.camera.getP()) + 90) < 0.001
            assert len(calls) == call_count and race.elapsed_seconds == frozen_time
            assert hasattr(app, "racing_result")
            capture("leaders-07-top-down")
            advance(1.5)
            capture("leaders-08-results")
            print("Verified three/two-car framing, final-lap winner lock, stationary finish shot, "
                  "one-second P2 hold, three-second top-down pullback, stopped physics, and delayed results.")
            return

        if args.camera_cuts:
            loop.input("space")
            advance(9.3)
            transition = app.racing_camera_transition
            for key, view in (
                ("w", CameraView.TOP_DOWN), ("e", CameraView.THREE_QUARTER), ("r", CameraView.HELICOPTER),
                ("t", CameraView.DRONE), ("y", CameraView.SPLIT_FOLLOW), ("u", CameraView.FOLLOW),
                ("q", CameraView.CINEMATIC), ("a", CameraView.LEADERS),
            ):
                loop.input(key)
                ursina.time.dt = 0
                loop.update()
                assert app.racing_camera_rig.view is view
                assert not transition.active and transition.progress == 1
                assert type(app.cam.node().getLens()).__name__ != "MatrixLens"
                assert ursina.camera.display_region.getRight() == (0.5 if key == "y" else 1.0)
                capture(f"cut-{key}")
                before = tuple(ursina.camera.position)
                loop.input(key)
                loop.update()
                assert tuple(ursina.camera.position) == before
            print("Verified immediate cuts for all eight shortcuts, native lenses, split panes, and repeated keys.")
            return

        if args.split_check:
            tower = app.racing_timing_tower
            loop.input("space")
            advance(9.3)
            order = [row.car_id for row in app.racing_timing.rows if not row.eliminated]
            split = app.racing_split_cameras
            assert split.active
            assert app.racing_split_target_ids == tuple(order[:2])
            verify_split_cards()
            capture("split-01-auto")
            focus_rank = 2 if args.h2h else 3
            loop.input(str(focus_rank))
            advance(1.0)
            focus_id = order[focus_rank - 1]
            competitor_id = order[0] if args.h2h else order[focus_rank]
            assert app.racing_camera_rig.view is CameraView.SPLIT_FOLLOW
            assert app.racing_split_target_ids == (focus_id, competitor_id)
            verify_split_cards()
            capture("split-02-selected")
            tower.row_buttons[focus_id]["command"]()
            advance(1 / 60)
            assert app.racing_camera_rig.view is CameraView.SPLIT_FOLLOW
            assert app.racing_split_target_ids == (focus_id, competitor_id)
            tower.row_buttons[order[-1]]["command"]()
            advance(1.0)
            assert app.racing_split_target_ids == (order[-1], order[-2])
            verify_split_cards()
            capture("split-03-last-place")
            loop.input("w")
            advance(1 / 60)
            assert not split.active
            assert ursina.camera.display_region.getRight() == 1.0
            loop.input("y")
            advance(1 / 60)
            assert split.active
            assert app.racing_split_target_ids[0] == order[-1]
            # V must include split view in heats as well as h2h.
            loop.input("u")
            advance(1 / 60)
            ursina.held_keys["v"] = 1
            advance(1 / 60)
            ursina.held_keys["v"] = 0
            advance(1 / 60)
            assert app.racing_camera_rig.view is CameraView.SPLIT_FOLLOW
            tower.auto_button["command"]()
            advance(1.0)
            assert app.racing_split_target_ids == tuple(order[:2])
            verify_split_cards()
            assert isclose(split.right_lens.getAspectRatio(), args.size[0] / args.size[1] / 2, abs_tol=1e-6)
            capture("split-04-restored")
            for entry, runtime in zip(app.racing_entries, app.racing_runtimes, strict=True):
                if f"{entry.role}:{entry.copy_index}" != order[0]:
                    runtime.robot.eliminated = True
            advance(1 / 60)
            assert app.racing_split_target_ids == (order[0], None)
            assert not split.active
            assert ursina.camera.display_region.getRight() == 1.0
            assert not app.racing_split_position_markers[1].positions
            capture("split-05-one-car")
            print("Verified focus-only place cards/pointers in each pane, split targets, repeat clicks, "
                  "last-place fallback, Y/V controls, and solo view.")
            return

        advance(10.0)
        assert not calls
        assert app.racing_start_sequence.elapsed_seconds == 0.0
        capture("01-lineup")
        loop.input("space")
        capture("02-overhead")
        opening_position = tuple(ursina.camera.position)
        opening_rotation = tuple(ursina.camera.rotation)
        advance(1.0)
        assert app.racing_start_sequence.holding_overview
        assert not app.racing_start_sequence.visible
        assert not calls
        assert tuple(ursina.camera.position) == opening_position
        assert tuple(ursina.camera.rotation) == opening_rotation
        capture("02a-overhead-hold")
        advance(1.0)
        assert not app.racing_start_sequence.holding_overview
        assert app.racing_start_sequence.light_count == 0
        assert tuple(ursina.camera.position) == opening_position
        capture("02b-countdown-start")
        advance(1.5)
        capture("03-transition")
        advance(1.5)
        assert app.racing_start_sequence.light_count == 3
        assert not calls
        pose = app.racing_camera_rig.cinematic.pose
        assert (ursina.camera.position - ursina.Vec3(*pose.position)).length() < 0.001
        capture("04-three-lights")
        advance(3.5)
        assert app.racing_start_sequence.racing
        assert not calls
        capture("05-lights-out")
        if args.lap_results:
            # Feed deterministic crossings through the actual LapRace model.
            # Finish order deliberately differs from the starting grid.
            race = app.racing_lap_race
            ids = [entry.car_id for entry in app.racing_entries]
            finishes = (85.0, 83.125, 83.819, 85.340, 86.011, 86.200, 87.123, 89.005)
            for elapsed in (80.0, 90.0):
                race.update(elapsed_seconds=elapsed, samples=tuple(
                    TimingSample(car_id, race.track_length_m + (elapsed - finishes[index]) * 5)
                    if index < len(finishes) else TimingSample(car_id, 0, eliminated=True)
                    for index, car_id in enumerate(ids)
                ))
            assert race.complete
            advance(1 / 60)
            capture_results()
            return
        advance(0.25)
        assert calls
        assert app.racing_start_sequence.waiting_for_start
        capture("06-next-round")
        count = len(calls)
        advance(1.0)
        assert len(calls) == count
        loop.input("left mouse down")
        assert not app.racing_start_sequence.waiting_for_start
        advance(0.5)
        assert isclose(app.racing_start_sequence.elapsed_seconds, 0.5)
        assert app.racing_start_sequence.holding_overview
        assert len(calls) == count
        # Render realistic live standings and verify the compact columns with the actual font.
        tower = app.racing_timing_tower
        car_ids = tuple(tower.row_buttons)
        abbreviations = ("JB", "JS+MH", "NN+LB", "MC", "KK", "YY", "SM", "SS", "WW+WW", "JH+BH")
        rows = tuple(
            TimingTowerRow(
                car_id, abbreviations[index], app.racing_display_colors[car_id], index + 1,
                None if index == 0 else (100.694 if index == 8 else index * 0.173),
            )
            for index, car_id in enumerate(car_ids)
        )
        tower.row_buttons[car_ids[1]]["command"]()
        assert app.racing_camera_rig.selected_car_id == car_ids[1]
        tower.update(rows, 30.0, car_ids[1], clock_text="LAP 3/10")
        for row in rows:
            texts = {
                path.node().getText(): path for path in tower.row_buttons[row.car_id].findAllMatches("**/+TextNode")
            }
            name, gap = texts[row.name], texts[format_timing_gap(row)]
            name_right = name.getX() + name.node().calcWidth(row.name) * name.getSx()
            gap_left = gap.getX() - gap.node().calcWidth(format_timing_gap(row)) * gap.getSx()
            assert name_right + 0.018 <= gap_left + 1e-6
        capture("07-timing-tower")
        tower.toggle_button["command"]()
        assert not tower.visible
        tower.toggle_button["command"]()
        assert tower.visible
        tower.auto_button["command"]()
        assert app.racing_camera_rig.selected_car_id is None
        print(
            "Verified both start inputs, two-second overhead hold, paused simulation, camera handoff, and round reset."
        )
        print("Verified full car abbreviations, gap spacing, row selection, clickable logo, and automatic targeting.")
        for _ in range(600):
            if hasattr(app, "racing_result"):
                break
            advance(1 / 60)
        capture_results()
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
