"""Build the graphical simulator scenes and connect them to physics and input."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from math import cos, pi, sin
from pathlib import Path
from typing import Any, cast

from racing.controls.gamepad import sync_gamepad_axes
from racing.controls.keyboard import manual_drive_command
from racing.game.config import (
    CameraView,
    CarShowcaseConfig,
    GameConfig,
    HeadToHeadViewerConfig,
    HeatViewerConfig,
    RunnableApp,
    configure_window,
    fps_text_for_delta,
)
from racing.game.recording import HumanGameplayRecorder
from racing.graphics.camera import (
    FORMULA_DRONE_CAMERA_SETTINGS,
    FORMULA_FOLLOW_CAMERA_SETTINGS,
    CameraRig,
    FollowCameraSettings,
    apply_camera_view,
    apply_follow_camera_view,
    update_camera_cycle,
)
from racing.graphics.lighting import add_lighting, add_showcase_lighting
from racing.graphics.panda_config import (
    configure_panda_antialiasing,
    configure_panda_y_up,
    enable_render_antialiasing,
    patch_ursina_window_coordinate_system,
    quiet_panda_image_logs,
)
from racing.graphics.render_assets import create_scene_assets
from racing.graphics.split_screen import SplitScreenCameras
from racing.graphics.timing_tower import TimingTower, TimingTowerRow
from racing.graphics.track_rendering import (
    NIGHT_SKY_COLOR,
    START_HEADING_DEGREES,
    add_racing_scene_collisions,
    add_track,
    add_trackside_scenery,
    add_world_floor,
    set_start_finish_gantry_pose,
    set_start_finish_pose,
    start_finish_render_pose,
)
from racing.graphics.vehicle_visuals import (
    add_robot_visuals,
    add_showcase_floor,
    apply_robot_team_color,
    apply_showcase_camera,
    create_showcase_robot,
    pose_showcase_car,
)
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    RobotVehicle,
    apply_robot_vehicle_command,
    apply_wall_impact_damage,
    create_physics_world,
    create_robot_vehicle,
)
from racing.race.head_to_head import (
    HeadToHeadRaceEntry,
    HeadToHeadRaceResult,
    HeadToHeadResult,
    classify_head_to_head_winner,
    controller_for_copy,
    format_head_to_head_result,
    format_head_to_head_result_banner,
    head_to_head_race_entries,
    head_to_head_race_margin,
    head_to_head_team_stats_from_runtimes,
)
from racing.race.heat import (
    HeatRaceEntry,
    HeatRaceResult,
    HeatResult,
    format_heat_result,
    format_heat_result_banner,
    heat_race_entries,
    heat_race_result_from_runtimes,
    validate_heat_entrants,
)
from racing.race.progress import (
    TrackProgressModel,
    TrackProjection,
    default_track_progress_model,
    project_track_position,
    resolve_track,
)
from racing.race.rules import HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M, HeadToHeadRaceRules
from racing.race.runtime import (
    RaceCarRuntime,
    RaceContactState,
    RaceRecoveryConfig,
    RaceSpawnPose,
    lap_progress_tracker_for_spawn_pose,
    maybe_marshal_race_runtimes,
    race_contact_states,
    race_scored_distance_m,
    race_spawn_poses,
    reset_robot_vehicle,
    robot_track_point,
    seeded_race_start_finish_pose,
    start_finish_pose_for_progress,
    update_race_runtime_after_step,
)
from racing.race.sensors import RobotSensorBuilderState, build_robot_sensors
from racing.race.timing import RaceTiming, TimingSample
from racing.sound.audio import (
    AudioKeyToggleState,
    RacingAudioRuntimeLike,
    create_racing_audio_runtime,
    update_audio_mute_key,
)
from racing.student.api import RobotCommand, RobotController
from racing.track.world import TRACK_ID_MUGELLO_SHORT, TrackPoint

PLAYABLE_MAX_FRAME_DELTA_SECONDS = 0.25
PLAYABLE_MAX_FIXED_STEPS_PER_FRAME = 8
DAMAGE_HUD_MAX_COLUMNS = 4
DAMAGE_HUD_USABLE_WIDTH = 3.30
DAMAGE_HUD_COLUMN_GAP = 0.055
DAMAGE_HUD_MAX_WIDTH = 0.78
DAMAGE_HUD_MIN_WIDTH = 0.42
DAMAGE_HUD_HEIGHT = 0.055
DAMAGE_HUD_BOTTOM_Y = -0.915
DAMAGE_HUD_ROW_SPACING = 0.150
DAMAGE_HUD_EMPTY_WIDTH = 0.004
DAMAGE_HUD_LABEL_OFFSET_Y = 0.082
DAMAGE_HUD_LABEL_SCALE = 0.045
HEAD_TO_HEAD_CAR_LABEL_BACKGROUND_ALPHA = 0.45
DAMAGE_HUD_SHADOW_COLOR = (0.0, 0.0, 0.0, 0.62)
DAMAGE_HUD_TRACK_COLOR = (0.020, 0.023, 0.030, 0.92)
DAMAGE_HUD_FRAME_COLOR = (0.92, 0.94, 0.98, 0.80)
DAMAGE_HUD_INNER_FRAME_COLOR = (0.18, 0.20, 0.24, 0.90)
DAMAGE_HUD_ZERO_FILL_COLOR = (0.28, 0.92, 0.40, 0.92)
DAMAGE_HUD_MID_FILL_COLOR = (1.00, 0.74, 0.16, 0.96)
DAMAGE_HUD_FULL_FILL_COLOR = (1.00, 0.12, 0.08, 0.98)
DAMAGE_HUD_ELIMINATED_FILL_COLOR = (0.74, 0.0, 0.0, 1.0)

ColorRGBA = tuple[float, float, float, float]
RaceViewerConfig = HeadToHeadViewerConfig | HeatViewerConfig
RaceViewerEntry = HeadToHeadRaceEntry | HeatRaceEntry
DEFAULT_WINDOW_ICON_PATH = Path(__file__).resolve().parents[1] / "assets" / "textures" / "ursina.ico"

quiet_panda_image_logs()


def _create_configured_ursina_app(*, app_kwargs: dict[str, Any], preserve_project_y_up: bool = True) -> tuple[Any, Any]:
    """Create an Ursina app after applying project-wide Panda3D window config."""
    configured_app_kwargs = {"icon": str(DEFAULT_WINDOW_ICON_PATH), **app_kwargs}
    if preserve_project_y_up:
        configure_panda_y_up()
    configure_panda_antialiasing()
    ursina = cast(Any, import_module("ursina"))

    if preserve_project_y_up:
        restore_ursina_coordinate_system = patch_ursina_window_coordinate_system()
        try:
            app = ursina.Ursina(**configured_app_kwargs)
        finally:
            restore_ursina_coordinate_system()
    else:
        app = ursina.Ursina(**configured_app_kwargs)
    enable_render_antialiasing(ursina.scene)
    return ursina, app


@dataclass(frozen=True, slots=True)
class DamageHudSlot:
    """Screen-space placement for one car damage bar."""

    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(slots=True)
class DamageHudBar:
    """Small group of 2D objects that show one car's damage."""

    slot: DamageHudSlot
    shadow: Any
    frame: Any
    track: Any
    fill: Any
    cap: Any
    accent: Any
    label: Any | None = None


@dataclass(slots=True)
class AudioHudControl:
    """Small on-screen audio toggle and its key state."""

    button: Any
    label: Any
    key_state: AudioKeyToggleState


@dataclass(slots=True)
class HeadToHeadCarLabel:
    """Fixed-size screen-space name tag that follows a race car."""

    background: Any
    text: Any


@dataclass(frozen=True, slots=True)
class HeadToHeadCarLabelLayout:
    """Camera-specific size and placement for a floating car label."""

    width: float
    height: float
    text_scale: float
    vertical_offset: float


def _follow_camera_settings_for_view(view: CameraView) -> FollowCameraSettings:
    if view in (CameraView.DRONE, CameraView.FOLLOW_CAR):
        return FORMULA_DRONE_CAMERA_SETTINGS
    if view is CameraView.FOLLOW:
        return FORMULA_FOLLOW_CAMERA_SETTINGS
    return FORMULA_DRONE_CAMERA_SETTINGS


def build_scene(config: GameConfig) -> RunnableApp:
    """Create the single-car scene used for manual driving or one student controller."""
    if config.fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    if config.camera_view is CameraView.SPLIT_FOLLOW:
        raise ValueError("split_follow requires the head-to-head viewer")
    if config.human_recording_path is not None and config.student_controller is not None:
        raise ValueError("human gameplay recording is only available with manual control")

    app_kwargs: dict[str, Any] = {
        "title": config.title,
        "borderless": config.borderless,
        "fullscreen": config.fullscreen,
        "vsync": config.vsync,
        "development_mode": config.development_mode,
        "size": config.size,
    }
    if config.window_type is not None:
        app_kwargs["window_type"] = config.window_type
    ursina, app = _create_configured_ursina_app(app_kwargs=app_kwargs)
    human_recorder = None if config.human_recording_path is None else HumanGameplayRecorder(config.human_recording_path)
    app.human_gameplay_recorder = human_recorder
    if config.window_type is None:
        configure_window(ursina.window, config)
    app.setBackgroundColor(*NIGHT_SKY_COLOR)
    assets = create_scene_assets()

    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])

    resolved_track = resolve_track(config.track_id, config.track_seed)
    track_model = resolved_track.model
    track_samples = resolved_track.samples
    app.racing_track = resolved_track
    seeded_spawn_pose = race_spawn_poses(
        1,
        model=track_model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=config.random_seed,
        race_index=1,
    )
    default_spawn_pose = seeded_spawn_pose[0]
    if config.spawn_position is None:
        spawn_position = default_spawn_pose.position
        default_spawn_heading_degrees = default_spawn_pose.heading_degrees
        default_spawn_progress_distance_m = default_spawn_pose.progress_distance_m
    else:
        spawn_position = config.spawn_position
        default_spawn_heading_degrees = START_HEADING_DEGREES
        default_spawn_progress_distance_m = project_track_position(
            track_model,
            TrackPoint(spawn_position[0], spawn_position[2]),
        ).progress_distance_m
    spawn_heading_degrees = (
        default_spawn_heading_degrees if config.spawn_heading_degrees is None else config.spawn_heading_degrees
    )
    spawn_progress_distance_m = (
        default_spawn_progress_distance_m
        if config.spawn_progress_distance_m is None
        else config.spawn_progress_distance_m
    )
    start_finish_progress_pose = start_finish_pose_for_progress(
        model=track_model,
        start_progress_distance_m=spawn_progress_distance_m,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )
    start_finish_pose = start_finish_render_pose(
        position=start_finish_progress_pose.position,
        samples=track_samples,
    )

    add_world_floor(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        include_collision=False,
        samples=track_samples,
    )
    add_track(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        samples=track_samples,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        include_collision=False,
        legacy_lighting=resolved_track.track_id == TRACK_ID_MUGELLO_SHORT,
    )
    add_racing_scene_collisions(physics_world=physics_world, render=ursina.scene, samples=track_samples)
    add_trackside_scenery(
        ursina=ursina,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        samples=track_samples,
    )

    robot = create_robot_vehicle(
        world=physics_world,
        render=ursina.scene,
        name="student-robot-0",
        position=spawn_position,
        heading_degrees=spawn_heading_degrees,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
    )
    physics_scene.vehicles.append(robot)
    app.racing_robot = robot
    add_robot_visuals(ursina=ursina, robot=robot, assets=assets, team_color=config.team_color)

    student_runtime = (
        student_marshal_runtime(
            robot=robot,
            start_position=TrackPoint(spawn_position[0], spawn_position[2]),
            starting_progress_distance_m=spawn_progress_distance_m,
            model=track_model,
        )
        if config.student_controller is not None
        else None
    )
    student_recovery_config = (
        _head_to_head_viewer_recovery_config(HeadToHeadRaceRules()) if student_runtime is not None else None
    )

    add_lighting(ursina)
    camera_rig = CameraRig(view=config.camera_view)
    apply_camera_view(
        ursina=ursina,
        view=camera_rig.view,
        target=robot.chassis_np,
        rig=camera_rig,
        follow_settings=_follow_camera_settings_for_view(camera_rig.view),
        track_model=track_model,
    )

    fps_display = ursina.Text(text="FPS: --", position=(-0.87, 0.46), scale=0.85, background=True)
    speed_display = ursina.Text(text="0.0 km/h", position=(-0.87, 0.40), scale=0.75, background=True)
    damage_bars = _add_damage_hud_bars(ursina=ursina, colors=(config.team_color,))
    audio_runtime = create_racing_audio_runtime(ursina=ursina, config=config.audio)
    _register_audio_vehicles(audio_runtime=audio_runtime, robots=(robot,))
    audio_control = _add_audio_hud_control(ursina=ursina, audio_runtime=audio_runtime)
    sensor_state = RobotSensorBuilderState()
    simulation_time_s = 0.0
    simulation_accumulator_seconds = 0.0

    def update() -> None:
        """Advance the playable scene by one rendered frame."""
        nonlocal sensor_state, simulation_accumulator_seconds, simulation_time_s
        frame_delta_seconds = min(float(ursina.time.dt), PLAYABLE_MAX_FRAME_DELTA_SECONDS)
        update_camera_cycle(camera_rig, cycle_key_down=bool(ursina.held_keys["v"]))
        _update_audio_key_control(
            audio_control=audio_control, audio_runtime=audio_runtime, mute_key_down=bool(ursina.held_keys["m"])
        )

        simulation_accumulator_seconds += frame_delta_seconds
        fixed_steps = 0
        while (
            simulation_accumulator_seconds >= config.fixed_delta_seconds
            and fixed_steps < PLAYABLE_MAX_FIXED_STEPS_PER_FRAME
        ):
            simulation_time_s += config.fixed_delta_seconds
            if not robot.eliminated:
                if config.student_controller is None:
                    sync_gamepad_axes(ursina.held_keys)
                    command = manual_drive_command(ursina.held_keys)
                    if human_recorder is not None:
                        sensors, sensor_state = build_robot_sensors(
                            physics_world=physics_world,
                            robot=robot,
                            track_model=track_model,
                            time_s=simulation_time_s,
                            dt_s=config.fixed_delta_seconds,
                            previous_state=sensor_state,
                        )
                        human_recorder.record(
                            simulation_time_s=simulation_time_s,
                            sensors=sensors,
                            command=command,
                        )
                else:
                    sensors, sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=robot,
                        track_model=track_model,
                        time_s=simulation_time_s,
                        dt_s=config.fixed_delta_seconds,
                        previous_state=sensor_state,
                    )
                    command = config.student_controller(sensors)
                audio_runtime.record_command(robot, command)
                apply_robot_vehicle_command(robot=robot, command=command)
            physics_scene.step(config.fixed_delta_seconds)
            simulation_accumulator_seconds -= config.fixed_delta_seconds
            fixed_steps += 1

            student_contact_state: RaceContactState | None = None
            student_projection: TrackProjection | None = None
            if student_runtime is not None and student_recovery_config is not None:
                student_contact_state = race_contact_states(physics_world=physics_world, runtimes=(student_runtime,))[0]
                student_projection = project_track_position(track_model, robot_track_point(robot))
            apply_wall_impact_damage(
                physics_world=physics_world,
                robots=(robot,),
                fixed_time_step=physics_scene.fixed_time_step,
            )
            if (
                student_runtime is not None
                and student_recovery_config is not None
                and student_contact_state is not None
                and student_projection is not None
            ):
                update_race_runtime_after_step(
                    runtime=student_runtime,
                    projection=student_projection,
                    contact_state=student_contact_state,
                    elapsed_seconds=simulation_time_s,
                    delta_seconds=config.fixed_delta_seconds,
                )
                if (
                    not robot.eliminated
                    and maybe_marshal_race_runtimes(
                        runtimes=(student_runtime,),
                        projections=(student_projection,),
                        recovery_config=student_recovery_config,
                        delta_seconds=config.fixed_delta_seconds,
                    )
                    > 0
                ):
                    sensor_state = student_sensor_state_after_marshal(
                        previous_state=sensor_state,
                        projection=student_projection,
                        time_s=simulation_time_s,
                    )
                    camera_rig.reset_follow_history()

        if fixed_steps == PLAYABLE_MAX_FIXED_STEPS_PER_FRAME:
            simulation_accumulator_seconds = min(simulation_accumulator_seconds, config.fixed_delta_seconds)

        apply_camera_view(
            ursina=ursina,
            view=camera_rig.view,
            target=robot.chassis_np,
            rig=camera_rig,
            delta_seconds=frame_delta_seconds,
            follow_settings=_follow_camera_settings_for_view(camera_rig.view),
            track_model=track_model,
        )
        audio_runtime.update(frame_delta_seconds)
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)
        fps_display.text = fps_text_for_delta(frame_delta_seconds)
        speed_display.text = (
            "OUT" if robot.eliminated else f"{abs(float(robot.vehicle.getCurrentSpeedKmHour())):>4.1f} km/h"
        )
        _update_damage_hud_bars(bars=damage_bars, robots=(robot,))

    ursina.Entity(name="simulation_loop", update=update, ignore_paused=True)
    return cast(RunnableApp, app)


def student_marshal_runtime(
    *,
    robot: RobotVehicle,
    start_position: TrackPoint,
    starting_progress_distance_m: float | None = None,
    model: TrackProgressModel | None = None,
) -> RaceCarRuntime:
    """Create the race bookkeeping needed to reset a stuck student car."""
    track_model = default_track_progress_model() if model is None else model
    start_projection = project_track_position(track_model, start_position)
    tracker = lap_progress_tracker_for_spawn_pose(
        model=track_model,
        spawn_pose=RaceSpawnPose(
            position=(start_position.x, 0.0, start_position.z),
            heading_degrees=start_projection.heading_degrees,
            progress_distance_m=(
                start_projection.progress_distance_m
                if starting_progress_distance_m is None
                else starting_progress_distance_m
            ),
        ),
    )
    return RaceCarRuntime(robot=robot, tracker=tracker)


def student_sensor_state_after_marshal(
    *,
    previous_state: RobotSensorBuilderState,
    projection: TrackProjection,
    time_s: float,
) -> RobotSensorBuilderState:
    """Reset sensor bookkeeping after the marshal moves a student car."""
    return RobotSensorBuilderState(
        time_s=time_s,
        position=projection.nearest_center,
        heading_degrees=projection.heading_degrees,
        speed_mps=0.0,
        distance_m=previous_state.distance_m,
        tick=previous_state.tick,
    )


def build_head_to_head_viewer_scene(config: HeadToHeadViewerConfig) -> RunnableApp:
    """Create the visual race viewer for two controller teams."""
    _validate_head_to_head_viewer_config(config)
    return _build_race_viewer_scene(config)


def create_heat_viewer_app(config: HeatViewerConfig) -> RunnableApp:
    """Create a four- or eight-controller heat using the shared viewer and physics."""
    validate_heat_entrants(config.entrants)
    if config.race_count < 1:
        raise ValueError("race_count must be at least one")
    if config.round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    if config.fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    if config.camera_view is CameraView.SPLIT_FOLLOW:
        raise ValueError("split_follow requires the two-team head-to-head viewer")
    return _build_race_viewer_scene(config)


def _race_viewer_entries(*, config: RaceViewerConfig, race_index: int) -> tuple[RaceViewerEntry, ...]:
    if isinstance(config, HeatViewerConfig):
        return heat_race_entries(
            entrant_count=len(config.entrants), race_index=race_index, random_seed=config.random_seed
        )
    return head_to_head_race_entries(
        challenger_copies=config.challenger_copies,
        incumbent_copies=config.incumbent_copies,
        race_index=race_index,
        random_seed=config.random_seed,
    )


def _race_viewer_car_id(entry: RaceViewerEntry) -> str:
    """Keep camera and timing identities independent of position or grid slot."""
    return f"{entry.role}:{entry.copy_index}"


def _build_race_viewer_scene(config: RaceViewerConfig) -> RunnableApp:
    """Share track, car physics, cameras and HUD between h2h and individual heats."""
    race_rules = _head_to_head_viewer_rules(config)
    recovery_config = _head_to_head_viewer_recovery_config(race_rules)

    app_kwargs: dict[str, Any] = {
        "title": config.title,
        "borderless": config.borderless,
        "fullscreen": config.fullscreen,
        "vsync": config.vsync,
        "development_mode": config.development_mode,
        "size": config.size,
    }
    if config.window_type is not None:
        app_kwargs["window_type"] = config.window_type
    ursina, app = _create_configured_ursina_app(app_kwargs=app_kwargs)
    if config.window_type is None:
        configure_window(ursina.window, config)
    app.setBackgroundColor(*NIGHT_SKY_COLOR)
    # Explicitly clear the scene region when cars move or the camera changes;
    # some offscreen pipelines disable the window-level clear.
    ursina.camera.display_region.setClearColor(app.win.getClearColor())
    ursina.camera.display_region.setClearColorActive(True)
    ursina.camera.display_region.setClearDepthActive(True)
    assets = create_scene_assets()

    physics_world = create_physics_world()
    physics_scene = PhysicsScene(world=physics_world, vehicles=[])

    resolved_track = resolve_track(config.track_id, config.track_seed)
    model = resolved_track.model
    track_samples = resolved_track.samples
    app.racing_track = resolved_track
    add_world_floor(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        include_collision=False,
        samples=track_samples,
    )
    start_finish_progress_pose = seeded_race_start_finish_pose(
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=config.random_seed,
        race_index=1,
    )
    start_finish_pose = start_finish_render_pose(
        position=start_finish_progress_pose.position,
        samples=track_samples,
    )
    start_finish_track_line = add_track(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        samples=track_samples,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        include_collision=False,
        legacy_lighting=resolved_track.track_id == TRACK_ID_MUGELLO_SHORT,
    )
    add_racing_scene_collisions(physics_world=physics_world, render=ursina.scene, samples=track_samples)
    start_finish_gantry = add_trackside_scenery(
        ursina=ursina,
        assets=assets,
        start_line_position=start_finish_pose.position,
        start_line_heading_degrees=start_finish_pose.heading_degrees,
        samples=track_samples,
    )

    entries = _race_viewer_entries(config=config, race_index=1)
    controllers = _head_to_head_viewer_controllers(config=config, entries=entries)
    spawn_poses = race_spawn_poses(
        len(entries),
        model=model,
        config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        random_seed=config.random_seed,
        race_index=1,
        shuffle_grid=not isinstance(config, HeatViewerConfig),
    )
    runtimes: list[RaceCarRuntime] = []
    for index, (entry, spawn_pose) in enumerate(zip(entries, spawn_poses, strict=True)):
        robot = create_robot_vehicle(
            world=physics_world,
            render=ursina.scene,
            name=f"h2h-robot-{entry.role}-{entry.copy_index}-{index}",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        add_robot_visuals(
            ursina=ursina,
            robot=robot,
            assets=assets,
            team_color=_head_to_head_car_paint_color(config=config, entry=entry),
        )
        label = _add_head_to_head_car_label(ursina=ursina, config=config, entry=entry)
        _style_head_to_head_label(label=label, config=config, entry=entry)
        runtimes.append(
            RaceCarRuntime(
                robot=robot,
                tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
                label=label,
            )
        )

    add_lighting(ursina)
    app.racing_runtimes = runtimes
    app.racing_entries = entries
    camera_rig = CameraRig(view=config.camera_view)
    split_rigs = (CameraRig(view=CameraView.FOLLOW), CameraRig(view=CameraView.FOLLOW))
    split_cameras: SplitScreenCameras | None = None
    app.racing_camera_rig = camera_rig

    hud_parent = ursina.application.base.aspect2d
    hud_aspect_ratio = float(ursina.window.aspect_ratio)
    hud_left_x = -hud_aspect_ratio + 0.06
    live_hud_width = 0.92
    status_background = _panda2d_hud_card(
        parent=hud_parent,
        name="head-to-head-status-background",
        position=(hud_left_x + live_hud_width / 2.0, 0.89),
        scale=(live_hud_width, 0.16),
        color=(0.020, 0.023, 0.030, 0.90),
        bin_order=90,
    )
    status_display = _panda2d_hud_text(
        parent=hud_parent,
        name="head-to-head-status",
        text="",
        position=(hud_left_x + 0.04, 0.87),
        scale=0.060,
        color=(0.96, 0.98, 1.0, 1.0),
        bin_order=91,
        align_left=True,
    )
    result_background = _panda2d_hud_card(
        parent=hud_parent,
        name="head-to-head-result-background",
        position=(0.0, 0.14),
        scale=(2.5, 0.76) if isinstance(config, HeatViewerConfig) else (2.5, 0.48),
        color=(0.020, 0.023, 0.030, 0.94),
        bin_order=110,
    )
    result_display = _panda2d_hud_text(
        parent=hud_parent,
        name="head-to-head-result",
        text="",
        position=(0.0, 0.38) if isinstance(config, HeatViewerConfig) else (0.0, 0.24),
        scale=0.070,
        color=(1.0, 1.0, 1.0, 1.0),
        bin_order=111,
    )
    result_background.hide()
    result_display.hide()
    damage_bars = _add_damage_hud_bars(
        ursina=ursina, colors=tuple(_head_to_head_car_paint_color(config=config, entry=entry) for entry in entries)
    )
    _add_head_to_head_damage_hud_labels(
        ursina=ursina,
        bars=damage_bars,
        config=config,
        entries=entries,
    )
    damage_hud_root = hud_parent.attachNewNode("race-damage-hud")
    for bar in damage_bars:
        for node in (bar.shadow, bar.frame, bar.track, bar.fill, bar.cap, bar.accent, bar.label):
            if node is not None:
                node.reparentTo(damage_hud_root)
    damage_hud_root.hide()
    app.racing_damage_hud = damage_hud_root
    audio_runtime = create_racing_audio_runtime(ursina=ursina, config=config.audio)
    _register_audio_vehicles(audio_runtime=audio_runtime, robots=tuple(runtime.robot for runtime in runtimes))
    audio_control = _add_audio_hud_control(ursina=ursina, audio_runtime=audio_runtime)

    timing_tower = TimingTower(
        ursina=ursina,
        on_select=camera_rig.select_follow_car,
        logo_path=Path(__file__).resolve().parents[1] / "assets" / "graphics" / "f110-blue.png",
    )
    race_timing = RaceTiming()
    app.racing_timing_tower = timing_tower
    app.racing_timing = race_timing

    split_hud = hud_parent.attachNewNode("head-to-head-split-hud")
    _panda2d_hud_card(
        parent=split_hud,
        name="head-to-head-split-divider",
        position=(0.0, 0.0),
        scale=(0.012, 2.0),
        color=(0.02, 0.025, 0.03, 1.0),
        bin_order=70,
    )
    split_title_backgrounds = tuple(
        _panda2d_hud_card(
            parent=split_hud,
            name=f"head-to-head-split-title-background-{side}",
            position=(0.0, 0.72),
            scale=(1.4, 0.13),
            color=(0.020, 0.023, 0.030, 0.90),
            bin_order=90,
        )
        for side in ("left", "right")
    )
    split_titles = tuple(
        _panda2d_hud_text(
            parent=split_hud,
            name=f"head-to-head-split-{role}",
            text=_short_head_to_head_name(name, max_length=26),
            position=(0.0, 0.70),
            scale=0.060,
            color=(0.96, 0.98, 1.0, 1.0),
            bin_order=91,
        )
        for role, name in (
            (("challenger", config.challenger_name), ("incumbent", config.incumbent_name))
            if isinstance(config, HeadToHeadViewerConfig)
            else ()
        )
    )
    split_hud.hide()
    regular_damage_slots = damage_hud_layout(len(damage_bars))

    def update_view(delta_seconds: float) -> None:
        """Keep the two split panes tied to teams, independently of grid order."""
        nonlocal split_cameras
        split = isinstance(config, HeadToHeadViewerConfig) and camera_rig.view is CameraView.SPLIT_FOLLOW
        if split:
            if split_cameras is None:
                split_cameras = SplitScreenCameras(ursina=ursina)
                app.racing_split_cameras = split_cameras
            if not split_cameras.active:
                for rig in split_rigs:
                    rig.reset_follow_history()
            split_cameras.set_active(True)
            ursina.camera.orthographic = False
            ursina.camera.fov = FORMULA_FOLLOW_CAMERA_SETTINGS.fov
            targets = _head_to_head_split_target_runtimes(
                entries=cast(tuple[HeadToHeadRaceEntry, ...], entries), runtimes=tuple(runtimes)
            )
            for runtime, rig, camera, lens in zip(
                targets,
                split_rigs,
                (ursina.camera, split_cameras.right_transform),
                (ursina.camera.perspective_lens, split_cameras.right_lens),
                strict=True,
            ):
                apply_follow_camera_view(
                    ursina=ursina,
                    camera=camera,
                    lens=lens,
                    target=runtime.robot.chassis_np,
                    rig=rig,
                    delta_seconds=delta_seconds,
                    follow_settings=FORMULA_FOLLOW_CAMERA_SETTINGS,
                    track_model=model,
                )
            aspect = float(app.win.getXSize()) / max(1, int(app.win.getYSize()))
            for title, background, side in zip(split_titles, split_title_backgrounds, (-1.0, 1.0), strict=True):
                left = -aspect + 0.06 if side < 0 else 0.06
                right = -0.06 if side < 0 else aspect - 0.06
                if side < 0 and timing_tower.visible:
                    left = max(left, timing_tower.right_edge + 0.04)
                width = min(1.4, max(0.2, right - left))
                center = (left + right) / 2.0
                title.setX(center)
                title.setScale(
                    min(0.060, (width - 0.08) / max(1.0, float(title.node().calcWidth(title.node().getText()))))
                )
                background.setX(center)
                background.setScale(width, 0.13, 1.0)
            if len(damage_bars) == 2:
                for bar, entry in zip(damage_bars, entries, strict=True):
                    side = -1.0 if entry.role == "challenger" else 1.0
                    _move_damage_hud_bar(bar=bar, center_x=side * aspect / 2.0)
            split_hud.show()
            # Each pane has a fixed team heading; single-camera floating tags
            # cannot be projected correctly into both views.
            for runtime in runtimes:
                if isinstance(runtime.label, HeadToHeadCarLabel):
                    runtime.label.background.hide()
                    runtime.label.text.hide()
        else:
            if split_cameras is not None:
                split_cameras.set_active(False)
            split_hud.hide()
            for bar, slot in zip(damage_bars, regular_damage_slots, strict=True):
                _move_damage_hud_bar(bar=bar, center_x=slot.center_x)
            target = _head_to_head_camera_target_runtime(
                config=config,
                entries=entries,
                runtimes=tuple(runtimes),
                selected_car_id=camera_rig.selected_car_id,
            )
            apply_camera_view(
                ursina=ursina,
                view=camera_rig.view,
                target=target.robot.chassis_np,
                rig=camera_rig,
                delta_seconds=delta_seconds,
                follow_settings=_follow_camera_settings_for_view(camera_rig.view),
                track_model=model,
            )
            _update_head_to_head_car_labels(ursina=ursina, view=camera_rig.view, runtimes=tuple(runtimes))

    race_index = 1
    race_elapsed_seconds = 0.0
    simulation_accumulator_seconds = 0.0
    race_concluded = False
    completed_race_results: list[HeadToHeadRaceResult] = []
    completed_heat_results: list[HeatRaceResult] = []
    timing_key_was_down = False
    previous_marshal_counts: dict[str, int] = {}

    def sample_timing() -> None:
        """Record progress at simulation time, regardless of tower visibility."""
        samples: list[TimingSample] = []
        for entry, runtime in zip(entries, runtimes, strict=True):
            car_id = _race_viewer_car_id(entry)
            samples.append(
                TimingSample(
                    car_id=car_id,
                    distance_m=race_scored_distance_m(runtime),
                    eliminated=runtime.robot.eliminated,
                    discontinuity=previous_marshal_counts.get(car_id, 0) != runtime.marshal_count,
                )
            )
            previous_marshal_counts[car_id] = runtime.marshal_count
        race_timing.update(elapsed_seconds=race_elapsed_seconds, samples=tuple(samples))

    def update_timing_hud() -> None:
        entries_by_id = {_race_viewer_car_id(entry): entry for entry in entries}
        runtimes_by_id = {_race_viewer_car_id(entry): runtime for entry, runtime in zip(entries, runtimes, strict=True)}
        for standing in race_timing.rows:
            label = runtimes_by_id[standing.car_id].label
            if isinstance(label, HeadToHeadCarLabel):
                name = _head_to_head_car_label(config=config, entry=entries_by_id[standing.car_id])
                _set_panda2d_hud_text(label.text, f"P{standing.rank} {name}")
        timing_tower.update(
            rows=tuple(
                TimingTowerRow(
                    car_id=standing.car_id,
                    name=_head_to_head_car_label(config=config, entry=entries_by_id[standing.car_id]),
                    color=_head_to_head_car_paint_color(config=config, entry=entries_by_id[standing.car_id]),
                    rank=standing.rank,
                    gap_seconds=standing.gap_seconds,
                    eliminated=standing.eliminated,
                )
                for standing in race_timing.rows
            ),
            remaining_seconds=max(0.0, config.round_seconds - race_elapsed_seconds),
            race_index=race_index,
            race_count=config.race_count,
            selected_car_id=camera_rig.selected_car_id,
        )
        for node in (status_background, status_display):
            if timing_tower.visible:
                node.hide()
            else:
                node.show()

    sample_timing()
    update_timing_hud()
    update_view(0.0)

    def start_race(
        next_race_index: int,
    ) -> tuple[tuple[RaceViewerEntry, ...], tuple[RobotController | None, ...]]:
        """Reset cars, labels, and start/finish art for the next race."""
        nonlocal race_elapsed_seconds
        race_elapsed_seconds = 0.0
        camera_rig.reset_follow_history()
        for rig in split_rigs:
            rig.reset_follow_history()
        next_entries = _race_viewer_entries(config=config, race_index=next_race_index)
        next_spawn_poses = race_spawn_poses(
            len(runtimes),
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=config.random_seed,
            race_index=next_race_index,
            shuffle_grid=not isinstance(config, HeatViewerConfig),
        )
        next_start_finish_progress_pose = seeded_race_start_finish_pose(
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=config.random_seed,
            race_index=next_race_index,
        )
        next_start_finish_pose = start_finish_render_pose(
            position=next_start_finish_progress_pose.position,
            samples=track_samples,
        )
        set_start_finish_pose(
            start_finish_track_line,
            position=next_start_finish_pose.position,
            heading_degrees=next_start_finish_pose.heading_degrees,
        )
        set_start_finish_gantry_pose(
            start_finish_gantry,
            position=next_start_finish_pose.position,
            heading_degrees=next_start_finish_pose.heading_degrees,
        )
        for index, (runtime, entry, spawn_pose) in enumerate(
            zip(runtimes, next_entries, next_spawn_poses, strict=True)
        ):
            runtime.tracker = lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose)
            runtime.stuck_seconds = 0.0
            runtime.low_progress_seconds = 0.0
            runtime.off_track_seconds = 0.0
            runtime.recent_progress_mps = 0.0
            runtime.max_speed_mps = 0.0
            runtime.contact_state = RaceContactState()
            runtime.sensor_state = RobotSensorBuilderState()
            runtime.marshal_count = 0
            runtime.marshal_penalty_m = 0.0
            runtime.marshal_cooldown_seconds = 0.0
            reset_robot_vehicle(
                runtime.robot,
                position=spawn_pose.position,
                heading_degrees=spawn_pose.heading_degrees,
                reset_damage=True,
            )
            team_color = _head_to_head_car_paint_color(config=config, entry=entry)
            apply_robot_team_color(
                robot=runtime.robot, assets=assets, team_color=_head_to_head_car_paint_color(config=config, entry=entry)
            )
            _style_damage_hud_bar(bar=damage_bars[index], color=team_color)
            _style_head_to_head_label(label=runtime.label, config=config, entry=entry)
            _update_head_to_head_damage_hud_label(
                bar=damage_bars[index],
                config=config,
                entry=entry,
                runtime=runtime,
            )
            runtime.robot.chassis_np.setName(f"h2h-robot-{entry.role}-{entry.copy_index}-{index}")
        return next_entries, _head_to_head_viewer_controllers(config=config, entries=next_entries)

    def update() -> None:
        """Advance the head-to-head viewer by one rendered frame."""
        nonlocal controllers, entries, race_concluded, race_elapsed_seconds, race_index, simulation_accumulator_seconds
        nonlocal timing_key_was_down
        frame_delta_seconds = min(float(ursina.time.dt), 0.25)
        timing_key_down = bool(ursina.held_keys["l"])
        if timing_key_down and not timing_key_was_down:
            timing_tower.set_visible(not timing_tower.visible)
        timing_key_was_down = timing_key_down
        update_camera_cycle(
            camera_rig,
            cycle_key_down=bool(ursina.held_keys["v"]),
            include_split=isinstance(config, HeadToHeadViewerConfig),
        )
        _update_audio_key_control(
            audio_control=audio_control, audio_runtime=audio_runtime, mute_key_down=bool(ursina.held_keys["m"])
        )
        if race_concluded:
            update_timing_hud()
            update_view(frame_delta_seconds)
            audio_runtime.update(frame_delta_seconds)
            _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)
            return

        simulation_accumulator_seconds += frame_delta_seconds
        while simulation_accumulator_seconds >= config.fixed_delta_seconds:
            for entry, controller, runtime in zip(entries, controllers, runtimes, strict=True):
                command = _head_to_head_viewer_command(
                    config=config,
                    entry=entry,
                    controller=controller,
                    model=model,
                    runtime=runtime,
                    physics_world=physics_world,
                    runtimes=tuple(runtimes),
                    time_s=race_elapsed_seconds,
                    dt_s=config.fixed_delta_seconds,
                    held_keys=ursina.held_keys,
                )
                audio_runtime.record_command(runtime.robot, command)
                apply_robot_vehicle_command(robot=runtime.robot, command=command)

            physics_scene.step(config.fixed_delta_seconds)
            race_elapsed_seconds += config.fixed_delta_seconds
            simulation_accumulator_seconds -= config.fixed_delta_seconds

            contact_states = race_contact_states(physics_world=physics_world, runtimes=tuple(runtimes))
            apply_wall_impact_damage(
                physics_world=physics_world,
                robots=tuple(runtime.robot for runtime in runtimes),
                fixed_time_step=physics_scene.fixed_time_step,
            )
            projections: list[TrackProjection] = []
            for runtime, contact_state in zip(runtimes, contact_states, strict=True):
                projection = project_track_position(model, robot_track_point(runtime.robot))
                projections.append(projection)
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=race_elapsed_seconds,
                    delta_seconds=config.fixed_delta_seconds,
                )
            if recovery_config is not None:
                maybe_marshal_race_runtimes(
                    runtimes=tuple(runtimes),
                    projections=tuple(projections),
                    recovery_config=recovery_config,
                    delta_seconds=config.fixed_delta_seconds,
                )

            sample_timing()
            if race_elapsed_seconds >= config.round_seconds:
                break

        update_timing_hud()
        update_view(frame_delta_seconds)
        audio_runtime.update(frame_delta_seconds)
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)
        _update_head_to_head_hud(
            status_display=status_display,
            damage_bars=damage_bars,
            config=config,
            race_index=race_index,
            entries=entries,
            runtimes=tuple(runtimes),
            race_elapsed_seconds=race_elapsed_seconds,
        )
        _update_damage_hud_bars(bars=damage_bars, robots=tuple(runtime.robot for runtime in runtimes))

        if race_elapsed_seconds < config.round_seconds:
            return

        if isinstance(config, HeatViewerConfig):
            completed_heat_results.append(
                heat_race_result_from_runtimes(
                    entrants=config.entrants,
                    entries=cast(tuple[HeatRaceEntry, ...], entries),
                    runtimes=tuple(runtimes),
                    race_index=race_index,
                )
            )
        else:
            completed_race_results.append(
                _head_to_head_race_result_from_runtimes(
                    config=config,
                    race_rules=race_rules,
                    race_index=race_index,
                    entries=cast(tuple[HeadToHeadRaceEntry, ...], entries),
                    runtimes=tuple(runtimes),
                )
            )
        if race_index >= config.race_count:
            if isinstance(config, HeatViewerConfig):
                heat_result = HeatResult(
                    entrant_names=tuple(entrant.name for entrant in config.entrants),
                    round_seconds=config.round_seconds,
                    races=tuple(completed_heat_results),
                    random_seed=config.random_seed,
                    track_id=resolved_track.track_id,
                    track_seed=resolved_track.seed,
                    rules=race_rules,
                    fixed_delta_seconds=config.fixed_delta_seconds,
                )
                print(format_heat_result(heat_result))
                _set_panda2d_hud_text(result_display, format_heat_result_banner(heat_result))
                app.racing_result = heat_result
            else:
                final_result = HeadToHeadResult(
                    challenger_name=config.challenger_name,
                    incumbent_name=config.incumbent_name,
                    round_seconds=config.round_seconds,
                    win_margin_m=race_rules.win_margin_m,
                    races=tuple(completed_race_results),
                    random_seed=config.random_seed,
                    track_id=resolved_track.track_id,
                    track_seed=resolved_track.seed,
                    rules=race_rules,
                    fixed_delta_seconds=config.fixed_delta_seconds,
                )
                print(format_head_to_head_result(final_result))
                _set_panda2d_hud_text(result_display, format_head_to_head_result_banner(final_result))
                _set_panda2d_hud_text_color(
                    result_display,
                    _head_to_head_result_color(config=config, result=final_result),
                )
                app.racing_result = final_result
            text_width = max(
                float(result_display.node().calcWidth(line)) for line in result_display.node().getText().splitlines()
            )
            result_display.setScale(min(0.070, 2.34 / max(1.0, text_width)))
            result_background.show()
            result_display.show()
            for runtime in runtimes:
                command = RobotCommand()
                audio_runtime.record_command(runtime.robot, command)
                apply_robot_vehicle_command(robot=runtime.robot, command=command)
            race_concluded = True
            simulation_accumulator_seconds = 0.0
            return

        race_index += 1
        entries, controllers = start_race(race_index)
        app.racing_entries = entries
        race_timing.reset()
        previous_marshal_counts.clear()
        sample_timing()
        update_timing_hud()
        update_view(0.0)
        simulation_accumulator_seconds = 0.0

    ursina.Entity(name="head_to_head_viewer_loop", update=update, ignore_paused=True)
    return cast(RunnableApp, app)


def _validate_head_to_head_viewer_config(config: HeadToHeadViewerConfig) -> None:
    if config.race_count < 1:
        raise ValueError("race_count must be at least one")
    if config.round_seconds <= 0.0:
        raise ValueError("round_seconds must be positive")
    if config.fixed_delta_seconds <= 0.0:
        raise ValueError("fixed_delta_seconds must be positive")
    if config.challenger_copies < 1:
        raise ValueError("challenger_copies must be at least one")
    if config.incumbent_copies < 1:
        raise ValueError("incumbent_copies must be at least one")
    if config.challenger_keyboard and config.incumbent_keyboard:
        raise ValueError("keyboard control can only be assigned to one head-to-head side")
    if config.challenger_keyboard and config.challenger_copies != 1:
        raise ValueError("keyboard-controlled challenger must use exactly one copy")
    if config.incumbent_keyboard and config.incumbent_copies != 1:
        raise ValueError("keyboard-controlled incumbent must use exactly one copy")
    if config.challenger_keyboard and config.challenger_controller is not None:
        raise ValueError("challenger keyboard control cannot be combined with a challenger controller")
    if config.incumbent_keyboard and config.incumbent_controller is not None:
        raise ValueError("incumbent keyboard control cannot be combined with an incumbent controller")
    if not (config.challenger_keyboard or config.challenger_controller is not None):
        raise ValueError("challenger needs keyboard control or a student controller")
    if not (config.incumbent_keyboard or config.incumbent_controller is not None):
        raise ValueError("incumbent needs keyboard control or a student controller")
    _head_to_head_viewer_rules(config)


def _head_to_head_viewer_rules(config: RaceViewerConfig) -> HeadToHeadRaceRules:
    if isinstance(config, HeatViewerConfig) or config.win_margin_m == HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M:
        return config.rules
    return replace(config.rules, win_margin_m=config.win_margin_m)


def _head_to_head_viewer_recovery_config(rules: HeadToHeadRaceRules) -> RaceRecoveryConfig | None:
    if not rules.marshal_enabled:
        return None
    return RaceRecoveryConfig(
        stuck_seconds=rules.marshal_stuck_seconds,
        distance_penalty_m=rules.marshal_penalty_m,
        cooldown_seconds=rules.marshal_cooldown_seconds,
    )


def _head_to_head_viewer_command(
    *,
    config: RaceViewerConfig,
    entry: RaceViewerEntry,
    controller: RobotController | None,
    model: Any,
    runtime: RaceCarRuntime,
    physics_world: Any,
    runtimes: tuple[RaceCarRuntime, ...],
    time_s: float,
    dt_s: float,
    held_keys: Any,
) -> RobotCommand:
    if _head_to_head_viewer_keyboard_controlled(config=config, entry=entry):
        sync_gamepad_axes(held_keys)
        return manual_drive_command(held_keys)
    if controller is None:
        raise ValueError("head-to-head entry has no controller")
    sensors, runtime.sensor_state = build_robot_sensors(
        physics_world=physics_world,
        robot=runtime.robot,
        track_model=model,
        time_s=time_s,
        dt_s=dt_s,
        previous_state=runtime.sensor_state,
        other_robot_node_names=_head_to_head_other_runtime_node_names(runtime=runtime, runtimes=runtimes),
        other_robots=_head_to_head_other_runtime_robots(runtime=runtime, runtimes=runtimes),
    )
    return controller(sensors)


def _head_to_head_viewer_controllers(
    *, config: RaceViewerConfig, entries: tuple[RaceViewerEntry, ...]
) -> tuple[RobotController | None, ...]:
    """Create independent controller state for every watched car and race."""
    controllers: list[RobotController | None] = []
    for entry in entries:
        prototype = _head_to_head_viewer_controller(config=config, entry=entry)
        controllers.append(None if prototype is None else controller_for_copy(prototype))
    return tuple(controllers)


def _head_to_head_viewer_controller(*, config: RaceViewerConfig, entry: RaceViewerEntry) -> RobotController | None:
    if isinstance(config, HeatViewerConfig):
        assert isinstance(entry, HeatRaceEntry)
        return config.entrants[entry.entrant_index].controller
    if entry.role == "challenger":
        return config.challenger_controller
    return config.incumbent_controller


def _head_to_head_viewer_keyboard_controlled(*, config: RaceViewerConfig, entry: RaceViewerEntry) -> bool:
    if isinstance(config, HeatViewerConfig):
        return False
    if entry.role == "challenger":
        return config.challenger_keyboard
    return config.incumbent_keyboard


def _head_to_head_camera_target_runtime(
    *,
    config: RaceViewerConfig,
    entries: tuple[RaceViewerEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    selected_car_id: str | None = None,
) -> RaceCarRuntime:
    if selected_car_id is not None:
        for entry, runtime in zip(entries, runtimes, strict=True):
            if _race_viewer_car_id(entry) == selected_car_id:
                return runtime
    for entry, runtime in zip(entries, runtimes, strict=True):
        if _head_to_head_viewer_keyboard_controlled(config=config, entry=entry):
            return runtime
    return _leader_runtime(runtimes)


def _head_to_head_split_target_runtimes(
    *, entries: tuple[HeadToHeadRaceEntry, ...], runtimes: tuple[RaceCarRuntime, ...]
) -> tuple[RaceCarRuntime, RaceCarRuntime]:
    """Follow copy zero of the challenger on the left and incumbent on the right."""
    targets = {entry.role: runtime for entry, runtime in zip(entries, runtimes, strict=True) if entry.copy_index == 0}
    return targets["challenger"], targets["incumbent"]


def _head_to_head_other_runtime_node_names(
    *, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]
) -> frozenset[str]:
    return frozenset(
        _head_to_head_runtime_node_name(other_runtime.robot.chassis_np.node())
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _head_to_head_other_runtime_robots(
    *, runtime: RaceCarRuntime, runtimes: tuple[RaceCarRuntime, ...]
) -> tuple[RobotVehicle, ...]:
    return tuple(
        other_runtime.robot
        for other_runtime in runtimes
        if other_runtime is not runtime and not bool(getattr(other_runtime.robot, "eliminated", False))
    )


def _head_to_head_runtime_node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""


def _style_head_to_head_label(
    *, label: HeadToHeadCarLabel | None, config: RaceViewerConfig, entry: RaceViewerEntry
) -> None:
    if label is None:
        return
    team_color = _head_to_head_car_paint_color(config=config, entry=entry)
    label.background.setColor(
        team_color[0],
        team_color[1],
        team_color[2],
        HEAD_TO_HEAD_CAR_LABEL_BACKGROUND_ALPHA,
    )
    _set_panda2d_hud_text(label.text, _head_to_head_car_label(config=config, entry=entry))
    _set_panda2d_hud_text_color(label.text, (1.0, 1.0, 1.0, 1.0))


def _add_head_to_head_car_label(*, ursina: Any, config: RaceViewerConfig, entry: RaceViewerEntry) -> HeadToHeadCarLabel:
    parent = ursina.application.base.aspect2d
    team_color = _head_to_head_car_paint_color(config=config, entry=entry)
    background = _panda2d_hud_rounded_card(
        parent=parent,
        name="head-to-head-car-label-background",
        position=(0.0, 0.0),
        scale=(0.44, 0.088),
        color=(
            team_color[0],
            team_color[1],
            team_color[2],
            HEAD_TO_HEAD_CAR_LABEL_BACKGROUND_ALPHA,
        ),
        bin_order=100,
    )
    text = _panda2d_hud_text(
        parent=parent,
        name="head-to-head-car-label",
        text=_head_to_head_car_label(config=config, entry=entry),
        position=(0.0, -0.014),
        scale=0.042,
        color=(1.0, 1.0, 1.0, 1.0),
        bin_order=101,
    )
    return HeadToHeadCarLabel(background=background, text=text)


def _update_head_to_head_car_labels(*, ursina: Any, view: CameraView, runtimes: tuple[RaceCarRuntime, ...]) -> None:
    layout = head_to_head_car_label_layout(view)
    for runtime in runtimes:
        if not isinstance(runtime.label, HeadToHeadCarLabel):
            continue
        if runtime.robot.eliminated:
            runtime.label.background.hide()
            runtime.label.text.hide()
            continue
        screen_position = _head_to_head_car_label_screen_position(ursina=ursina, robot=runtime.robot)
        if screen_position is None:
            runtime.label.background.hide()
            runtime.label.text.hide()
            continue
        runtime.label.background.show()
        runtime.label.text.show()
        label_x, label_y = screen_position
        runtime.label.background.setScale(layout.width, layout.height, 1.0)
        text_width = float(runtime.label.text.node().calcWidth(runtime.label.text.node().getText()))
        runtime.label.text.setScale(min(layout.text_scale, (layout.width - 0.03) / max(1.0, text_width)))
        runtime.label.background.setPos(label_x, label_y + layout.vertical_offset, 0.0)
        runtime.label.text.setPos(label_x, label_y + layout.vertical_offset - layout.height * 0.15, 0.0)


def head_to_head_car_label_layout(view: CameraView) -> HeadToHeadCarLabelLayout:
    """Choose a readable floating-label layout for a camera view."""
    if view in (CameraView.TOP_DOWN, CameraView.THREE_QUARTER):
        return HeadToHeadCarLabelLayout(width=0.28, height=0.060, text_scale=0.030, vertical_offset=0.080)
    if view is CameraView.FOLLOW:
        return HeadToHeadCarLabelLayout(width=0.35, height=0.070, text_scale=0.034, vertical_offset=0.160)
    return HeadToHeadCarLabelLayout(width=0.37, height=0.074, text_scale=0.036, vertical_offset=0.100)


def _head_to_head_car_label_screen_position(*, ursina: Any, robot: RobotVehicle) -> tuple[float, float] | None:
    car_position = robot.chassis_np.getPos(ursina.scene)
    camera_relative = ursina.camera.getRelativePoint(
        ursina.scene,
        ursina.Vec3(float(car_position[0]), float(car_position[1]) + 0.95, float(car_position[2])),
    )
    projected = (
        active_scene_camera_lens(ursina)
        .getProjectionMat()
        .xform(ursina.Vec4(float(camera_relative[0]), float(camera_relative[1]), float(camera_relative[2]), 1.0))
    )
    if float(projected[3]) <= 0.0:
        return None
    inverse_w = 1.0 / float(projected[3])
    return (
        float(projected[0]) * inverse_w * float(ursina.camera.aspect_ratio),
        float(projected[1]) * inverse_w,
    )


def active_scene_camera_lens(ursina: Any) -> Any:
    """Return the lens currently installed on Panda's scene camera."""
    return ursina.application.base.cam.node().getLens()


def _head_to_head_car_label(*, config: RaceViewerConfig, entry: RaceViewerEntry) -> str:
    if isinstance(config, HeatViewerConfig):
        assert isinstance(entry, HeatRaceEntry)
        return _short_head_to_head_name(config.entrants[entry.entrant_index].name, max_length=18)
    team_name = config.challenger_name if entry.role == "challenger" else config.incumbent_name
    return f"{_short_head_to_head_name(team_name, max_length=18)} {entry.copy_index + 1}"


def _head_to_head_team_color(*, config: HeadToHeadViewerConfig, role: str) -> ColorRGBA:
    if role == "challenger":
        return config.challenger_team_color
    return config.incumbent_team_color


def _head_to_head_result_color(*, config: HeadToHeadViewerConfig, result: HeadToHeadResult) -> ColorRGBA:
    if result.winner == "tie":
        return (1.0, 1.0, 1.0, 1.0)
    return _head_to_head_team_color(config=config, role=result.winner)


def _head_to_head_car_paint_color(*, config: RaceViewerConfig, entry: RaceViewerEntry) -> ColorRGBA:
    if isinstance(config, HeatViewerConfig):
        assert isinstance(entry, HeatRaceEntry)
        return config.entrants[entry.entrant_index].team_color
    return _head_to_head_team_color(config=config, role=entry.role)


def _short_head_to_head_name(name: str, *, max_length: int) -> str:
    if len(name) <= max_length:
        return name
    if max_length <= 3:
        return name[:max_length]
    return f"{name[: max_length - 3]}..."


def _leader_runtime(runtimes: tuple[RaceCarRuntime, ...]) -> RaceCarRuntime:
    active_runtimes = tuple(runtime for runtime in runtimes if not runtime.robot.eliminated)
    if not active_runtimes:
        return runtimes[0]
    return max(active_runtimes, key=lambda runtime: runtime.tracker.best_distance_m)


def _register_audio_vehicles(*, audio_runtime: RacingAudioRuntimeLike, robots: tuple[RobotVehicle, ...]) -> None:
    for robot in robots:
        audio_runtime.register_vehicle(robot)


def _add_audio_hud_control(*, ursina: Any, audio_runtime: RacingAudioRuntimeLike) -> AudioHudControl | None:
    if not audio_runtime.enabled:
        return None
    parent = ursina.application.base.aspect2d
    _panda2d_hud_card(
        parent=parent,
        name="audio-hud-background",
        position=(1.43, 0.89),
        scale=(0.46, 0.090),
        color=(0.020, 0.023, 0.030, 0.88),
        bin_order=90,
    )
    label = _panda2d_hud_text(
        parent=parent,
        name="audio-hud-label",
        text=audio_runtime.button_text(),
        position=(1.43, 0.872),
        scale=0.043,
        color=(0.96, 0.98, 1.0, 1.0),
        bin_order=91,
    )
    button_color = ursina.color.rgba(0.020, 0.023, 0.030, 0.88)
    button = ursina.Button(
        text="",
        position=(0.76, 0.44),
        scale=(0.22, 0.060),
        color=ursina.color.rgba(0.0, 0.0, 0.0, 0.0),
        highlight_color=button_color.tint(0.18),
        pressed_color=button_color.tint(0.32),
    )
    control = AudioHudControl(button=button, label=label, key_state=AudioKeyToggleState())

    def toggle_audio() -> None:
        """Toggle mute from the on-screen audio button."""
        audio_runtime.toggle_muted()
        _sync_audio_hud_control(audio_control=control, audio_runtime=audio_runtime)

    button.on_click = toggle_audio
    return control


def _update_audio_key_control(
    *, audio_control: AudioHudControl | None, audio_runtime: RacingAudioRuntimeLike, mute_key_down: bool
) -> None:
    if audio_control is None:
        return
    if update_audio_mute_key(audio_control.key_state, mute_key_down=mute_key_down, audio_runtime=audio_runtime):
        _sync_audio_hud_control(audio_control=audio_control, audio_runtime=audio_runtime)


def _sync_audio_hud_control(*, audio_control: AudioHudControl | None, audio_runtime: RacingAudioRuntimeLike) -> None:
    if audio_control is not None:
        _set_panda2d_hud_text(audio_control.label, audio_runtime.button_text())


def damage_hud_layout(count: int) -> tuple[DamageHudSlot, ...]:
    """Place one compact damage bar for each visible car."""
    if count < 0:
        raise ValueError("count cannot be negative")
    if count == 0:
        return ()
    column_count = min(count, DAMAGE_HUD_MAX_COLUMNS)
    slot_width = min(
        DAMAGE_HUD_MAX_WIDTH,
        max(
            DAMAGE_HUD_MIN_WIDTH, (DAMAGE_HUD_USABLE_WIDTH - DAMAGE_HUD_COLUMN_GAP * (column_count - 1)) / column_count
        ),
    )
    slots: list[DamageHudSlot] = []
    for row_index, row_start in enumerate(range(0, count, DAMAGE_HUD_MAX_COLUMNS)):
        row_count = min(DAMAGE_HUD_MAX_COLUMNS, count - row_start)
        row_width = row_count * slot_width + (row_count - 1) * DAMAGE_HUD_COLUMN_GAP
        first_center_x = -row_width / 2.0 + slot_width / 2.0
        center_y = DAMAGE_HUD_BOTTOM_Y + row_index * DAMAGE_HUD_ROW_SPACING
        for column_index in range(row_count):
            slots.append(
                DamageHudSlot(
                    center_x=first_center_x + column_index * (slot_width + DAMAGE_HUD_COLUMN_GAP),
                    center_y=center_y,
                    width=slot_width,
                    height=DAMAGE_HUD_HEIGHT,
                )
            )
    return tuple(slots)


def _add_damage_hud_bars(*, ursina: Any, colors: tuple[ColorRGBA, ...]) -> tuple[DamageHudBar, ...]:
    parent = ursina.application.base.aspect2d
    bars: list[DamageHudBar] = []
    for slot, color in zip(damage_hud_layout(len(colors)), colors, strict=True):
        shadow = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-shadow",
            position=(slot.center_x, slot.center_y - 0.006),
            scale=(slot.width + 0.050, slot.height + 0.030),
            color=DAMAGE_HUD_SHADOW_COLOR,
            bin_order=80,
        )
        frame = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-frame",
            position=(slot.center_x, slot.center_y),
            scale=(slot.width + 0.026, slot.height + 0.018),
            color=DAMAGE_HUD_FRAME_COLOR,
            bin_order=81,
        )
        track = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-track",
            position=(slot.center_x, slot.center_y),
            scale=(slot.width, slot.height),
            color=DAMAGE_HUD_TRACK_COLOR,
            bin_order=82,
        )
        fill = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-fill",
            position=(slot.center_x, slot.center_y),
            scale=(DAMAGE_HUD_EMPTY_WIDTH, slot.height - 0.018),
            color=DAMAGE_HUD_ZERO_FILL_COLOR,
            bin_order=83,
        )
        fill.hide()
        cap = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-cap",
            position=(slot.center_x, slot.center_y),
            scale=(0.010, slot.height - 0.010),
            color=(1.0, 1.0, 1.0, 0.88),
            bin_order=84,
        )
        cap.hide()
        accent = _panda2d_hud_card(
            parent=parent,
            name="damage-hud-accent",
            position=(slot.center_x - slot.width / 2.0 - 0.026, slot.center_y),
            scale=(0.018, slot.height + 0.020),
            color=color,
            bin_order=85,
        )
        bar = DamageHudBar(slot=slot, shadow=shadow, frame=frame, track=track, fill=fill, cap=cap, accent=accent)
        _style_damage_hud_bar(bar=bar, color=color)
        bars.append(bar)
    return tuple(bars)


def _move_damage_hud_bar(*, bar: DamageHudBar, center_x: float) -> None:
    """Move a whole damage bar to keep it aligned with its team's viewport."""
    offset = center_x - bar.slot.center_x
    if abs(offset) < 1e-9:
        return
    bar.slot = replace(bar.slot, center_x=center_x)
    for node in (bar.shadow, bar.frame, bar.track, bar.fill, bar.cap, bar.accent, bar.label):
        if node is not None:
            node.setX(float(node.getX()) + offset)


def _add_head_to_head_damage_hud_labels(
    *,
    ursina: Any,
    bars: tuple[DamageHudBar, ...],
    config: RaceViewerConfig,
    entries: tuple[RaceViewerEntry, ...],
) -> None:
    parent = ursina.application.base.aspect2d
    for bar, entry in zip(bars, entries, strict=True):
        bar.label = _panda2d_hud_text(
            parent=parent,
            name="head-to-head-damage-label",
            text=head_to_head_damage_hud_text(config=config, entry=entry, distance_m=0.0),
            position=(bar.slot.center_x, bar.slot.center_y + DAMAGE_HUD_LABEL_OFFSET_Y),
            scale=DAMAGE_HUD_LABEL_SCALE,
            color=(0.96, 0.98, 1.0, 1.0),
            bin_order=86,
        )


def _update_head_to_head_damage_hud_label(
    *,
    bar: DamageHudBar,
    config: RaceViewerConfig,
    entry: RaceViewerEntry,
    runtime: RaceCarRuntime,
) -> None:
    if bar.label is None:
        return
    _set_panda2d_hud_text(
        bar.label,
        head_to_head_damage_hud_text(
            config=config,
            entry=entry,
            distance_m=_head_to_head_runtime_score(runtime),
        ),
    )


def head_to_head_damage_hud_text(*, config: RaceViewerConfig, entry: RaceViewerEntry, distance_m: float) -> str:
    """Format the compact label positioned above one car's damage bar."""
    return f"{_head_to_head_car_label(config=config, entry=entry)}  {distance_m:.1f} m"


def _style_damage_hud_bar(*, bar: DamageHudBar, color: ColorRGBA) -> None:
    bar.accent.setColor(*color)
    bar.shadow.setColor(*DAMAGE_HUD_SHADOW_COLOR)
    bar.frame.setColor(*DAMAGE_HUD_FRAME_COLOR)
    bar.track.setColor(*DAMAGE_HUD_TRACK_COLOR)


def _update_damage_hud_bars(*, bars: tuple[DamageHudBar, ...], robots: tuple[RobotVehicle, ...]) -> None:
    for bar, robot in zip(bars, robots, strict=True):
        damage = 1.0 if robot.eliminated else _clamp01(robot.damage)
        fill_width = max(DAMAGE_HUD_EMPTY_WIDTH, (bar.slot.width - 0.024) * damage)
        fill_left_x = bar.slot.center_x - (bar.slot.width - 0.024) / 2.0
        if damage > 0.0 or robot.eliminated:
            bar.fill.show()
            bar.cap.show()
        else:
            bar.fill.hide()
            bar.cap.hide()
        bar.fill.setScale(fill_width, bar.slot.height - 0.018, 1.0)
        bar.fill.setPos(fill_left_x + fill_width / 2.0, bar.slot.center_y, 0.0)
        bar.fill.setColor(*damage_hud_fill_color(damage=damage, eliminated=robot.eliminated))
        bar.cap.setPos(fill_left_x + fill_width, bar.slot.center_y, 0.0)
        bar.cap.setColor(*(1.0, 0.94, 0.80, 0.95) if not robot.eliminated else (0.18, 0.0, 0.0, 1.0))
        bar.frame.setColor(*(DAMAGE_HUD_INNER_FRAME_COLOR if robot.eliminated else DAMAGE_HUD_FRAME_COLOR))


def _panda2d_hud_card(
    *,
    parent: Any,
    name: str,
    position: tuple[float, float],
    scale: tuple[float, float],
    color: ColorRGBA,
    bin_order: int,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    card_maker = core.CardMaker(name)
    card_maker.setFrame(-0.5, 0.5, -0.5, 0.5)
    card = parent.attachNewNode(card_maker.generate())
    card.setPos(position[0], position[1], 0.0)
    card.setScale(scale[0], scale[1], 1.0)
    card.setColor(*color)
    card.setTransparency(core.TransparencyAttrib.MAlpha)
    card.setDepthTest(False)
    card.setDepthWrite(False)
    card.setBin("fixed", bin_order)
    card.setLightOff(1)
    return card


def _panda2d_hud_rounded_card(
    *,
    parent: Any,
    name: str,
    position: tuple[float, float],
    scale: tuple[float, float],
    color: ColorRGBA,
    bin_order: int,
    corner_radius_fraction: float = 0.28,
    corner_segments: int = 6,
) -> Any:
    """Create an antialiased-looking rounded rectangle in the native 2D HUD."""
    core = cast(Any, import_module("panda3d.core"))
    radius_y = min(0.5, max(0.0, corner_radius_fraction))
    radius_x = min(0.5, radius_y * scale[1] / scale[0]) if scale[0] > 0.0 else 0.0
    corners = (
        (0.5 - radius_x, -0.5 + radius_y, -pi / 2.0),
        (0.5 - radius_x, 0.5 - radius_y, 0.0),
        (-0.5 + radius_x, 0.5 - radius_y, pi / 2.0),
        (-0.5 + radius_x, -0.5 + radius_y, pi),
    )
    boundary: list[tuple[float, float]] = []
    for center_x, center_y, start_angle in corners:
        for segment_index in range(corner_segments + 1):
            angle = start_angle + pi / 2.0 * segment_index / corner_segments
            boundary.append((center_x + cos(angle) * radius_x, center_y + sin(angle) * radius_y))

    vertex_data = core.GeomVertexData(name, core.GeomVertexFormat.getV3(), core.Geom.UHStatic)
    vertex_data.setNumRows(len(boundary) + 1)
    vertex_writer = core.GeomVertexWriter(vertex_data, "vertex")
    vertex_writer.addData3f(0.0, 0.0, 0.0)
    for x, y in boundary:
        vertex_writer.addData3f(x, y, 0.0)

    triangles = core.GeomTriangles(core.Geom.UHStatic)
    for boundary_index in range(len(boundary)):
        triangles.addVertices(0, boundary_index + 1, (boundary_index + 1) % len(boundary) + 1)
    geometry = core.Geom(vertex_data)
    geometry.addPrimitive(triangles)
    geometry_node = core.GeomNode(name)
    geometry_node.addGeom(geometry)
    card = parent.attachNewNode(geometry_node)
    card.setPos(position[0], position[1], 0.0)
    card.setScale(scale[0], scale[1], 1.0)
    card.setColor(*color)
    card.setTransparency(core.TransparencyAttrib.MAlpha)
    card.setTwoSided(True)
    card.setDepthTest(False)
    card.setDepthWrite(False)
    card.setBin("fixed", bin_order)
    card.setLightOff(1)
    return card


def _panda2d_hud_text(
    *,
    parent: Any,
    name: str,
    text: str,
    position: tuple[float, float],
    scale: float,
    color: ColorRGBA,
    bin_order: int,
    align_left: bool = False,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))
    text_node = core.TextNode(name)
    text_node.setText(text)
    text_node.setAlign(core.TextNode.ALeft if align_left else core.TextNode.ACenter)
    text_node.setTextColor(*color)
    text_path = parent.attachNewNode(text_node)
    text_path.setPos(position[0], position[1], 0.0)
    text_path.setScale(scale)
    text_path.setDepthTest(False)
    text_path.setDepthWrite(False)
    text_path.setBin("fixed", bin_order)
    text_path.setLightOff(1)
    return text_path


def _set_panda2d_hud_text(text_path: Any, text: str) -> None:
    node = text_path.node()
    if hasattr(node, "setText"):
        node.setText(text)


def _set_panda2d_hud_text_color(text_path: Any, color: ColorRGBA) -> None:
    node = text_path.node()
    if hasattr(node, "setTextColor"):
        node.setTextColor(*color)


def damage_hud_fill_color(*, damage: float, eliminated: bool) -> ColorRGBA:
    """Choose the damage bar color for a car's current damage state."""
    if eliminated:
        return DAMAGE_HUD_ELIMINATED_FILL_COLOR
    normalized_damage = _clamp01(damage)
    if normalized_damage <= 0.5:
        return _interpolate_color(DAMAGE_HUD_ZERO_FILL_COLOR, DAMAGE_HUD_MID_FILL_COLOR, normalized_damage * 2.0)
    return _interpolate_color(DAMAGE_HUD_MID_FILL_COLOR, DAMAGE_HUD_FULL_FILL_COLOR, (normalized_damage - 0.5) * 2.0)


def _interpolate_color(start: ColorRGBA, end: ColorRGBA, amount: float) -> ColorRGBA:
    clamped_amount = _clamp01(amount)
    return (
        start[0] + (end[0] - start[0]) * clamped_amount,
        start[1] + (end[1] - start[1]) * clamped_amount,
        start[2] + (end[2] - start[2]) * clamped_amount,
        start[3] + (end[3] - start[3]) * clamped_amount,
    )


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _update_head_to_head_hud(
    *,
    status_display: Any,
    damage_bars: tuple[DamageHudBar, ...],
    config: RaceViewerConfig,
    race_index: int,
    entries: tuple[RaceViewerEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
    race_elapsed_seconds: float,
) -> None:
    time_left = max(0.0, config.round_seconds - race_elapsed_seconds)
    _set_panda2d_hud_text(
        status_display,
        f"Race {race_index}/{config.race_count}   Time {time_left:04.1f}s",
    )
    for bar, entry, runtime in zip(damage_bars, entries, runtimes, strict=True):
        _update_head_to_head_damage_hud_label(bar=bar, config=config, entry=entry, runtime=runtime)


def _head_to_head_runtime_score(runtime: RaceCarRuntime) -> float:
    return race_scored_distance_m(runtime)


def _head_to_head_race_result_from_runtimes(
    *,
    config: HeadToHeadViewerConfig,
    race_rules: HeadToHeadRaceRules,
    race_index: int,
    entries: tuple[HeadToHeadRaceEntry, ...],
    runtimes: tuple[RaceCarRuntime, ...],
) -> HeadToHeadRaceResult:
    challenger_stats = head_to_head_team_stats_from_runtimes(entries=entries, runtimes=runtimes, role="challenger")
    incumbent_stats = head_to_head_team_stats_from_runtimes(entries=entries, runtimes=runtimes, role="incumbent")
    winner = classify_head_to_head_winner(
        margin_m=head_to_head_race_margin(
            challenger=challenger_stats, incumbent=incumbent_stats, scoring=race_rules.scoring
        ),
        win_margin_m=race_rules.win_margin_m,
    )
    return HeadToHeadRaceResult(
        race_index=race_index,
        winner=winner,
        challenger=challenger_stats,
        incumbent=incumbent_stats,
        scoring=race_rules.scoring,
    )


def create_app(config: GameConfig | None = None) -> RunnableApp:
    """Create the normal playable simulator app."""
    return build_scene(GameConfig() if config is None else config)


def create_head_to_head_viewer_app(config: HeadToHeadViewerConfig | None = None) -> RunnableApp:
    """Create the app that watches two controller teams race."""
    return build_head_to_head_viewer_scene(HeadToHeadViewerConfig() if config is None else config)


def create_car_showcase_app(config: CarShowcaseConfig | None = None) -> RunnableApp:
    """Create the small scene used to inspect the car model."""
    showcase_config = CarShowcaseConfig() if config is None else config
    ursina, app = _create_configured_ursina_app(
        app_kwargs={
            "title": showcase_config.title,
            "borderless": False,
            "fullscreen": False,
            "vsync": False,
            "development_mode": showcase_config.development_mode,
            "editor_ui_enabled": False,
            "size": showcase_config.size,
            "window_type": showcase_config.window_type,
        },
        preserve_project_y_up=False,
    )
    app.setBackgroundColor(0.96, 0.96, 0.94, 1)

    assets = create_scene_assets()
    add_showcase_floor(ursina=ursina)
    add_showcase_lighting(ursina)

    robot = create_showcase_robot(ursina, config=FORMULA_VEHICLE_PHYSICS_CONFIG)
    add_robot_visuals(ursina=ursina, robot=robot, assets=assets, team_color=showcase_config.team_color)
    pose_showcase_car(robot.chassis_np, showcase_config.view)
    apply_showcase_camera(ursina=ursina, view=showcase_config.view)
    return cast(RunnableApp, app)
