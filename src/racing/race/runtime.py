"""Shared race runtime helpers for student and manual cars."""

from __future__ import annotations

from dataclasses import dataclass, field
from importlib import import_module
from math import ceil
from random import Random
from typing import Any, cast

from racing.graphics.track_rendering import TRACK_EDGE_BUFFER, TRACK_SURFACE_Y
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    RobotVehicle,
    VehiclePhysicsConfig,
    apply_vehicle_command,
    restore_robot_vehicle,
    vehicle_collision_bounds,
    vehicle_reset_pose_is_clear,
    vehicle_spawn_height,
)
from racing.race.progress import (
    LapProgressTracker,
    TrackPose,
    TrackProgressModel,
    TrackProjection,
    heading_error_degrees,
    project_track_position,
    track_pose_at_distance,
)
from racing.race.sensors import RobotSensorBuilderState
from racing.student.api import RobotCommand
from racing.track.spatial import node_position, track_left_vector
from racing.track.world import TRACK_WIDTH, TrackPoint

DEFAULT_RACE_RANDOM_SEED = 110
RACE_SPAWN_RANDOM_ROUND_FACTOR = 130_363
RACE_SPAWN_RANDOM_OFFSET = 41_947
RACE_SPAWN_EDGE_MARGIN = 0.18 * 2.0
RACE_GRID_LONGITUDINAL_SPACING_CAR_LENGTHS = 2.25
RACE_GRID_LANE_OFFSET_FRACTION = 0.5
RACE_GRID_CURVE_DIRECTION_THRESHOLD_DEGREES = 0.5
RACE_MARSHAL_RESET_LANE_OFFSET_FRACTION = 0.5
RACE_MARSHAL_RESET_LONGITUDINAL_SPACING_CAR_LENGTHS = 1.5
RACE_MARSHAL_RESET_SEARCH_ROWS = 8
RACE_OFF_TRACK_RESET_DISTANCE_M = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER
RACE_MARSHAL_CAR_CONTACT_MAX_SPEED_MPS = 3.0 * 0.44704
RACE_START_FINISH_AHEAD_CAR_LENGTHS = 2.0


@dataclass(frozen=True, slots=True)
class RaceSpawnPose:
    """World spawn pose for one race car."""

    position: tuple[float, float, float]
    heading_degrees: float
    progress_distance_m: float


@dataclass(frozen=True, slots=True)
class RaceContactState:
    """Contact state sampled for one race car."""

    wall_contact: float = 0.0
    car_contact: float = 0.0


@dataclass(frozen=True, slots=True)
class RaceRecoveryConfig:
    """Deterministic marshal settings for competitive races."""

    stuck_seconds: float
    distance_penalty_m: float
    cooldown_seconds: float

    def __post_init__(self) -> None:
        if self.stuck_seconds < 0.0:
            raise ValueError("stuck_seconds cannot be negative")
        if self.distance_penalty_m < 0.0:
            raise ValueError("distance_penalty_m cannot be negative")
        if self.cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds cannot be negative")


@dataclass(slots=True)
class RaceCarRuntime:
    """Mutable runtime state for one car in a race."""

    robot: RobotVehicle
    tracker: LapProgressTracker
    label: Any | None = None
    sensor_state: RobotSensorBuilderState = field(default_factory=RobotSensorBuilderState)
    stuck_seconds: float = 0.0
    low_progress_seconds: float = 0.0
    off_track_seconds: float = 0.0
    recent_progress_mps: float = 0.0
    max_speed_mps: float = 0.0
    contact_state: RaceContactState = RaceContactState()
    marshal_count: int = 0
    marshal_penalty_m: float = 0.0
    marshal_cooldown_seconds: float = 0.0


def race_spawn_poses(
    car_count: int,
    *,
    model: TrackProgressModel,
    config: VehiclePhysicsConfig = FORMULA_VEHICLE_PHYSICS_CONFIG,
    random_seed: int = DEFAULT_RACE_RANDOM_SEED,
    race_index: int = 1,
    shuffle_grid: bool = True,
) -> tuple[RaceSpawnPose, ...]:
    """Create seeded grid poses, optionally ordered from pole to the back."""
    if car_count < 1:
        raise ValueError("car_count must be at least one")
    if race_index < 1:
        raise ValueError("race_index must be at least one")

    rng = _race_spawn_rng(random_seed=random_seed, race_index=race_index)
    shared_progress = _race_grid_front_progress_m(model=model, rng=rng)
    safe_half_width = max(
        0.0,
        TRACK_WIDTH / 2
        - max(vehicle_collision_bounds(config).half_width, config.wheel_track_half_width + config.wheel_width / 2)
        - RACE_SPAWN_EDGE_MARGIN,
    )
    spawn_y = vehicle_spawn_height(config, surface_y=TRACK_SURFACE_Y)

    if car_count == 1:
        track_pose = track_pose_at_distance(model, shared_progress)
        return (_race_spawn_pose_at_grid_slot(track_pose=track_pose, lateral_offset=0.0, spawn_y=spawn_y),)

    car_length_m = vehicle_collision_bounds(config).half_length * 2.0
    longitudinal_spacing_m = car_length_m * RACE_GRID_LONGITUDINAL_SPACING_CAR_LENGTHS
    lane_offset_m = safe_half_width * RACE_GRID_LANE_OFFSET_FRACTION
    inside_lateral_sign = _race_inside_lateral_sign(
        model=model,
        progress_distance_m=shared_progress,
        sample_spacing_m=car_length_m,
    )
    outside_lateral_sign = -inside_lateral_sign
    grid_slots = [
        (
            shared_progress - slot_index * longitudinal_spacing_m,
            (outside_lateral_sign if slot_index % 2 == 0 else inside_lateral_sign) * lane_offset_m,
        )
        for slot_index in range(car_count)
    ]
    if shuffle_grid:
        rng.shuffle(grid_slots)
    return tuple(
        _race_spawn_pose_at_grid_slot(
            track_pose=track_pose_at_distance(model, progress_distance_m),
            lateral_offset=lateral_offset_m,
            spawn_y=spawn_y,
        )
        for progress_distance_m, lateral_offset_m in grid_slots
    )


def start_finish_pose_for_progress(
    *,
    model: TrackProgressModel,
    start_progress_distance_m: float,
    config: VehiclePhysicsConfig = FORMULA_VEHICLE_PHYSICS_CONFIG,
    ahead_car_lengths: float = RACE_START_FINISH_AHEAD_CAR_LENGTHS,
) -> TrackPose:
    """Place the start/finish line ahead of the race grid."""
    if ahead_car_lengths < 0.0:
        raise ValueError("ahead_car_lengths cannot be negative")
    car_length_m = vehicle_collision_bounds(config).half_length * 2.0
    return track_pose_at_distance(model, start_progress_distance_m + car_length_m * ahead_car_lengths)


def seeded_race_start_finish_pose(
    *,
    model: TrackProgressModel,
    config: VehiclePhysicsConfig = FORMULA_VEHICLE_PHYSICS_CONFIG,
    random_seed: int = DEFAULT_RACE_RANDOM_SEED,
    race_index: int = 1,
) -> TrackPose:
    """Choose a repeatable start/finish line for a seeded race."""
    if race_index < 1:
        raise ValueError("race_index must be at least one")
    rng = _race_spawn_rng(random_seed=random_seed, race_index=race_index)
    shared_progress = _race_grid_front_progress_m(model=model, rng=rng)
    return start_finish_pose_for_progress(
        model=model,
        start_progress_distance_m=shared_progress,
        config=config,
    )


def lap_progress_tracker_for_spawn_pose(*, model: TrackProgressModel, spawn_pose: RaceSpawnPose) -> LapProgressTracker:
    """Create lap bookkeeping for a car at its spawn pose."""
    return LapProgressTracker(
        total_length_m=model.total_length_m, starting_progress_distance_m=spawn_pose.progress_distance_m
    )


def race_contact_states(*, physics_world: Any, runtimes: tuple[RaceCarRuntime, ...]) -> tuple[RaceContactState, ...]:
    """Read wall and car contact state for each active race car."""
    car_node_names = {
        _node_name(runtime.robot.chassis_np.node()): index
        for index, runtime in enumerate(runtimes)
        if not robot_is_eliminated(runtime.robot)
    }
    states: list[RaceContactState] = []
    for index, runtime in enumerate(runtimes):
        if robot_is_eliminated(runtime.robot):
            states.append(RaceContactState())
            continue
        wall_contact = False
        car_contact = False
        contact_result = physics_world.contactTest(runtime.robot.chassis_np.node())
        for contact in contact_result.getContacts():
            node_names = (_node_name(contact.getNode0()), _node_name(contact.getNode1()))
            wall_contact = wall_contact or any(name.startswith("track-barrier") for name in node_names)
            car_contact = car_contact or any(
                name in car_node_names and car_node_names[name] != index for name in node_names
            )
        states.append(
            RaceContactState(wall_contact=1.0 if wall_contact else 0.0, car_contact=1.0 if car_contact else 0.0)
        )
    return tuple(states)


def update_race_runtime_after_step(
    *,
    runtime: RaceCarRuntime,
    projection: TrackProjection,
    contact_state: RaceContactState,
    elapsed_seconds: float,
    delta_seconds: float,
) -> None:
    """Record progress, contacts, speed, damage, and lap data after one tick."""
    contact_delta_seconds = max(0.0, delta_seconds)
    current_speed_mps = abs(float(runtime.robot.vehicle.getCurrentSpeedKmHour()) / 3.6)
    runtime.max_speed_mps = max(runtime.max_speed_mps, current_speed_mps)
    runtime.contact_state = RaceContactState(
        wall_contact=runtime.contact_state.wall_contact + contact_delta_seconds
        if contact_state.wall_contact > 0.0
        else 0.0,
        car_contact=runtime.contact_state.car_contact + contact_delta_seconds
        if contact_state.car_contact > 0.0
        else 0.0,
    )
    runtime.tracker.update(
        projection.progress_distance_m,
        elapsed_seconds,
        wall_contact=runtime.contact_state.wall_contact > 0.0,
        car_contact=runtime.contact_state.car_contact > 0.0,
    )
    runtime.recent_progress_mps = (
        runtime.tracker.last_counted_progress_delta_m / delta_seconds if delta_seconds > 0.0 else 0.0
    )
    if not robot_is_eliminated(runtime.robot) and _track_projection_is_outside_drivable_surface(projection):
        runtime.off_track_seconds += delta_seconds
    if _race_runtime_is_stuck(runtime):
        runtime.stuck_seconds += delta_seconds
        runtime.low_progress_seconds += delta_seconds
    else:
        runtime.stuck_seconds = max(0.0, runtime.stuck_seconds - delta_seconds * 2.0)


def maybe_marshal_race_runtimes(
    *,
    model: TrackProgressModel,
    runtimes: tuple[RaceCarRuntime, ...],
    projections: tuple[TrackProjection, ...],
    recovery_config: RaceRecoveryConfig,
    delta_seconds: float,
) -> int:
    """Marshal cars into clear track space, retrying later if it is occupied."""
    if len(runtimes) != len(projections):
        raise ValueError("runtimes and projections must have the same length")
    marshal_count = 0
    for runtime_index, (runtime, projection) in enumerate(zip(runtimes, projections, strict=True)):
        if robot_is_eliminated(runtime.robot):
            continue
        runtime.marshal_cooldown_seconds = max(0.0, runtime.marshal_cooldown_seconds - delta_seconds)
        off_track = _track_projection_is_off_track(projection)
        if runtime.marshal_cooldown_seconds > 0.0 and not off_track:
            continue
        if runtime.stuck_seconds < recovery_config.stuck_seconds and not off_track:
            continue

        reset_pose = _marshal_reset_pose(
            model=model,
            runtime=runtime,
            projection=projection,
            runtime_index=runtime_index,
            runtimes=runtimes,
        )
        if reset_pose is None:
            continue
        reset_robot_vehicle(runtime.robot, position=reset_pose.position, heading_degrees=reset_pose.heading_degrees)
        runtime.stuck_seconds = 0.0
        runtime.recent_progress_mps = 0.0
        runtime.contact_state = RaceContactState()
        runtime.sensor_state = sensor_state_after_runtime_reset(
            runtime=runtime,
            projection=project_track_position(model, TrackPoint(reset_pose.position[0], reset_pose.position[2])),
            time_s=runtime.sensor_state.time_s,
        )
        runtime.marshal_count += 1
        runtime.marshal_penalty_m += recovery_config.distance_penalty_m
        runtime.marshal_cooldown_seconds = recovery_config.cooldown_seconds
        marshal_count += 1
    return marshal_count


def sensor_state_after_runtime_reset(
    *,
    runtime: RaceCarRuntime,
    projection: TrackProjection,
    time_s: float,
) -> RobotSensorBuilderState:
    """Reset sensor bookkeeping after the marshal moves a car."""
    return RobotSensorBuilderState(
        time_s=time_s,
        position=projection.position,
        heading_degrees=projection.heading_degrees,
        speed_mps=0.0,
        distance_m=runtime.sensor_state.distance_m,
        tick=runtime.sensor_state.tick,
    )


def robot_track_point(robot: RobotVehicle) -> TrackPoint:
    """Return the robot's X/Z track point."""
    x, _, z = node_position(robot.chassis_np)
    return TrackPoint(x=x, z=z)


def robot_score_damage(robot: RobotVehicle) -> float:
    """Return clamped damage for race stats and the damage HUD."""
    if robot_is_eliminated(robot):
        return 1.0
    return min(1.0, max(0.0, float(getattr(robot, "damage", 0.0))))


def race_scored_distance_m(runtime: RaceCarRuntime) -> float:
    """Return distance progress after marshal penalties, independent of damage."""
    return max(0.0, runtime.tracker.best_distance_m - runtime.marshal_penalty_m)


def race_track_position_m(runtime: RaceCarRuntime, *, start_finish_progress_m: float) -> float:
    """Current lap-aware position from the shared line, including the grid offset.

    Cars start a negative distance behind the line. Signed progress preserves
    reversing and completed laps instead of measuring each car's best journey.
    """
    tracker = runtime.tracker
    if tracker.starting_progress_distance_m is None:
        raise ValueError("track position requires an initialized starting progress")
    grid_offset = -((start_finish_progress_m - tracker.starting_progress_distance_m) % tracker.total_length_m)
    return grid_offset + tracker.unwrapped_progress_distance_m


def robot_is_eliminated(robot: RobotVehicle) -> bool:
    """Return whether a robot has been removed from the active race."""
    return bool(getattr(robot, "eliminated", False))


def reset_robot_vehicle(
    robot: RobotVehicle,
    *,
    position: tuple[float, float, float],
    heading_degrees: float,
    pitch_degrees: float = 0.0,
    roll_degrees: float = 0.0,
    reset_damage: bool = False,
) -> None:
    """Reset a robot vehicle pose and clear body motion."""
    core = cast(Any, import_module("panda3d.core"))
    if reset_damage:
        restore_robot_vehicle(robot)
    elif robot_is_eliminated(robot):
        return
    robot.chassis_np.setPos(*position)
    robot.chassis_np.setHpr(heading_degrees, pitch_degrees, roll_degrees)
    body = robot.chassis_np.node()
    if hasattr(body, "setLinearVelocity"):
        body.setLinearVelocity(core.Vec3(0.0, 0.0, 0.0))
    if hasattr(body, "setAngularVelocity"):
        body.setAngularVelocity(core.Vec3(0.0, 0.0, 0.0))
    if hasattr(body, "clearForces"):
        body.clearForces()
    if hasattr(body, "resetPrevTransform"):
        body.resetPrevTransform()
    if hasattr(body, "setActive"):
        body.setActive(True)
    if hasattr(robot.vehicle, "resetSuspension"):
        robot.vehicle.resetSuspension()
    robot.pending_drive_direction = 0
    robot.pre_step_linear_velocity_mps = None
    apply_vehicle_command(vehicle=robot.vehicle, command=RobotCommand(), config=robot.config)


def _race_spawn_rng(*, random_seed: int, race_index: int) -> Random:
    return Random(random_seed + race_index * RACE_SPAWN_RANDOM_ROUND_FACTOR + RACE_SPAWN_RANDOM_OFFSET)


def _race_grid_front_progress_m(*, model: TrackProgressModel, rng: Random) -> float:
    return rng.random() * model.total_length_m


def _race_inside_lateral_sign(
    *,
    model: TrackProgressModel,
    progress_distance_m: float,
    sample_spacing_m: float,
) -> int:
    """Return -1 for a right-side apex and +1 for a left-side apex."""
    if sample_spacing_m <= 0.0:
        raise ValueError("sample_spacing_m must be positive")
    previous_heading = track_pose_at_distance(model, progress_distance_m).heading_degrees
    sample_count = max(1, ceil(model.total_length_m / sample_spacing_m))
    for sample_index in range(1, sample_count + 1):
        next_heading = track_pose_at_distance(
            model,
            progress_distance_m + sample_index * sample_spacing_m,
        ).heading_degrees
        turn_degrees = heading_error_degrees(
            current_heading_degrees=previous_heading,
            target_heading_degrees=next_heading,
        )
        if abs(turn_degrees) >= RACE_GRID_CURVE_DIRECTION_THRESHOLD_DEGREES:
            return -1 if turn_degrees > 0.0 else 1
        previous_heading = next_heading
    return 1


def _race_spawn_pose_at_grid_slot(*, track_pose: TrackPose, lateral_offset: float, spawn_y: float) -> RaceSpawnPose:
    left_x, left_z = track_left_vector(track_pose.heading_degrees)
    return RaceSpawnPose(
        position=(
            track_pose.position.x + left_x * lateral_offset,
            spawn_y,
            track_pose.position.z + left_z * lateral_offset,
        ),
        heading_degrees=track_pose.heading_degrees,
        progress_distance_m=track_pose.progress_distance_m,
    )


def _marshal_reset_pose(
    *,
    model: TrackProgressModel,
    runtime: RaceCarRuntime,
    projection: TrackProjection,
    runtime_index: int,
    runtimes: tuple[RaceCarRuntime, ...],
) -> RaceSpawnPose | None:
    config = runtime.robot.config
    spawn_y = vehicle_spawn_height(config, surface_y=TRACK_SURFACE_Y)
    safe_half_width = max(
        0.0,
        TRACK_WIDTH / 2
        - max(vehicle_collision_bounds(config).half_width, config.wheel_track_half_width + config.wheel_width / 2)
        - RACE_SPAWN_EDGE_MARGIN,
    )
    side = -1.0 if runtime_index % 2 == 0 else 1.0
    lateral_offset_m = side * safe_half_width * RACE_MARSHAL_RESET_LANE_OFFSET_FRACTION
    lateral_offsets = (lateral_offset_m, -lateral_offset_m, 0.0)
    if len(runtimes) == 1:
        lateral_offsets = (0.0, lateral_offset_m, -lateral_offset_m)
    car_length_m = vehicle_collision_bounds(config).half_length * 2.0
    spacing_m = car_length_m * RACE_MARSHAL_RESET_LONGITUDINAL_SPACING_CAR_LENGTHS
    other_robots = tuple(other.robot for other in runtimes if other is not runtime)
    for row_index in range(RACE_MARSHAL_RESET_SEARCH_ROWS):
        setback_m = row_index * spacing_m
        # Search backward along the centerline, including bends and the wrapped
        # start line. Keep the move local so lap bookkeeping sees a backward move.
        if setback_m > model.total_length_m / 4:
            break
        track_pose = track_pose_at_distance(model, projection.progress_distance_m - setback_m)
        for lateral_offset in lateral_offsets:
            pose = _race_spawn_pose_at_grid_slot(track_pose=track_pose, lateral_offset=lateral_offset, spawn_y=spawn_y)
            if vehicle_reset_pose_is_clear(
                robot=runtime.robot,
                position=pose.position,
                heading_degrees=pose.heading_degrees,
                other_robots=other_robots,
            ):
                return pose
    return None


def _track_projection_is_off_track(projection: TrackProjection) -> bool:
    return abs(projection.signed_distance_to_center_m) > RACE_OFF_TRACK_RESET_DISTANCE_M


def _track_projection_is_outside_drivable_surface(projection: TrackProjection) -> bool:
    return abs(projection.signed_distance_to_center_m) > TRACK_WIDTH / 2


def _race_runtime_is_stuck(runtime: RaceCarRuntime) -> bool:
    if robot_is_eliminated(runtime.robot):
        return False
    if runtime.contact_state.wall_contact:
        return True
    speed_mps = abs(float(runtime.robot.vehicle.getCurrentSpeedKmHour()) / 3.6)
    if runtime.contact_state.car_contact and speed_mps <= RACE_MARSHAL_CAR_CONTACT_MAX_SPEED_MPS:
        return True
    if speed_mps < 0.25:
        return True
    return runtime.recent_progress_mps < 0.04 and speed_mps < 1.20


def _node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""
