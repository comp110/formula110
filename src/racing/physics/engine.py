"""Bullet physics setup for cars, walls, collisions, and damage."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from racing.student.api import RobotCommand, clamp_command

REFERENCE_VEHICLE_WHEEL_RADIUS = 0.72
REFERENCE_VEHICLE_LOWER_CHASSIS_HALF_WIDTH = 0.54
DEFAULT_VEHICLE_SPAWN_SURFACE_Y = 0.0
RC_CAR_LINEAR_SCALE = 0.5
RC_CAR_VOLUME_SCALE = RC_CAR_LINEAR_SCALE**3
RC_CAR_SUSPENSION_STIFFNESS = 100.0
RC_CAR_WHEEL_CONNECTION_HEIGHT = 0.38
RC_CAR_WHEEL_RADIUS = REFERENCE_VEHICLE_WHEEL_RADIUS * RC_CAR_LINEAR_SCALE * 0.80
WALL_DAMAGE_FULL_IMPACT_SPEED_KMH = 100.0
KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND = 1.0 / 3.6
WALL_DAMAGE_MIN_CLOSING_SPEED_MPS = 0.25
VEHICLE_IMPACT_RESPONSE_MULTIPLIER = 2.0
VEHICLE_IMPACT_MIN_CLOSING_SPEED_MPS = 0.25
# With forward along +Z, Bullet's positive wheel rotation needs a -X axle so
# the top of the rendered wheel travels forward rather than backward.
WHEEL_AXLE_DIRECTION = (-1.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class VehicleCollisionBox:
    """One box-shaped part of a car's collision body."""

    half_extents: tuple[float, float, float]
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True, slots=True)
class VehicleCollisionHull:
    """One convex hull part of a car's collision body."""

    footprint: tuple[tuple[float, float], ...]
    min_y: float
    max_y: float
    margin: float = 0.0

    @property
    def points(self) -> tuple[tuple[float, float, float], ...]:
        """List the bottom and top points passed to Bullet."""
        return tuple((x, y, z) for y in (self.min_y, self.max_y) for x, z in self.footprint)


@dataclass(frozen=True, slots=True)
class VehicleCollisionBounds:
    """Simple min/max dimensions for a car's collision body."""

    min_x: float
    max_x: float
    min_y: float
    max_y: float
    min_z: float
    max_z: float

    @property
    def half_width(self) -> float:
        """Measure how far the collider reaches left or right."""
        return max(abs(self.min_x), abs(self.max_x))

    @property
    def half_length(self) -> float:
        """Measure how far the collider reaches forward or backward."""
        return max(abs(self.min_z), abs(self.max_z))

    @property
    def height(self) -> float:
        """Measure the collider from bottom to top."""
        return self.max_y - self.min_y


@dataclass(frozen=True, slots=True)
class VehiclePhysicsConfig:
    """Numbers that describe the simulated car's size, grip, and motor."""

    lower_chassis_half_extents: tuple[float, float, float] = (0.27, 0.06, 0.41)
    lower_chassis_offset: tuple[float, float, float] = (0.0, -0.04, 0.0)
    cabin_half_extents: tuple[float, float, float] = (0.18, 0.07, 0.21)
    cabin_offset: tuple[float, float, float] = (0.0, 0.04, 0.015)
    additional_chassis_collision_boxes: tuple[VehicleCollisionBox, ...] = ()
    chassis_collision_hull: VehicleCollisionHull | None = None
    mass_kg: float = 38.0 * RC_CAR_VOLUME_SCALE
    linear_damping: float = 0.1
    angular_damping: float = 0.62
    max_engine_force: float = 144.0 * RC_CAR_VOLUME_SCALE
    max_brake_force: float = 90.0 * RC_CAR_VOLUME_SCALE
    direction_change_speed_threshold_kmh: float = 1.0
    brake_response_exponent: float = 2.0
    front_brake_bias: float = 0.0
    rear_brake_bias: float = 0.0175
    max_steering_degrees: float = 25.0
    front_steering_scale: float = 1.0
    rear_steering_scale: float = -1.0
    drive_wheel_indices: tuple[int, ...] = (0, 1, 2, 3)
    wheel_radius: float = RC_CAR_WHEEL_RADIUS
    wheel_width: float = 0.21
    wheel_track_half_width: float = 0.47
    wheelbase_half_length: float = 0.38
    wheel_connection_height: float = RC_CAR_WHEEL_CONNECTION_HEIGHT
    suspension_stiffness: float = RC_CAR_SUSPENSION_STIFFNESS
    suspension_rest_length: float = 0.12
    max_suspension_travel_cm: float = 22.0 * RC_CAR_LINEAR_SCALE
    wheels_damping_relaxation: float = 4.7
    wheels_damping_compression: float = 10.2
    friction_slip: float = 12.0
    roll_influence: float = 0.0


FORMULA_VEHICLE_PHYSICS_CONFIG = VehiclePhysicsConfig(
    lower_chassis_half_extents=(0.18, 0.045, 0.72),
    lower_chassis_offset=(0.0, -0.015, 0.03),
    cabin_half_extents=(0.13, 0.10, 0.24),
    cabin_offset=(0.0, 0.075, -0.22),
    chassis_collision_hull=VehicleCollisionHull(
        footprint=(
            (-0.546, -1.058),
            (0.546, -1.058),
            (0.630, -0.700),
            (0.600, 1.150),
            (-0.600, 1.150),
            (-0.630, -0.700),
        ),
        min_y=-0.060,
        max_y=0.600,
    ),
    mass_kg=92.0 * RC_CAR_VOLUME_SCALE,
    linear_damping=0.08,
    angular_damping=0.42,
    max_engine_force=800.0 * RC_CAR_VOLUME_SCALE,
    max_brake_force=15.0 * RC_CAR_VOLUME_SCALE,
    front_brake_bias=0.48,
    rear_brake_bias=0.52,
    max_steering_degrees=25.0,
    front_steering_scale=1.0,
    rear_steering_scale=0.0,
    drive_wheel_indices=(2, 3),
    wheel_radius=0.18,
    wheel_width=0.19,
    wheel_track_half_width=0.50,
    wheelbase_half_length=0.70,
    wheel_connection_height=0.40,
    suspension_stiffness=50.0,
    suspension_rest_length=0.095,
    max_suspension_travel_cm=45.0,
    wheels_damping_relaxation=3.5,
    wheels_damping_compression=5.5,
    friction_slip=5.0,
    roll_influence=0.05,
)
DEFAULT_VEHICLE_PHYSICS_CONFIG = FORMULA_VEHICLE_PHYSICS_CONFIG


def vehicle_collision_bounds(config: VehiclePhysicsConfig) -> VehicleCollisionBounds:
    """Measure the space occupied by the car collider."""
    hull = config.chassis_collision_hull
    if hull is not None:
        points = hull.points
        return VehicleCollisionBounds(
            min_x=min(point[0] for point in points),
            max_x=max(point[0] for point in points),
            min_y=min(point[1] for point in points),
            max_y=max(point[1] for point in points),
            min_z=min(point[2] for point in points),
            max_z=max(point[2] for point in points),
        )

    boxes = _chassis_collision_boxes(config)
    return VehicleCollisionBounds(
        min_x=min(offset[0] - half_extents[0] for half_extents, offset in boxes),
        max_x=max(offset[0] + half_extents[0] for half_extents, offset in boxes),
        min_y=min(offset[1] - half_extents[1] for half_extents, offset in boxes),
        max_y=max(offset[1] + half_extents[1] for half_extents, offset in boxes),
        min_z=min(offset[2] - half_extents[2] for half_extents, offset in boxes),
        max_z=max(offset[2] + half_extents[2] for half_extents, offset in boxes),
    )


@dataclass(slots=True)
class RobotVehicle:
    """Bullet vehicle nodes and configuration for one robot car."""

    chassis_np: Any
    vehicle: Any
    wheel_nodes: tuple[Any, ...]
    config: VehiclePhysicsConfig
    physics_world: Any | None = None
    pending_drive_direction: int = 0
    damage: float = 0.0
    eliminated: bool = False
    pre_step_linear_velocity_mps: tuple[float, float, float] | None = None
    team_paint_entities: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class WallImpactDamageEvent:
    """Damage applied to one vehicle by a wall impact."""

    robot: RobotVehicle
    impulse_n_s: float
    force_n: float
    damage_delta: float
    total_damage: float
    eliminated: bool


@dataclass(frozen=True, slots=True)
class WallImpactContact:
    """Robot/wall manifold ownership for one wall impact."""

    robot_name: str
    robot_is_node0: bool


@dataclass(frozen=True, slots=True)
class VehicleImpactContact:
    """Robot/robot manifold ownership for one vehicle impact."""

    robot0_name: str
    robot1_name: str


@dataclass(slots=True)
class PhysicsScene:
    """Bullet physics world plus registered robot vehicles."""

    world: Any
    vehicles: list[RobotVehicle]
    fixed_time_step: float = 1 / 120

    def step(self, delta_seconds: float) -> None:
        """Advance Bullet and then apply car-to-car impact response."""
        for vehicle in self.vehicles:
            vehicle.pre_step_linear_velocity_mps = _robot_linear_velocity_tuple(vehicle)
        self.world.doPhysics(delta_seconds, 4, self.fixed_time_step)
        apply_vehicle_impact_response(physics_world=self.world, robots=tuple(self.vehicles))


@dataclass(frozen=True, slots=True)
class VehicleActuatorCommand:
    """Resolved physical actuator forces for a normalized robot command."""

    steering_degrees: float
    engine_force: float
    brake_force: float
    next_pending_drive_direction: int = 0


class VehicleLike(Protocol):
    """Protocol for Bullet vehicle methods used by command mapping."""

    def setSteeringValue(self, value: float, wheel_index: int) -> None:
        """Set steering angle for one wheel."""

    def applyEngineForce(self, value: float, wheel_index: int) -> None:
        """Apply engine force to one wheel."""

    def setBrake(self, value: float, wheel_index: int) -> None:
        """Apply brake force to one wheel."""

    def getCurrentSpeedKmHour(self) -> float:
        """Return signed vehicle speed in kilometers per hour."""
        ...


def create_physics_world() -> Any:
    """Start an empty Bullet world with gravity for the race."""
    bullet = cast(Any, import_module("panda3d.bullet"))
    core = cast(Any, import_module("panda3d.core"))

    world = bullet.BulletWorld()
    world.setGravity(core.Vec3(0, -9.81, 0))
    return world


def attach_static_box(
    *,
    world: Any,
    render: Any,
    name: str,
    position: tuple[float, float, float],
    half_extents: tuple[float, float, float],
    heading_degrees: float = 0.0,
    friction: float | None = None,
) -> Any:
    """Attach a static Bullet box collider to the physics world."""
    bullet = cast(Any, import_module("panda3d.bullet"))
    core = cast(Any, import_module("panda3d.core"))
    if friction is not None and friction < 0:
        raise ValueError("friction must be non-negative")

    body = bullet.BulletRigidBodyNode(name)
    body.addShape(bullet.BulletBoxShape(core.Vec3(*half_extents)))
    body.setMass(0)
    if friction is not None:
        body.setFriction(friction)

    node_path = render.attachNewNode(body)
    node_path.setPos(*position)
    yaw = core.Quat()
    # The physics scene uses Y as up, so track-aligned boxes must yaw around Y.
    yaw.setFromAxisAngle(heading_degrees, core.Vec3(0.0, 1.0, 0.0))
    node_path.setQuat(yaw)
    world.attachRigidBody(body)
    return node_path


def _chassis_collision_boxes(
    config: VehiclePhysicsConfig,
) -> tuple[tuple[tuple[float, float, float], tuple[float, float, float]], ...]:
    return (
        (config.lower_chassis_half_extents, config.lower_chassis_offset),
        (config.cabin_half_extents, config.cabin_offset),
        *((box.half_extents, box.offset) for box in config.additional_chassis_collision_boxes),
    )


def create_robot_vehicle(
    *,
    world: Any,
    render: Any,
    name: str,
    position: tuple[float, float, float],
    heading_degrees: float = 0.0,
    config: VehiclePhysicsConfig = DEFAULT_VEHICLE_PHYSICS_CONFIG,
) -> RobotVehicle:
    """Build the physical car body and four raycast wheels."""
    bullet = cast(Any, import_module("panda3d.bullet"))
    core = cast(Any, import_module("panda3d.core"))

    chassis = bullet.BulletRigidBodyNode(name)
    _add_chassis_collision_shapes(chassis=chassis, bullet=bullet, core=core, config=config)
    chassis.setMass(config.mass_kg)
    chassis.setLinearDamping(config.linear_damping)
    chassis.setAngularDamping(config.angular_damping)
    chassis.setDeactivationEnabled(False)

    chassis_np = render.attachNewNode(chassis)
    chassis_np.setPos(*position)
    chassis_np.setHpr(heading_degrees, 0, 0)
    world.attachRigidBody(chassis)

    vehicle = bullet.BulletVehicle(world, chassis)
    vehicle.setCoordinateSystem(bullet.YUp)
    world.attachVehicle(vehicle)

    wheel_nodes = tuple(
        _configure_wheel(
            vehicle=vehicle,
            render=render,
            name=f"{name}-wheel-{index}",
            connection=connection,
            is_front=index in front_wheel_indices(),
            config=config,
        )
        for index, connection in enumerate(wheel_connection_points(config), start=0)
    )

    return RobotVehicle(
        chassis_np=chassis_np,
        vehicle=vehicle,
        wheel_nodes=wheel_nodes,
        config=config,
        physics_world=world,
    )


def vehicle_reset_pose_is_clear(
    *,
    robot: RobotVehicle,
    position: tuple[float, float, float],
    heading_degrees: float,
    other_robots: tuple[RobotVehicle, ...],
    clearance_m: float = 0.25,
) -> bool:
    """Test a padded car and wheel envelope without moving the real vehicle."""
    world = robot.physics_world
    if world is None:
        return False
    bullet = cast(Any, import_module("panda3d.bullet"))
    core = cast(Any, import_module("panda3d.core"))
    config = robot.config
    bounds = vehicle_collision_bounds(config)
    half_width = max(bounds.half_width, config.wheel_track_half_width + config.wheel_width / 2) + clearance_m
    half_length = max(bounds.half_length, config.wheelbase_half_length + config.wheel_radius) + clearance_m
    # Include the space beneath the chassis so wheel rays cannot land on another
    # car. Leave the supporting floor just outside the query envelope.
    min_y = min(bounds.min_y, -vehicle_spawn_height(config) + 0.02)
    max_y = max(bounds.max_y, config.wheel_connection_height) + clearance_m
    probe = bullet.BulletRigidBodyNode("vehicle-reset-probe")
    probe.addShape(
        bullet.BulletBoxShape(core.Vec3(half_width, (max_y - min_y) / 2, half_length)),
        core.TransformState.makePos(core.Vec3(0.0, (max_y + min_y) / 2, 0.0)),
    )
    probe_np = core.NodePath(probe)
    probe_np.setPos(*position)
    probe_np.setHpr(heading_degrees, 0.0, 0.0)
    try:
        own_body = robot.chassis_np.node()
        for contact in world.contactTest(probe).getContacts():
            other_body = contact.getNode1() if contact.getNode0() == probe else contact.getNode0()
            if other_body != own_body:
                return False
        # Bullet's broadphase still has the old bounds for cars marshalled
        # earlier this tick. Pair queries use their current transforms directly.
        return not any(
            world.contactTestPair(probe, other.chassis_np.node()).getNumContacts() > 0
            for other in other_robots
            if other is not robot and not other.eliminated
        )
    finally:
        probe_np.removeNode()


def _add_chassis_collision_shapes(*, chassis: Any, bullet: Any, core: Any, config: VehiclePhysicsConfig) -> None:
    hull = config.chassis_collision_hull
    if hull is not None:
        chassis.addShape(_bullet_convex_hull_shape(bullet=bullet, core=core, hull=hull))
        return

    for half_extents, offset in _chassis_collision_boxes(config):
        chassis.addShape(
            bullet.BulletBoxShape(core.Vec3(*half_extents)),
            core.TransformState.makePos(core.Vec3(*offset)),
        )


def _bullet_convex_hull_shape(*, bullet: Any, core: Any, hull: VehicleCollisionHull) -> Any:
    _validate_collision_hull(hull)
    shape = bullet.BulletConvexHullShape()
    for point in hull.points:
        shape.addPoint(core.Point3(*point))
    shape.setMargin(hull.margin)
    return shape


def _validate_collision_hull(hull: VehicleCollisionHull) -> None:
    if len(hull.footprint) < 3:
        raise ValueError("chassis collision hull footprint must have at least three points")
    if hull.max_y <= hull.min_y:
        raise ValueError("chassis collision hull max_y must be greater than min_y")
    if hull.margin < 0.0:
        raise ValueError("chassis collision hull margin must be non-negative")


def apply_vehicle_command(
    *,
    vehicle: VehicleLike,
    command: RobotCommand,
    config: VehiclePhysicsConfig = DEFAULT_VEHICLE_PHYSICS_CONFIG,
) -> None:
    """Send signed throttle and steering values to a Bullet vehicle."""
    actuator_command = resolve_vehicle_actuator_command(
        command=command,
        current_speed_kmh=float(vehicle.getCurrentSpeedKmHour()),
        config=config,
    )
    _apply_vehicle_actuator_command(
        vehicle=vehicle,
        actuator_command=actuator_command,
        config=config,
    )


def apply_robot_vehicle_command(*, robot: RobotVehicle, command: RobotCommand) -> None:
    """Send a command to a robot while handling forward/reverse transitions."""
    if robot.eliminated:
        return
    actuator_command = resolve_vehicle_actuator_command(
        command=command,
        current_speed_kmh=float(robot.vehicle.getCurrentSpeedKmHour()),
        config=robot.config,
        pending_drive_direction=robot.pending_drive_direction,
    )
    robot.pending_drive_direction = actuator_command.next_pending_drive_direction
    _apply_vehicle_actuator_command(
        vehicle=robot.vehicle,
        actuator_command=actuator_command,
        config=robot.config,
    )


def apply_vehicle_impact_response(*, physics_world: Any, robots: tuple[RobotVehicle, ...]) -> None:
    """Amplify active robot-to-robot chassis impacts after Bullet solves contact."""
    extra_impulse_scale = VEHICLE_IMPACT_RESPONSE_MULTIPLIER - 1.0
    if extra_impulse_scale <= 0.0 or len(robots) < 2 or not hasattr(physics_world, "getManifolds"):
        return

    active_robot_by_name = {
        _node_name(robot.chassis_np.node()): robot
        for robot in robots
        if not robot.eliminated and _node_name(robot.chassis_np.node()) != ""
    }
    if len(active_robot_by_name) < 2:
        return

    core = cast(Any, import_module("panda3d.core"))
    for manifold in physics_world.getManifolds():
        contact = _vehicle_impact_contact(manifold=manifold, robot_names=frozenset(active_robot_by_name))
        if contact is None:
            continue
        robot0 = active_robot_by_name[contact.robot0_name]
        robot1 = active_robot_by_name[contact.robot1_name]
        for point in _manifold_points(manifold):
            applied_impulse_n_s = _manifold_point_applied_impulse(point)
            if applied_impulse_n_s <= 0.0:
                continue
            normal = _vehicle_impact_planar_normal(point)
            if _vehicle_impact_closing_speed_mps(robot0=robot0, robot1=robot1, normal=normal) <= (
                VEHICLE_IMPACT_MIN_CLOSING_SPEED_MPS
            ):
                continue
            extra_impulse_n_s = applied_impulse_n_s * extra_impulse_scale
            impulse = core.Vec3(
                normal[0] * extra_impulse_n_s,
                normal[1] * extra_impulse_n_s,
                normal[2] * extra_impulse_n_s,
            )
            _apply_central_impulse(robot0, impulse)
            _apply_central_impulse(robot1, core.Vec3(-impulse[0], -impulse[1], -impulse[2]))


def wall_damage_reference_impulse_n_s(
    config: VehiclePhysicsConfig,
    *,
    impact_speed_kmh: float = WALL_DAMAGE_FULL_IMPACT_SPEED_KMH,
) -> float:
    """Calculate the wall hit size that counts as full damage."""
    if config.mass_kg <= 0.0:
        raise ValueError("vehicle mass must be positive")
    if impact_speed_kmh <= 0.0:
        raise ValueError("impact speed must be positive")
    return config.mass_kg * impact_speed_kmh * KILOMETERS_PER_HOUR_TO_METERS_PER_SECOND


def wall_damage_reference_force_n(
    config: VehiclePhysicsConfig,
    *,
    fixed_time_step: float,
    impact_speed_kmh: float = WALL_DAMAGE_FULL_IMPACT_SPEED_KMH,
) -> float:
    """Calculate the one-step wall force that counts as full damage."""
    if fixed_time_step <= 0.0:
        raise ValueError("fixed_time_step must be positive")
    return wall_damage_reference_impulse_n_s(config, impact_speed_kmh=impact_speed_kmh) / fixed_time_step


def wall_damage_from_impact_impulse(
    impulse_n_s: float,
    config: VehiclePhysicsConfig,
    *,
    full_damage_impact_speed_kmh: float = WALL_DAMAGE_FULL_IMPACT_SPEED_KMH,
) -> float:
    """Convert a wall impact impulse into a 0.0-to-1.0 damage amount."""
    if impulse_n_s <= 0.0:
        return 0.0
    reference_impulse = wall_damage_reference_impulse_n_s(
        config,
        impact_speed_kmh=full_damage_impact_speed_kmh,
    )
    impact_ratio = impulse_n_s / reference_impulse
    return min(1.0, impact_ratio * impact_ratio)


def wall_damage_from_impact_force(
    force_n: float,
    config: VehiclePhysicsConfig,
    *,
    fixed_time_step: float,
    full_damage_impact_speed_kmh: float = WALL_DAMAGE_FULL_IMPACT_SPEED_KMH,
) -> float:
    """Convert an average wall force into a 0.0-to-1.0 damage amount."""
    if force_n <= 0.0:
        return 0.0
    reference_force = wall_damage_reference_force_n(
        config,
        fixed_time_step=fixed_time_step,
        impact_speed_kmh=full_damage_impact_speed_kmh,
    )
    impact_ratio = force_n / reference_force
    return min(1.0, impact_ratio * impact_ratio)


def apply_wall_impact_damage(
    *,
    physics_world: Any,
    robots: tuple[RobotVehicle, ...],
    fixed_time_step: float,
) -> tuple[WallImpactDamageEvent, ...]:
    """Accumulate wall-impact damage from Bullet solver impulses."""
    if fixed_time_step <= 0.0:
        raise ValueError("fixed_time_step must be positive")
    if not hasattr(physics_world, "getManifolds"):
        return ()

    active_robot_by_name = {
        _node_name(robot.chassis_np.node()): robot
        for robot in robots
        if not robot.eliminated and _node_name(robot.chassis_np.node()) != ""
    }
    impact_impulses_by_robot: dict[str, float] = {}
    for manifold in physics_world.getManifolds():
        contact = _wall_impact_contact(manifold=manifold, robot_names=frozenset(active_robot_by_name))
        if contact is None:
            continue
        impulse_n_s = _wall_impact_manifold_impulse(
            manifold=manifold,
            robot=active_robot_by_name[contact.robot_name],
            robot_is_node0=contact.robot_is_node0,
        )
        if impulse_n_s > 0.0:
            impact_impulses_by_robot[contact.robot_name] = (
                impact_impulses_by_robot.get(contact.robot_name, 0.0) + impulse_n_s
            )

    events: list[WallImpactDamageEvent] = []
    for robot_name, impulse_n_s in impact_impulses_by_robot.items():
        robot = active_robot_by_name[robot_name]
        damage_delta = wall_damage_from_impact_impulse(impulse_n_s, robot.config)
        if damage_delta <= 0.0:
            continue
        robot.damage = min(1.0, robot.damage + damage_delta)
        eliminated = False
        if robot.damage >= 1.0:
            eliminated = eliminate_robot_vehicle(robot)
        events.append(
            WallImpactDamageEvent(
                robot=robot,
                impulse_n_s=impulse_n_s,
                force_n=impulse_n_s / fixed_time_step,
                damage_delta=damage_delta,
                total_damage=robot.damage,
                eliminated=eliminated,
            )
        )
    return tuple(events)


def eliminate_robot_vehicle(robot: RobotVehicle) -> bool:
    """Remove a fully damaged robot from the physics world and hide it."""
    if robot.eliminated:
        return False
    robot.damage = 1.0
    robot.eliminated = True
    robot.pending_drive_direction = 0
    _clear_robot_body_motion(robot)
    if robot.physics_world is not None:
        _remove_robot_from_physics_world(robot)
    _set_robot_visible(robot, visible=False)
    return True


def restore_robot_vehicle(robot: RobotVehicle) -> None:
    """Restore an eliminated robot for a new round or race."""
    was_eliminated = robot.eliminated
    robot.damage = 0.0
    robot.eliminated = False
    robot.pending_drive_direction = 0
    if was_eliminated and robot.physics_world is not None:
        if hasattr(robot.physics_world, "attachRigidBody"):
            robot.physics_world.attachRigidBody(robot.chassis_np.node())
        if hasattr(robot.physics_world, "attachVehicle"):
            robot.physics_world.attachVehicle(robot.vehicle)
    _set_robot_visible(robot, visible=True)


def _remove_robot_from_physics_world(robot: RobotVehicle) -> None:
    world = robot.physics_world
    if world is None:
        return
    if hasattr(world, "removeVehicle"):
        world.removeVehicle(robot.vehicle)
        return
    if hasattr(world, "removeRigidBody"):
        world.removeRigidBody(robot.chassis_np.node())


def resolve_vehicle_actuator_command(
    *,
    command: RobotCommand,
    current_speed_kmh: float,
    config: VehiclePhysicsConfig = DEFAULT_VEHICLE_PHYSICS_CONFIG,
    pending_drive_direction: int = 0,
) -> VehicleActuatorCommand:
    """Resolve normalized command inputs into engine and brake forces."""
    bounded = clamp_command(command)
    steering_degrees = bounded.steer * config.max_steering_degrees
    requested_direction = _throttle_direction(bounded.throttle)

    if requested_direction == 0:
        return VehicleActuatorCommand(
            steering_degrees=steering_degrees,
            engine_force=0.0,
            brake_force=0.0,
        )

    if pending_drive_direction != 0:
        if abs(current_speed_kmh) <= config.direction_change_speed_threshold_kmh:
            pending_drive_direction = 0
        else:
            return VehicleActuatorCommand(
                steering_degrees=steering_degrees,
                engine_force=0.0,
                brake_force=_brake_force_for_amount(abs(bounded.throttle), config),
                next_pending_drive_direction=requested_direction,
            )

    if (bounded.throttle < 0.0 and current_speed_kmh > config.direction_change_speed_threshold_kmh) or (
        bounded.throttle > 0.0 and current_speed_kmh < -config.direction_change_speed_threshold_kmh
    ):
        return VehicleActuatorCommand(
            steering_degrees=steering_degrees,
            engine_force=0.0,
            brake_force=_brake_force_for_amount(abs(bounded.throttle), config),
            next_pending_drive_direction=requested_direction,
        )

    return VehicleActuatorCommand(
        steering_degrees=steering_degrees,
        engine_force=bounded.throttle * config.max_engine_force,
        brake_force=0.0,
    )


def _brake_force_for_amount(amount: float, config: VehiclePhysicsConfig) -> float:
    return amount**config.brake_response_exponent * config.max_brake_force


def _throttle_direction(throttle: float) -> int:
    if throttle > 0.0:
        return 1
    if throttle < 0.0:
        return -1
    return 0


def _wall_impact_contact(*, manifold: Any, robot_names: frozenset[str]) -> WallImpactContact | None:
    node0_name = _node_name(manifold.getNode0()) if hasattr(manifold, "getNode0") else ""
    node1_name = _node_name(manifold.getNode1()) if hasattr(manifold, "getNode1") else ""
    if node0_name in robot_names and node1_name.startswith("track-barrier"):
        return WallImpactContact(robot_name=node0_name, robot_is_node0=True)
    if node1_name in robot_names and node0_name.startswith("track-barrier"):
        return WallImpactContact(robot_name=node1_name, robot_is_node0=False)
    return None


def _vehicle_impact_contact(*, manifold: Any, robot_names: frozenset[str]) -> VehicleImpactContact | None:
    node0_name = _node_name(manifold.getNode0()) if hasattr(manifold, "getNode0") else ""
    node1_name = _node_name(manifold.getNode1()) if hasattr(manifold, "getNode1") else ""
    if node0_name in robot_names and node1_name in robot_names and node0_name != node1_name:
        return VehicleImpactContact(robot0_name=node0_name, robot1_name=node1_name)
    return None


def _vehicle_impact_planar_normal(point: Any) -> tuple[float, float, float]:
    if not hasattr(point, "getNormalWorldOnB"):
        return (0.0, 0.0, 0.0)
    normal = _vector3_tuple(point.getNormalWorldOnB())
    planar_length = (normal[0] * normal[0] + normal[2] * normal[2]) ** 0.5
    if planar_length <= 0.000001:
        return (0.0, 0.0, 0.0)
    return (normal[0] / planar_length, 0.0, normal[2] / planar_length)


def _vehicle_impact_closing_speed_mps(
    *,
    robot0: RobotVehicle,
    robot1: RobotVehicle,
    normal: tuple[float, float, float],
) -> float:
    robot0_velocity = _robot_impact_velocity_tuple(robot0)
    robot1_velocity = _robot_impact_velocity_tuple(robot1)
    relative_velocity = (
        robot0_velocity[0] - robot1_velocity[0],
        robot0_velocity[1] - robot1_velocity[1],
        robot0_velocity[2] - robot1_velocity[2],
    )
    return max(0.0, -_vector3_dot(relative_velocity, normal))


def _wall_impact_manifold_impulse(*, manifold: Any, robot: RobotVehicle, robot_is_node0: bool) -> float:
    solver_impulse_n_s = 0.0
    velocity_impulse_n_s = 0.0
    for point in _manifold_points(manifold):
        angular_damping = _wall_impact_angular_damping(
            robot=robot,
            point=point,
            robot_is_node0=robot_is_node0,
        )
        if _manifold_point_is_new_contact(point):
            applied_impulse_n_s = _manifold_point_applied_impulse(point)
            if angular_damping > 0.0:
                applied_impulse_n_s = min(applied_impulse_n_s, robot.config.mass_kg * _robot_linear_speed_mps(robot))
                applied_impulse_n_s *= angular_damping * angular_damping
            solver_impulse_n_s += applied_impulse_n_s
        velocity_impulse_n_s = max(
            velocity_impulse_n_s,
            _wall_impact_velocity_impulse(
                robot=robot,
                point=point,
                robot_is_node0=robot_is_node0,
                angular_damping=angular_damping,
            ),
        )
    return max(solver_impulse_n_s, velocity_impulse_n_s)


def _wall_impact_velocity_impulse(
    *,
    robot: RobotVehicle,
    point: Any,
    robot_is_node0: bool,
    angular_damping: float,
) -> float:
    closing_speed_mps = _wall_impact_closing_speed_mps(robot=robot, point=point, robot_is_node0=robot_is_node0)
    if closing_speed_mps <= WALL_DAMAGE_MIN_CLOSING_SPEED_MPS:
        return 0.0
    return robot.config.mass_kg * closing_speed_mps * angular_damping


def _wall_impact_angular_damping(*, robot: RobotVehicle, point: Any, robot_is_node0: bool) -> float:
    closing_speed_mps = _wall_impact_closing_speed_mps(robot=robot, point=point, robot_is_node0=robot_is_node0)
    if closing_speed_mps <= 0.0:
        return 0.0
    speed_mps = _robot_linear_speed_mps(robot)
    if speed_mps <= 0.0:
        return 0.0
    return min(1.0, closing_speed_mps / speed_mps)


def _wall_impact_closing_speed_mps(*, robot: RobotVehicle, point: Any, robot_is_node0: bool) -> float:
    if not hasattr(point, "getNormalWorldOnB"):
        return 0.0
    normal = _vector3_tuple(point.getNormalWorldOnB())
    velocity = _robot_impact_velocity_tuple(robot)
    velocity_along_wall_normal = _vector3_dot(velocity, normal)
    if robot_is_node0:
        return max(0.0, -velocity_along_wall_normal)
    return max(0.0, velocity_along_wall_normal)


def _robot_linear_speed_mps(robot: RobotVehicle) -> float:
    velocity = _robot_impact_velocity_tuple(robot)
    return (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]) ** 0.5


def _robot_impact_velocity_tuple(robot: RobotVehicle) -> tuple[float, float, float]:
    pre_step_velocity = getattr(robot, "pre_step_linear_velocity_mps", None)
    if pre_step_velocity is not None and any(abs(component) > 0.0 for component in pre_step_velocity):
        return pre_step_velocity
    return _robot_linear_velocity_tuple(robot)


def _robot_linear_velocity_tuple(robot: RobotVehicle) -> tuple[float, float, float]:
    body = robot.chassis_np.node()
    if not hasattr(body, "getLinearVelocity"):
        return (0.0, 0.0, 0.0)
    return _vector3_tuple(body.getLinearVelocity())


def _vector3_tuple(vector: Any) -> tuple[float, float, float]:
    return (_vector3_component(vector, 0), _vector3_component(vector, 1), _vector3_component(vector, 2))


def _vector3_component(vector: Any, index: int) -> float:
    method_names = ("getX", "getY", "getZ")
    method_name = method_names[index]
    if hasattr(vector, method_name):
        return float(getattr(vector, method_name)())
    attribute_names = ("x", "y", "z")
    attribute_name = attribute_names[index]
    if hasattr(vector, attribute_name):
        return float(getattr(vector, attribute_name))
    return float(vector[index])


def _vector3_dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _manifold_points(manifold: Any) -> tuple[Any, ...]:
    if not hasattr(manifold, "getManifoldPoints"):
        return ()
    return tuple(manifold.getManifoldPoints())


def _manifold_point_applied_impulse(point: Any) -> float:
    if not hasattr(point, "getAppliedImpulse"):
        return 0.0
    return max(0.0, float(point.getAppliedImpulse()))


def _manifold_point_is_new_contact(point: Any) -> bool:
    if not hasattr(point, "getLifeTime"):
        return True
    return int(point.getLifeTime()) <= 1


def _clear_robot_body_motion(robot: RobotVehicle) -> None:
    core = cast(Any, import_module("panda3d.core"))
    body = robot.chassis_np.node()
    if hasattr(body, "setLinearVelocity"):
        body.setLinearVelocity(core.Vec3(0.0, 0.0, 0.0))
    if hasattr(body, "setAngularVelocity"):
        body.setAngularVelocity(core.Vec3(0.0, 0.0, 0.0))
    if hasattr(body, "clearForces"):
        body.clearForces()


def _apply_central_impulse(robot: RobotVehicle, impulse: Any) -> None:
    body = robot.chassis_np.node()
    if hasattr(body, "applyCentralImpulse"):
        body.applyCentralImpulse(impulse)


def _set_robot_visible(robot: RobotVehicle, *, visible: bool) -> None:
    method_name = "show" if visible else "hide"
    if hasattr(robot.chassis_np, method_name):
        getattr(robot.chassis_np, method_name)()
    for wheel_node in robot.wheel_nodes:
        if hasattr(wheel_node, method_name):
            getattr(wheel_node, method_name)()


def _node_name(node: Any) -> str:
    return str(node.getName()) if hasattr(node, "getName") else ""


def _apply_vehicle_actuator_command(
    *,
    vehicle: VehicleLike,
    actuator_command: VehicleActuatorCommand,
    config: VehiclePhysicsConfig,
) -> None:
    front_indices = front_wheel_indices()
    rear_indices = rear_wheel_indices()
    drive_indices = set(config.drive_wheel_indices)

    for wheel_index in front_indices:
        vehicle.setSteeringValue(actuator_command.steering_degrees * config.front_steering_scale, wheel_index)
        vehicle.setBrake(actuator_command.brake_force * config.front_brake_bias, wheel_index)

    for wheel_index in rear_indices:
        vehicle.setSteeringValue(actuator_command.steering_degrees * config.rear_steering_scale, wheel_index)
        vehicle.setBrake(actuator_command.brake_force * config.rear_brake_bias, wheel_index)

    for wheel_index in (*front_indices, *rear_indices):
        vehicle.applyEngineForce(actuator_command.engine_force if wheel_index in drive_indices else 0.0, wheel_index)


def wheel_connection_points(config: VehiclePhysicsConfig) -> tuple[tuple[float, float, float], ...]:
    """List where the four wheel suspensions attach to the chassis."""
    return tuple(
        (axis_x, axis_y + config.wheel_connection_height, axis_z)
        for axis_x, axis_y, axis_z in wheel_axis_points(config)
    )


def wheel_axis_points(config: VehiclePhysicsConfig) -> tuple[tuple[float, float, float], ...]:
    """List the four axle positions before suspension height is added."""
    side_offset = config.wheel_track_half_width
    front = config.wheelbase_half_length
    rear = -config.wheelbase_half_length
    return (
        (-side_offset, 0.0, front),
        (side_offset, 0.0, front),
        (-side_offset, 0.0, rear),
        (side_offset, 0.0, rear),
    )


def front_wheel_indices() -> tuple[int, ...]:
    """Name which wheel slots steer."""
    return (0, 1)


def rear_wheel_indices() -> tuple[int, ...]:
    """Name which wheel slots are rear wheels."""
    return (2, 3)


def vehicle_spawn_height(
    config: VehiclePhysicsConfig = DEFAULT_VEHICLE_PHYSICS_CONFIG,
    *,
    surface_y: float = DEFAULT_VEHICLE_SPAWN_SURFACE_Y,
) -> float:
    """Place the chassis high enough that the wheels rest on a surface."""
    if config.wheel_radius <= 0:
        raise ValueError("wheel_radius must be positive")
    return surface_y + max(config.wheel_radius, 2 * config.wheel_radius - config.wheel_connection_height)


def _configure_wheel(
    *,
    vehicle: Any,
    render: Any,
    name: str,
    connection: tuple[float, float, float],
    is_front: bool,
    config: VehiclePhysicsConfig,
) -> Any:
    core = cast(Any, import_module("panda3d.core"))

    wheel_np = render.attachNewNode(name)
    wheel = vehicle.createWheel()
    wheel.setNode(wheel_np.node())
    wheel.setChassisConnectionPointCs(core.Point3(*connection))
    wheel.setFrontWheel(is_front)
    wheel.setWheelDirectionCs(core.Vec3(0.0, -1.0, 0.0))
    wheel.setWheelAxleCs(core.Vec3(*WHEEL_AXLE_DIRECTION))
    wheel.setWheelRadius(config.wheel_radius)
    wheel.setMaxSuspensionTravelCm(config.max_suspension_travel_cm)
    wheel.setSuspensionStiffness(config.suspension_stiffness)
    wheel.setWheelsDampingRelaxation(config.wheels_damping_relaxation)
    wheel.setWheelsDampingCompression(config.wheels_damping_compression)
    wheel.setFrictionSlip(config.friction_slip)
    wheel.setRollInfluence(config.roll_influence)
    return wheel_np
