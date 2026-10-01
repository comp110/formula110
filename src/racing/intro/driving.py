"""A deterministic Bullet stunt run, controlled exclusively through actuators.

The run is simulated once at startup. Playback interpolates the physical body
and wheel transforms; it never substitutes a path or a hand-animated yaw.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from math import atan2, cos, degrees, floor, hypot, radians, tan
from typing import Any

from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    PhysicsScene,
    apply_robot_vehicle_command,
    create_physics_world,
    create_robot_vehicle,
)
from racing.student.api import RobotCommand

STEP = 1 / 120
DURATION = 40.0
VISUAL_SCALE = 1.75
START_HEADING = 110.0
DRIFT_GRIP = 1.2
DONUT_POWER_MULTIPLIER = 3.0
RESET_X = 20.0


def bounded(value: float, limit: float = 1.0) -> float:
    return max(-limit, min(limit, value))


def angle_difference(target: float, current: float) -> float:
    return (target - current + 180) % 360 - 180


@dataclass(frozen=True, slots=True)
class Transform:
    position: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Frame:
    chassis: Transform
    wheels: tuple[Transform, ...]
    heading: float
    speed: float
    skid: float
    stage: str
    throttle: float
    steer: float

    @property
    def visible(self) -> bool:
        return self.stage != "waiting"


def snapshot(node: Any) -> Transform:
    position, rotation = node.getPos(), node.getQuat()
    return Transform(
        (float(position.x), float(position.y), float(position.z)),
        (float(rotation.getR()), float(rotation.getI()), float(rotation.getJ()), float(rotation.getK())),
    )


def interpolate(a: Transform, b: Transform, amount: float) -> Transform:
    position = tuple(x + (y - x) * amount for x, y in zip(a.position, b.position, strict=True))
    sign = -1 if sum(x * y for x, y in zip(a.rotation, b.rotation, strict=True)) < 0 else 1
    quat = tuple(x + (sign * y - x) * amount for x, y in zip(a.rotation, b.rotation, strict=True))
    length = sum(x * x for x in quat) ** 0.5
    return Transform(position, tuple(x / length for x in quat))  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True)
class DrivingLoop:
    frames: tuple[Frame, ...]
    turn_direction: int = -1

    def sample(self, seconds: float) -> Frame:
        # Snap floating-point noise at tick boundaries so the reset happens
        # on the same frame even after many whole-loop additions.
        index = round((seconds % DURATION) / STEP, 9) % (len(self.frames) - 1)
        lower = int(index)
        a, b = self.frames[lower], self.frames[lower + 1]
        # A reset is an offscreen teleport. Interpolating it would sweep the
        # car back through the middle of the stage for one physics frame.
        if a.visible and not b.visible:
            return a
        blend = index - lower
        return Frame(
            interpolate(a.chassis, b.chassis, blend),
            tuple(interpolate(x, y, blend) for x, y in zip(a.wheels, b.wheels, strict=True)),
            a.heading + angle_difference(b.heading, a.heading) * blend,
            a.speed + (b.speed - a.speed) * blend,
            a.skid,
            a.stage,
            a.throttle,
            a.steer,
        )


@dataclass(frozen=True, slots=True)
class DrivingSequence:
    runs: tuple[DrivingLoop, DrivingLoop]

    def sample(self, seconds: float) -> Frame:
        return self.runs[floor(seconds / DURATION) % 2].sample(seconds)


def simulate_sequence() -> DrivingSequence:
    """Alternate away-from-camera and toward-camera donuts on successive loops."""
    return DrivingSequence((simulate_loop(-1), simulate_loop(1)))


def simulate_loop(turn_direction: int = -1) -> DrivingLoop:
    """Drive in, brake, hold, power through a donut, then accelerate away."""
    if turn_direction not in (-1, 1):
        raise ValueError("turn_direction must be -1 or 1")
    core = import_module("panda3d.core")
    bullet = import_module("panda3d.bullet")
    world = create_physics_world()
    root = core.NodePath("intro-physics")
    # An infinite plane gives reliable wheel raycasts without large-box
    # precision loss, and keeps the departure grounded beyond the camera.
    floor = bullet.BulletRigidBodyNode("intro-floor")
    floor.addShape(bullet.BulletPlaneShape(core.Vec3(0, 1, 0), 0))
    floor.setFriction(1.0)
    root.attachNewNode(floor)
    world.attachRigidBody(floor)
    config = FORMULA_VEHICLE_PHYSICS_CONFIG
    park_z = -0.45 / VISUAL_SCALE
    robot = create_robot_vehicle(
        world=world, render=root, name="intro-physical-car", position=(-10, 0.2, park_z + 10 * tan(radians(20)))
    )
    orientation = core.Quat()
    orientation.setFromAxisAngle(START_HEADING, core.Vec3(0, 1, 0))
    robot.chassis_np.setQuat(orientation)
    simulation = PhysicsScene(world, [robot])
    wheels = [robot.vehicle.getWheel(i) for i in range(4)]
    # Match the existing art's larger rear tires so the contact patches meet
    # the visible tread. All other chassis/suspension values are the game's.
    for i, wheel in enumerate(wheels):
        wheel.setWheelRadius(config.wheel_radius * (1.08 if i < 2 else 1.18))

    def command(throttle: float, steer: float, brake: float = 0.0) -> None:
        apply_robot_vehicle_command(robot=robot, command=RobotCommand(throttle=throttle, steer=steer))
        if brake:
            for i in range(4):
                robot.vehicle.setBrake(brake, i)

    # Settle suspension before the run, while the car is outside the view.
    for _ in range(180):
        command(0, 0, 0.5)
        simulation.step(STEP)

    frames: list[Frame] = []
    stage, parked_at = "waiting", 0.0
    last_heading, turn = START_HEADING, 0.0
    for tick in range(round(DURATION / STEP) + 1):
        seconds = tick * STEP
        position = robot.chassis_np.getPos()
        if stage == "exit" and float(position.x) >= RESET_X:
            # The whole car is well beyond the right edge. Teleport the
            # playback to its hidden start and hold until the next entrance.
            reset = replace(frames[0], speed=0.0, skid=0.0, throttle=0.0, steer=0.0)
            frames.extend([reset] * (round(DURATION / STEP) + 1 - tick))
            break
        forward = robot.chassis_np.getQuat().xform(core.Vec3(0, 0, 1))
        heading = degrees(atan2(float(forward.x), float(forward.z)))
        yaw_rate = angle_difference(heading, last_heading) / STEP
        if stage == "donut":
            turn += turn_direction * angle_difference(heading, last_heading)
        last_heading = heading
        velocity = robot.chassis_np.node().getLinearVelocity()
        speed = float(velocity.dot(forward))
        throttle = steer = brake = 0.0

        if stage == "waiting":
            brake = 0.5
            if seconds >= 3.0:
                stage = "approach"
        if stage == "approach":
            remaining = -float(position.x) / cos(radians(20))
            desired_speed = min(2.1, max(0.0, remaining * 1.4))
            error = desired_speed - speed
            throttle = max(0.0, bounded(error * 0.3, 0.35))
            brake = max(0.0, min(0.7, -error * 0.3))
            steer = bounded(angle_difference(START_HEADING, heading) * 0.035 - yaw_rate * 0.008)
            if remaining < 0.035 and abs(speed) < 0.06:
                stage, parked_at = "parked", seconds
        if stage == "parked":
            throttle, steer, brake = 0.0, 0.0, 0.5
            if seconds - parked_at >= 1.6:
                stage = "donut"
        if stage == "donut":
            throttle, steer, brake = 1.0, float(turn_direction), 0.0
            for wheel in wheels[2:]:
                wheel.setFrictionSlip(DRIFT_GRIP)
            if turn >= 320:
                stage = "exit"
        if stage == "exit":
            throttle, brake = 1.0, 0.0
            steer = bounded(angle_difference(START_HEADING, heading) * 0.055 - yaw_rate * 0.014)
            for wheel in wheels[2:]:
                wheel.setFrictionSlip(config.friction_slip)

        skid = min(float(wheel.getSkidInfo()) for wheel in wheels[2:])
        frames.append(
            Frame(
                snapshot(robot.chassis_np),
                tuple(snapshot(node) for node in robot.wheel_nodes),
                heading,
                hypot(float(velocity.x), float(velocity.z)),
                1 - skid,
                stage,
                throttle,
                steer,
            )
        )
        command(throttle, steer, brake)
        if stage == "donut":
            # Overpower the rear contact patches. Bullet resolves the resulting
            # loss of lateral grip and yaw; no body position/rotation is imposed.
            for index in config.drive_wheel_indices:
                robot.vehicle.applyEngineForce(config.max_engine_force * DONUT_POWER_MULTIPLIER, index)
        simulation.step(STEP)
    world.removeVehicle(robot.vehicle)
    root.removeNode()
    return DrivingLoop(tuple(frames), turn_direction)
