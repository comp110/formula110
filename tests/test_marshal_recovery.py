from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, cast

import pytest

from racing.graphics.panda_config import configure_panda_y_up
from racing.graphics.track_rendering import TRACK_SURFACE_Y, add_racing_scene_collisions
from racing.physics import (
    PhysicsScene,
    attach_static_box,
    create_physics_world,
    create_robot_vehicle,
    vehicle_reset_pose_is_clear,
    vehicle_spawn_height,
)
from racing.race.progress import (
    LapProgressTracker,
    TrackProgressModel,
    build_track_progress_model,
    project_track_position,
    resolve_track,
)
from racing.race.runtime import (
    RaceCarRuntime,
    RaceRecoveryConfig,
    maybe_marshal_race_runtimes,
    robot_track_point,
)
from racing.track.world import TrackPoint

RECOVERY = RaceRecoveryConfig(stuck_seconds=1.5, distance_penalty_m=5.0, cooldown_seconds=2.0)
SPAWN_Y = vehicle_spawn_height(surface_y=TRACK_SURFACE_Y)


@dataclass
class RecoveryScene:
    world: Any
    render: Any
    model: TrackProgressModel

    def car(self, name: str, x: float, z: float, *, stuck: bool = True) -> RaceCarRuntime:
        return RaceCarRuntime(
            robot=create_robot_vehicle(world=self.world, render=self.render, name=name, position=(x, SPAWN_Y, z)),
            tracker=LapProgressTracker(total_length_m=self.model.total_length_m),
            stuck_seconds=2.0 if stuck else 0.0,
        )

    def marshal(self, runtimes: tuple[RaceCarRuntime, ...]) -> int:
        return maybe_marshal_race_runtimes(
            model=self.model,
            runtimes=runtimes,
            projections=tuple(project_track_position(self.model, robot_track_point(r.robot)) for r in runtimes),
            recovery_config=RECOVERY,
            delta_seconds=1 / 120,
        )


@pytest.fixture
def scene() -> RecoveryScene:
    configure_panda_y_up()
    core = cast(Any, import_module("panda3d.core"))
    world = create_physics_world()
    render = core.NodePath("marshal-test")
    attach_static_box(
        world=world,
        render=render,
        name="floor",
        position=(0.0, TRACK_SURFACE_Y - 0.1, 0.0),
        half_extents=(200.0, 0.1, 200.0),
    )
    model = build_track_progress_model(
        (TrackPoint(0.0, 0.0), TrackPoint(0.0, 40.0), TrackPoint(40.0, 40.0), TrackPoint(40.0, 0.0))
    )
    return RecoveryScene(world, render, model)


def test_clearance_query_ignores_own_body_and_floor_without_moving_car(scene: RecoveryScene) -> None:
    runtime = scene.car("recovering", 0.0, 20.0)
    robot = runtime.robot
    before = robot.chassis_np.getTransform()
    body_count = scene.world.getNumRigidBodies()

    assert vehicle_reset_pose_is_clear(robot=robot, position=(0.0, SPAWN_Y, 20.0), heading_degrees=0.0, other_robots=())
    assert robot.chassis_np.getTransform() == before
    assert scene.world.getNumRigidBodies() == body_count


@pytest.mark.parametrize("heading", [0.0, 90.0, 180.0])
def test_clearance_query_detects_a_car_moved_this_tick(scene: RecoveryScene, heading: float) -> None:
    runtime = scene.car("recovering", 10.0, 20.0)
    blocker = scene.car("blocker", 20.0, 20.0)
    blocker.robot.chassis_np.setPos(0.0, SPAWN_Y, 20.0)
    blocker.robot.chassis_np.setHpr(heading, 0.0, 0.0)

    assert not vehicle_reset_pose_is_clear(
        robot=runtime.robot,
        position=(0.0, SPAWN_Y, 20.0),
        heading_degrees=0.0,
        other_robots=(blocker.robot,),
    )


def test_marshal_avoids_occupied_lane_and_clears_crash_motion(scene: RecoveryScene) -> None:
    runtime = scene.car("recovering", 8.0, 20.0)
    blocker = scene.car("parked", 1.155, 20.0, stuck=False)
    robot = runtime.robot
    body = robot.chassis_np.node()
    core = cast(Any, import_module("panda3d.core"))
    body.setLinearVelocity(core.Vec3(5.0, 30.0, -5.0))
    body.setAngularVelocity(core.Vec3(20.0, 5.0, -30.0))
    body.applyCentralForce(core.Vec3(0.0, 1000.0, 0.0))
    robot.pre_step_linear_velocity_mps = (5.0, 30.0, -5.0)
    robot.pending_drive_direction = -1
    robot.vehicle.applyEngineForce(100.0, 2)
    robot.vehicle.setBrake(100.0, 3)

    assert scene.marshal((runtime, blocker)) == 1

    assert float(robot.chassis_np.getX()) < 0.0
    assert body.getLinearVelocity().length() == 0.0
    assert body.getAngularVelocity().length() == 0.0
    assert body.getTotalForce().length() == 0.0
    assert robot.pre_step_linear_velocity_mps is None
    assert robot.pending_drive_direction == 0
    assert robot.vehicle.getWheel(2).getEngineForce() == 0.0
    assert robot.vehicle.getWheel(3).getBrake() == 0.0
    assert runtime.sensor_state.position is not None
    assert runtime.sensor_state.position.x == pytest.approx(robot_track_point(robot).x)
    assert runtime.sensor_state.position.z == pytest.approx(robot_track_point(robot).z)
    assert runtime.marshal_count == 1
    assert runtime.marshal_penalty_m == 5.0
    assert blocker.marshal_count == 0
    assert scene.world.contactTestPair(body, blocker.robot.chassis_np.node()).getNumContacts() == 0
    _assert_cars_settle(scene, (runtime, blocker))


def test_marshal_searches_behind_a_blocked_row(scene: RecoveryScene) -> None:
    runtime = scene.car("recovering", 8.0, 20.0)
    barrier = attach_static_box(
        world=scene.world,
        render=scene.render,
        name="track-barrier-blocked-row",
        position=(0.0, 0.5, 20.0),
        half_extents=(4.0, 0.5, 0.1),
    )

    assert scene.marshal((runtime,)) == 1

    assert float(runtime.robot.chassis_np.getZ()) < 18.0
    assert scene.world.contactTestPair(runtime.robot.chassis_np.node(), barrier.node()).getNumContacts() == 0
    assert runtime.sensor_state.position is not None
    assert runtime.sensor_state.position.x == pytest.approx(robot_track_point(runtime.robot).x)
    assert runtime.sensor_state.position.z == pytest.approx(robot_track_point(runtime.robot).z)
    _assert_cars_settle(scene, (runtime,))


@pytest.mark.parametrize("progress", [41.0, 1.0])
def test_backward_search_follows_bends_and_wraps_at_start(scene: RecoveryScene, progress: float) -> None:
    # Block the row just after a corner so recovery must follow the preceding
    # segment, including the segment before the wrapped start line.
    after_corner = progress > 40.0
    runtime = scene.car("recovering", 1.0 if after_corner else -8.0, 48.0 if after_corner else 1.0)
    attach_static_box(
        world=scene.world,
        render=scene.render,
        name="track-barrier-corner",
        position=(1.0 if after_corner else 0.0, 0.5, 40.0 if after_corner else 1.0),
        half_extents=(1.4, 0.5, 4.0) if after_corner else (4.0, 0.5, 1.4),
    )

    assert scene.marshal((runtime,)) == 1

    point = robot_track_point(runtime.robot)
    projection = project_track_position(scene.model, point)
    assert abs(projection.signed_distance_to_center_m) < 0.01
    assert (progress - projection.progress_distance_m) % scene.model.total_length_m < 10.0
    assert float(runtime.robot.chassis_np.getH()) == pytest.approx(projection.heading_degrees)
    assert (point.x if after_corner else point.z) == pytest.approx(0.0)


def test_blocked_recovery_retries_without_teleport_or_penalty(scene: RecoveryScene) -> None:
    runtime = scene.car("recovering", 8.0, 20.0)
    obstacle = attach_static_box(
        world=scene.world,
        render=scene.render,
        name="track-barrier-blocking-all-candidates",
        position=(0.0, 0.5, 0.0),
        half_extents=(100.0, 0.5, 100.0),
    )
    original_transform = runtime.robot.chassis_np.getTransform()
    original_sensor_state = runtime.sensor_state

    for _ in range(2):
        assert scene.marshal((runtime,)) == 0
        assert runtime.robot.chassis_np.getTransform() == original_transform
        assert runtime.sensor_state is original_sensor_state
        assert runtime.stuck_seconds == 2.0
        assert runtime.marshal_count == 0
        assert runtime.marshal_penalty_m == 0.0
        assert runtime.marshal_cooldown_seconds == 0.0

    scene.world.removeRigidBody(obstacle.node())
    assert scene.marshal((runtime,)) == 1
    assert runtime.marshal_count == 1
    assert runtime.marshal_penalty_m == 5.0


@pytest.mark.parametrize("car_count", [2, 8, 10])
def test_simultaneous_recoveries_use_separate_spaces_and_stay_grounded(scene: RecoveryScene, car_count: int) -> None:
    runtimes = tuple(scene.car(f"car-{index}", -8.0 - index * 3.0, 20.0) for index in range(car_count))

    assert scene.marshal(runtimes) == car_count

    positions = {robot_track_point(runtime.robot) for runtime in runtimes}
    assert len(positions) == car_count
    for index, runtime in enumerate(runtimes):
        body = runtime.robot.chassis_np.node()
        for other in runtimes[index + 1 :]:
            assert scene.world.contactTestPair(body, other.robot.chassis_np.node()).getNumContacts() == 0
    _assert_cars_settle(scene, runtimes)


def test_recovery_stays_clear_of_real_bahrain_barriers(scene: RecoveryScene) -> None:
    track = resolve_track("bahrain")
    scene.model = track.model
    add_racing_scene_collisions(physics_world=scene.world, render=scene.render, samples=track.samples)
    runtime = scene.car("recovering", 0.0, 0.0)
    # Visit the whole course to exercise recovery near every bend, with actual
    # Bullet barrier geometry rather than a geometric approximation.
    for index in range(0, len(track.model.points), 5):
        point = track.model.points[index]
        runtime.robot.chassis_np.setPos(point.x, SPAWN_Y, point.z)
        runtime.stuck_seconds = 2.0
        runtime.marshal_cooldown_seconds = 0.0

        assert scene.marshal((runtime,)) == 1

        contacts = scene.world.contactTest(runtime.robot.chassis_np.node()).getContacts()
        assert all(
            not str(node.getName()).startswith("track-barrier")
            for contact in contacts
            for node in (contact.getNode0(), contact.getNode1())
        )


def _assert_cars_settle(scene: RecoveryScene, runtimes: tuple[RaceCarRuntime, ...]) -> None:
    physics = PhysicsScene(scene.world, [runtime.robot for runtime in runtimes])
    for _ in range(120):
        physics.step(1 / 120)
        for runtime in runtimes:
            assert float(runtime.robot.chassis_np.getY()) < SPAWN_Y + 0.2
            assert abs(float(runtime.robot.chassis_np.node().getLinearVelocity()[1])) < 1.0
