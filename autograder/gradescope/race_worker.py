#!/usr/bin/env python3
"""Trusted physics worker for one controller validation or race trial."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import pickle
import select
import shutil
import signal
import struct
import subprocess
import sys
import time
from collections import deque
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Any, cast

RESULT_PREFIX = "FORMULA110_RESULT="
CONTROL_WORKER_PATH = Path(os.environ.get("FORMULA110_CONTROL_WORKER", "/opt/formula110-autograder/control_worker.py"))
MAX_RESPONSE_BYTES = 4096
COMMAND_TIMEOUT_SECONDS = 0.5
CONTROLLER_STARTUP_TIMEOUT_SECONDS = 30.0
CONTROLLER_MEMORY_LIMIT_BYTES = 1536 * 1024 * 1024
CONTROLLER_MEMORY_POLL_SECONDS = 0.02
STANDARD_GRAVITY_MPS2 = 9.80665
BRAKE_FORCE_EPSILON = 1e-9
DRIFT_MIN_SPEED_MPS = 8.0
DRIFT_MIN_SLIP_DEGREES = 12.0
DRIFT_MAX_SLIP_DEGREES = 45.0
SUSTAINED_SPEED_WINDOW_SECONDS = 1.0
CPU_ONLY_ENVIRONMENT = {
    "FORMULA110_DEVICE": "cpu",
    "CUDA_VISIBLE_DEVICES": "",
    "HIP_VISIBLE_DEVICES": "",
    "ROCR_VISIBLE_DEVICES": "",
    "JAX_PLATFORMS": "cpu",
    "JAX_PLATFORM_NAME": "cpu",
    "ONEAPI_DEVICE_SELECTOR": "*:cpu",
    "SYCL_DEVICE_FILTER": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


class LapTelemetryAccumulator:
    """Trusted metrics accumulated between two lap boundaries."""

    __slots__ = (
        "_speed_window",
        "_speed_window_duration_seconds",
        "_speed_window_integral_m",
        "brake_applied",
        "damage_at_start",
        "drift_distance_m",
        "horizontal_g_seconds",
        "started_at_seconds",
        "sustained_top_speed_mps",
        "wall_contact_seconds",
    )

    def __init__(self, *, started_at_seconds: float, damage_at_start: float) -> None:
        self.started_at_seconds = started_at_seconds
        self.damage_at_start = damage_at_start
        self.wall_contact_seconds = 0.0
        self.horizontal_g_seconds = 0.0
        self.brake_applied = False
        self.drift_distance_m = 0.0
        self.sustained_top_speed_mps = 0.0
        self._speed_window = deque()
        self._speed_window_duration_seconds = 0.0
        self._speed_window_integral_m = 0.0

    def record_step(
        self,
        *,
        delta_seconds: float,
        wall_contact: bool,
        horizontal_g_seconds: float,
        brake_applied: bool,
        drift_distance_m: float,
        forward_speed_mps: float,
    ) -> None:
        """Add one fixed physics step to this lap."""
        bounded_delta_seconds = max(0.0, delta_seconds)
        if wall_contact:
            self.wall_contact_seconds += bounded_delta_seconds
        self.horizontal_g_seconds += max(0.0, horizontal_g_seconds)
        self.brake_applied = self.brake_applied or brake_applied
        self.drift_distance_m += max(0.0, drift_distance_m)
        self._record_speed_sample(
            speed_mps=max(0.0, forward_speed_mps),
            delta_seconds=bounded_delta_seconds,
        )

    def _record_speed_sample(self, *, speed_mps: float, delta_seconds: float) -> None:
        if delta_seconds <= 0.0:
            return
        self._speed_window.append((delta_seconds, speed_mps))
        self._speed_window_duration_seconds += delta_seconds
        self._speed_window_integral_m += speed_mps * delta_seconds

        excess_seconds = self._speed_window_duration_seconds - SUSTAINED_SPEED_WINDOW_SECONDS
        while excess_seconds > 1e-12 and self._speed_window:
            first_duration_seconds, first_speed_mps = self._speed_window[0]
            removed_seconds = min(excess_seconds, first_duration_seconds)
            self._speed_window_duration_seconds -= removed_seconds
            self._speed_window_integral_m -= first_speed_mps * removed_seconds
            excess_seconds -= removed_seconds
            if removed_seconds >= first_duration_seconds - 1e-12:
                self._speed_window.popleft()
            else:
                self._speed_window[0] = (first_duration_seconds - removed_seconds, first_speed_mps)

        if self._speed_window_duration_seconds >= SUSTAINED_SPEED_WINDOW_SECONDS - 1e-12:
            rolling_average_mps = self._speed_window_integral_m / SUSTAINED_SPEED_WINDOW_SECONDS
            self.sustained_top_speed_mps = max(self.sustained_top_speed_mps, rolling_average_mps)

    def completed_lap(self, *, ended_at_seconds: float, damage_at_end: float) -> dict[str, object]:
        """Return the JSON-compatible metrics for a completed lap."""
        return {
            "duration_seconds": max(0.0, ended_at_seconds - self.started_at_seconds),
            "damage_delta": max(0.0, damage_at_end - self.damage_at_start),
            "wall_contact_seconds": self.wall_contact_seconds,
            "horizontal_g_seconds": self.horizontal_g_seconds,
            "brake_applied": self.brake_applied,
            "drift_distance_m": self.drift_distance_m,
            "sustained_top_speed_mps": self.sustained_top_speed_mps,
        }


def _wrapped_angle_degrees(angle_degrees: float) -> float:
    return (angle_degrees + 180.0) % 360.0 - 180.0


def horizontal_g_seconds_for_step(
    *,
    speed_before_mps: float,
    speed_after_mps: float,
    heading_before_degrees: float,
    heading_after_degrees: float,
    delta_seconds: float,
) -> float:
    """Measure absolute horizontal g-load integrated over one physics step."""
    if delta_seconds <= 0.0:
        return 0.0
    forward_acceleration_mps2 = (speed_after_mps - speed_before_mps) / delta_seconds
    yaw_rate_degrees_per_s = _wrapped_angle_degrees(heading_after_degrees - heading_before_degrees) / delta_seconds
    lateral_acceleration_mps2 = speed_after_mps * math.radians(yaw_rate_degrees_per_s)
    horizontal_g = math.hypot(forward_acceleration_mps2, lateral_acceleration_mps2) / STANDARD_GRAVITY_MPS2
    return horizontal_g * delta_seconds


def absolute_slip_angle_degrees(*, velocity_x_mps: float, velocity_z_mps: float, heading_degrees: float) -> float:
    """Return the unsigned angle between chassis heading and horizontal velocity."""
    if math.hypot(velocity_x_mps, velocity_z_mps) <= 1e-9:
        return 0.0
    velocity_heading_degrees = math.degrees(math.atan2(velocity_x_mps, velocity_z_mps))
    return abs(_wrapped_angle_degrees(velocity_heading_degrees - heading_degrees))


def _robot_signed_speed_mps(robot: Any) -> float:
    return float(robot.vehicle.getCurrentSpeedKmHour()) / 3.6


def _robot_heading_degrees(robot: Any) -> float:
    return float(robot.chassis_np.getH())


def _robot_horizontal_velocity_mps(robot: Any) -> tuple[float, float]:
    velocity = robot.chassis_np.node().getLinearVelocity()
    return float(velocity[0]), float(velocity[2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--function", default="control")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--marshal-stuck-seconds", type=float, default=2.0)
    parser.add_argument("--marshal-penalty-m", type=float, default=5.0)
    parser.add_argument("--marshal-cooldown-seconds", type=float, default=2.0)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


class ControllerClient:
    """Request commands from a persistent, unprivileged student process."""

    def __init__(self, *, submission: Path, module_file: Path, function_name: str) -> None:
        self.submission = submission
        self.module_file = module_file
        self.function_name = function_name
        self.process: subprocess.Popen[bytes] | None = None
        self.response_fd: int | None = None

    def __enter__(self) -> ControllerClient:
        response_read_fd, response_write_fd = os.pipe()
        worker_arguments = [
            str(CONTROL_WORKER_PATH),
            "--submission",
            str(self.submission),
            "--module-file",
            str(self.module_file),
            "--function",
            self.function_name,
            "--response-fd",
            str(response_write_fd),
        ]
        process_environment: dict[str, str] | None = None
        if os.environ.get("FORMULA110_LOCAL_CONTROL") == "1":
            command = [sys.executable, *worker_arguments]
            process_environment = {**os.environ, **CPU_ONLY_ENVIRONMENT}
        else:
            runuser = shutil.which("runuser") or "/usr/sbin/runuser"
            command = [
                runuser,
                "-u",
                "student",
                "--",
                "/usr/bin/env",
                "-i",
                "HOME=/tmp",
                "PATH=/autograder/venv/bin:/usr/bin:/bin",
                "PYTHONPATH=/opt/formula110-runtime",
                *(f"{name}={value}" for name, value in CPU_ONLY_ENVIRONMENT.items()),
                "/autograder/venv/bin/python",
                *worker_arguments,
            ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                pass_fds=(response_write_fd,),
                close_fds=True,
                start_new_session=True,
                env=process_environment,
            )
        finally:
            os.close(response_write_fd)
        self.response_fd = response_read_fd
        try:
            response = self._read_response(
                timeout_seconds=CONTROLLER_STARTUP_TIMEOUT_SECONDS,
                timeout_message=f"controller startup exceeded {CONTROLLER_STARTUP_TIMEOUT_SECONDS:.0f} seconds",
            )
            if response.get("ok") is not True:
                raise RuntimeError(str(response.get("error", "controller did not load"))[:1000])
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def command(self, sensors: object) -> Any:
        from racing import RobotCommand

        if self.process is None or self.process.stdin is None or self.response_fd is None:
            raise RuntimeError("controller process is not running")
        payload = pickle.dumps(sensors, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self.process.stdin.write(struct.pack("!I", len(payload)) + payload)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError(self._unexpected_exit_message()) from error
        response = self._read_response(
            timeout_seconds=COMMAND_TIMEOUT_SECONDS,
            timeout_message=f"control call exceeded {COMMAND_TIMEOUT_SECONDS:.1f} seconds",
        )
        if response.get("ok") is not True:
            message = response.get("error", "controller call failed")
            raise RuntimeError(str(message)[:1000])
        values = tuple(float(response[key]) for key in ("throttle", "steer"))
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("control must return finite command values")
        return RobotCommand(throttle=values[0], steer=values[1])

    def close(self) -> None:
        if self.process is not None:
            process_group_id = self.process.pid
            if self.process.stdin is not None:
                self.process.stdin.close()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.process.wait(timeout=1.0)
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process_group_id, signal.SIGKILL)
            if self.process.poll() is None:
                self.process.wait(timeout=1.0)
            self.process = None
        if self.response_fd is not None:
            os.close(self.response_fd)
            self.response_fd = None

    def _read_response(self, *, timeout_seconds: float, timeout_message: str) -> dict[str, Any]:
        response_size = struct.unpack(
            "!I",
            self._read_exact(4, timeout_seconds=timeout_seconds, timeout_message=timeout_message),
        )[0]
        if response_size > MAX_RESPONSE_BYTES:
            raise RuntimeError("controller response exceeded size limit")
        try:
            response = json.loads(
                self._read_exact(
                    response_size,
                    timeout_seconds=timeout_seconds,
                    timeout_message=timeout_message,
                )
            )
        except json.JSONDecodeError as error:
            raise RuntimeError("controller returned an invalid response") from error
        if not isinstance(response, dict):
            raise RuntimeError("controller returned an invalid response")
        return response

    def _read_exact(self, byte_count: int, *, timeout_seconds: float, timeout_message: str) -> bytes:
        if self.response_fd is None:
            raise RuntimeError("controller response pipe is closed")
        chunks: list[bytes] = []
        remaining = byte_count
        deadline = time.monotonic() + timeout_seconds
        while remaining:
            self._check_memory_limit()
            timeout = deadline - time.monotonic()
            if timeout <= 0.0:
                raise TimeoutError(timeout_message)
            readable, _, _ = select.select(
                [self.response_fd],
                [],
                [],
                min(timeout, CONTROLLER_MEMORY_POLL_SECONDS),
            )
            if not readable:
                continue
            chunk = os.read(self.response_fd, remaining)
            if not chunk:
                raise RuntimeError(self._unexpected_exit_message())
            chunks.append(chunk)
            remaining -= len(chunk)
        self._check_memory_limit()
        return b"".join(chunks)

    def _check_memory_limit(self) -> None:
        if self.process is None:
            return
        resident_memory_bytes = _linux_process_tree_resident_memory_bytes(self.process.pid)
        if resident_memory_bytes is None or resident_memory_bytes <= CONTROLLER_MEMORY_LIMIT_BYTES:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)
        limit_mib = CONTROLLER_MEMORY_LIMIT_BYTES // (1024 * 1024)
        used_mib = resident_memory_bytes / (1024 * 1024)
        raise MemoryError(f"controller exceeded {limit_mib} MiB memory limit ({used_mib:.1f} MiB resident)")

    def _unexpected_exit_message(self) -> str:
        if self.process is None:
            return "controller process exited unexpectedly"
        return_code = self.process.poll()
        if return_code is None:
            with contextlib.suppress(subprocess.TimeoutExpired):
                return_code = self.process.wait(timeout=0.1)
        diagnostic = ""
        if return_code is not None and self.process.stderr is not None:
            diagnostic = self.process.stderr.read(2000).decode("utf-8", errors="replace").strip()
        message = "controller process exited unexpectedly"
        if return_code is not None:
            message += f" (exit {return_code})"
        if diagnostic:
            message += f": {diagnostic}"
        return message


def controller_client(args: argparse.Namespace) -> ControllerClient:
    return ControllerClient(
        submission=args.submission.resolve(),
        module_file=args.module_file.resolve(),
        function_name=str(args.function),
    )


def _linux_resident_memory_bytes(process_id: int) -> int | None:
    """Read one process's resident set size on the Linux grading host."""
    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    for line in status.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        fields = line.split()
        if len(fields) >= 2:
            return int(fields[1]) * 1024
    return None


def _linux_process_tree_resident_memory_bytes(process_id: int) -> int | None:
    """Sum resident memory for a controller launcher and all descendants."""
    pending = [process_id]
    visited: set[int] = set()
    total_bytes = 0
    measured_process = False
    while pending:
        current_process_id = pending.pop()
        if current_process_id in visited:
            continue
        visited.add(current_process_id)
        resident_bytes = _linux_resident_memory_bytes(current_process_id)
        if resident_bytes is not None:
            total_bytes += resident_bytes
            measured_process = True
        try:
            children_text = Path(f"/proc/{current_process_id}/task/{current_process_id}/children").read_text(
                encoding="utf-8"
            )
        except (FileNotFoundError, OSError):
            continue
        pending.extend(int(child_process_id) for child_process_id in children_text.split())
    return total_bytes if measured_process else None


def validate_controller(args: argparse.Namespace) -> dict[str, object]:
    from racing import RobotCommand, RobotSensors

    with controller_client(args) as client:
        command = client.command(RobotSensors())
    if not isinstance(command, RobotCommand):
        raise TypeError(f"{args.function} must return racing.RobotCommand")
    return {"ok": True, "callable": True}


def run_trial(args: argparse.Namespace) -> dict[str, object]:
    from racing.graphics.panda_config import configure_headless_panda
    from racing.graphics.track_rendering import add_racing_scene_collisions
    from racing.physics import (
        FORMULA_VEHICLE_PHYSICS_CONFIG,
        PhysicsScene,
        apply_robot_vehicle_command,
        apply_wall_impact_damage,
        create_physics_world,
        create_robot_vehicle,
        resolve_vehicle_actuator_command,
    )
    from racing.race.progress import default_track_progress_model, project_track_position
    from racing.race.runtime import (
        RaceCarRuntime,
        RaceRecoveryConfig,
        lap_progress_tracker_for_spawn_pose,
        maybe_marshal_race_runtimes,
        race_contact_states,
        race_scored_distance_m,
        race_spawn_poses,
        robot_is_eliminated,
        robot_score_damage,
        robot_track_point,
        update_race_runtime_after_step,
    )
    from racing.race.sensors import build_robot_sensors
    from racing.track.world import TRACK_WIDTH

    seconds = float(args.seconds)
    seed = int(args.seed)
    if seconds <= 0.0:
        raise ValueError("trial duration must be positive")
    # Exercise solo trials also call this worker and do not enable the marshal.
    marshal_stuck_seconds = getattr(args, "marshal_stuck_seconds", None)
    recovery_config = (
        RaceRecoveryConfig(
            stuck_seconds=float(marshal_stuck_seconds),
            distance_penalty_m=float(args.marshal_penalty_m),
            cooldown_seconds=float(args.marshal_cooldown_seconds),
        )
        if marshal_stuck_seconds is not None
        else None
    )
    fixed_delta_seconds = 1.0 / 60.0
    configure_headless_panda()
    showbase = cast(Any, import_module("direct.showbase.ShowBase"))
    base = showbase.ShowBase(windowType="none")
    root: Any | None = None
    try:
        model = default_track_progress_model()
        physics_world = create_physics_world()
        physics_scene = PhysicsScene(world=physics_world, vehicles=[])
        root = base.render.attachNewNode(f"gradescope-{seed}")
        add_racing_scene_collisions(physics_world=physics_world, render=root)
        spawn_pose = race_spawn_poses(
            1,
            model=model,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
            random_seed=seed,
            race_index=1,
        )[0]
        robot = create_robot_vehicle(
            world=physics_world,
            render=root,
            name=f"gradescope-{seed}-car",
            position=spawn_pose.position,
            heading_degrees=spawn_pose.heading_degrees,
            config=FORMULA_VEHICLE_PHYSICS_CONFIG,
        )
        physics_scene.vehicles.append(robot)
        runtime = RaceCarRuntime(
            robot=robot,
            tracker=lap_progress_tracker_for_spawn_pose(model=model, spawn_pose=spawn_pose),
        )
        elapsed_seconds = 0.0
        lap_crossing_times: list[float] = []
        completed_laps: list[dict[str, object]] = []
        previous_lap_count = 0
        lap_telemetry = LapTelemetryAccumulator(started_at_seconds=0.0, damage_at_start=0.0)
        with controller_client(args) as client:
            while elapsed_seconds < seconds:
                step_was_active = not robot_is_eliminated(runtime.robot)
                speed_before_mps = _robot_signed_speed_mps(runtime.robot)
                heading_before_degrees = _robot_heading_degrees(runtime.robot)
                brake_applied = False
                if step_was_active:
                    sensors, runtime.sensor_state = build_robot_sensors(
                        physics_world=physics_world,
                        robot=runtime.robot,
                        track_model=model,
                        time_s=elapsed_seconds,
                        dt_s=fixed_delta_seconds,
                        previous_state=runtime.sensor_state,
                    )
                    command = client.command(sensors)
                    actuator_command = resolve_vehicle_actuator_command(
                        command=command,
                        current_speed_kmh=float(runtime.robot.vehicle.getCurrentSpeedKmHour()),
                        config=runtime.robot.config,
                        pending_drive_direction=runtime.robot.pending_drive_direction,
                    )
                    brake_applied = actuator_command.brake_force > BRAKE_FORCE_EPSILON
                    apply_robot_vehicle_command(robot=runtime.robot, command=command)

                physics_scene.step(fixed_delta_seconds)
                next_elapsed_seconds = min(seconds, elapsed_seconds + fixed_delta_seconds)
                speed_after_mps = _robot_signed_speed_mps(runtime.robot)
                heading_after_degrees = _robot_heading_degrees(runtime.robot)
                velocity_x_mps, velocity_z_mps = _robot_horizontal_velocity_mps(runtime.robot)
                contact_state = race_contact_states(physics_world=physics_world, runtimes=(runtime,))[0]
                apply_wall_impact_damage(
                    physics_world=physics_world,
                    robots=(runtime.robot,),
                    fixed_time_step=physics_scene.fixed_time_step,
                )
                projection = project_track_position(model, robot_track_point(runtime.robot))
                update_race_runtime_after_step(
                    runtime=runtime,
                    projection=projection,
                    contact_state=contact_state,
                    elapsed_seconds=next_elapsed_seconds,
                    delta_seconds=fixed_delta_seconds,
                )
                if step_was_active:
                    slip_angle_degrees = absolute_slip_angle_degrees(
                        velocity_x_mps=velocity_x_mps,
                        velocity_z_mps=velocity_z_mps,
                        heading_degrees=heading_after_degrees,
                    )
                    wall_contact = contact_state.wall_contact > 0.0
                    drift_distance_m = (
                        max(0.0, runtime.tracker.last_counted_progress_delta_m)
                        if (
                            speed_after_mps > DRIFT_MIN_SPEED_MPS
                            and not wall_contact
                            and abs(projection.signed_distance_to_center_m) <= TRACK_WIDTH / 2
                            and DRIFT_MIN_SLIP_DEGREES <= slip_angle_degrees <= DRIFT_MAX_SLIP_DEGREES
                        )
                        else 0.0
                    )
                    lap_telemetry.record_step(
                        delta_seconds=fixed_delta_seconds,
                        wall_contact=wall_contact,
                        horizontal_g_seconds=horizontal_g_seconds_for_step(
                            speed_before_mps=speed_before_mps,
                            speed_after_mps=speed_after_mps,
                            heading_before_degrees=heading_before_degrees,
                            heading_after_degrees=heading_after_degrees,
                            delta_seconds=fixed_delta_seconds,
                        ),
                        brake_applied=brake_applied,
                        drift_distance_m=drift_distance_m,
                        forward_speed_mps=speed_after_mps,
                    )
                while previous_lap_count < runtime.tracker.lap_count:
                    lap_crossing_times.append(next_elapsed_seconds)
                    lap_damage = robot_score_damage(runtime.robot)
                    completed_laps.append(
                        lap_telemetry.completed_lap(
                            ended_at_seconds=next_elapsed_seconds,
                            damage_at_end=lap_damage,
                        )
                    )
                    lap_telemetry = LapTelemetryAccumulator(
                        started_at_seconds=next_elapsed_seconds,
                        damage_at_start=lap_damage,
                    )
                    previous_lap_count += 1
                if recovery_config is not None:
                    maybe_marshal_race_runtimes(
                        runtimes=(runtime,),
                        projections=(projection,),
                        recovery_config=recovery_config,
                        delta_seconds=fixed_delta_seconds,
                    )
                elapsed_seconds = next_elapsed_seconds

        lap_durations = [
            crossing - (lap_crossing_times[index - 1] if index else 0.0)
            for index, crossing in enumerate(lap_crossing_times)
        ]
        damage = robot_score_damage(runtime.robot)
        scored_distance_m = race_scored_distance_m(runtime)
        return {
            "ok": True,
            "seed": seed,
            "elapsed_seconds": elapsed_seconds,
            "raw_distance_m": runtime.tracker.best_distance_m,
            "scored_distance_m": scored_distance_m,
            "partial_laps": scored_distance_m / model.total_length_m,
            "raw_partial_laps": runtime.tracker.best_distance_m / model.total_length_m,
            "lap_count": runtime.tracker.lap_count,
            "damage": damage,
            "survived": not robot_is_eliminated(runtime.robot) and damage < 1.0,
            "wall_contact_seconds": runtime.tracker.wall_contact_seconds,
            "off_track_seconds": runtime.off_track_seconds,
            "marshal_count": runtime.marshal_count,
            "marshal_penalty_m": runtime.marshal_penalty_m,
            "max_speed_mps": runtime.max_speed_mps,
            "first_lap_time_seconds": lap_crossing_times[0] if lap_crossing_times else None,
            "best_lap_time_seconds": min(lap_durations) if lap_durations else None,
            "laps": completed_laps,
        }
    finally:
        if root is not None:
            root.removeNode()
        base.destroy()


def main() -> None:
    args = parse_args()
    try:
        result = validate_controller(args) if args.validate_only else run_trial(args)
    except BaseException as error:
        result = {"ok": False, "seed": args.seed, "error": f"{type(error).__name__}: {error}"[:1000]}
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
