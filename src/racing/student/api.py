"""Student-facing robot command, sensor, and controller types."""

from __future__ import annotations

import ast
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from hashlib import blake2s
from importlib import import_module, util
from importlib.machinery import ModuleSpec
from math import isfinite
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from racing.student.controller_imports import controller_module_spec
from racing.track.world import clamp

ColorRGBA = tuple[float, float, float, float]
DEFAULT_LIDAR_MAX_DISTANCE_M = float("inf")
DEFAULT_LIDAR_ANGLES_DEGREES: tuple[float, ...] = (-90.0, -45.0, -20.0, 0.0, 20.0, 45.0, 90.0)
DEFAULT_LIDAR_DISTANCES_M: tuple[float, ...] = tuple(
    DEFAULT_LIDAR_MAX_DISTANCE_M for _angle in DEFAULT_LIDAR_ANGLES_DEGREES
)
DEFAULT_CAMERA_LOOKAHEAD_DISTANCES_M: tuple[float, ...] = (4.0, 9.0, 16.0)
DEFAULT_CAMERA_LOOKAHEAD_OFFSETS_M: tuple[float, ...] = tuple(0.0 for _distance in DEFAULT_CAMERA_LOOKAHEAD_DISTANCES_M)
MAX_CAMERA_COMPETITORS = 3
RACING_NAME_GLOBAL = "RACING_NAME"
RACING_COLOR_GLOBAL = "RACING_COLOR"
CONTROLLER_FACTORY_FUNCTION = "create_controller"


@dataclass(frozen=True, slots=True)
class RobotCommand:
    """Normalized signed-throttle and steering values sent to a robot vehicle.

    Attributes:
        throttle: Signed drive command from `-1.0` to `1.0`. Positive values
            request forward drive and negative values request reverse drive.
            When the requested direction opposes the robot's motion, the
            simulator brakes before applying drive in the new direction.
            `0.0` means coasting with no drive or braking force.
        steer: Steering command from `-1.0` to `1.0`. `-1.0` means full left,
            `0.0` means straight ahead, and `1.0` means full right. The
            simulator maps this normalized value to the car's physical steering
            limit.
    """

    throttle: float = 0.0
    steer: float = 0.0


@dataclass(frozen=True, slots=True)
class ImuSensors:
    """Motion readings like a small robot IMU would report.

    Attributes:
        heading_degrees: Compass-like yaw heading in degrees. `0.0` points
            along world +Z; positive values turn toward the robot's right.
        yaw_rate_degrees_per_s: Turn rate in degrees per second. Positive
            values mean the robot is rotating to its right.
        pitch_degrees: Nose-up/nose-down tilt in degrees from the vehicle body.
        roll_degrees: Side-to-side tilt in degrees from the vehicle body.
        forward_acceleration_mps2: Acceleration along the robot's forward axis
            in meters per second squared. Positive means speeding up forward.
        lateral_acceleration_mps2: Sideways acceleration in meters per second
            squared. Positive means acceleration toward the robot's right.
    """

    heading_degrees: float = 0.0
    yaw_rate_degrees_per_s: float = 0.0
    pitch_degrees: float = 0.0
    roll_degrees: float = 0.0
    forward_acceleration_mps2: float = 0.0
    lateral_acceleration_mps2: float = 0.0


@dataclass(frozen=True, slots=True)
class OdometrySensors:
    """Estimated robot motion from wheel/vehicle odometry.

    Attributes:
        speed_mps: Signed forward speed in meters per second. Positive values
            mean forward motion; negative values mean reversing.
        distance_m: Accumulated distance traveled in meters since this
            controller run began. This behaves like wheel odometry, not a lap
            counter.
    """

    speed_mps: float = 0.0
    distance_m: float = 0.0


@dataclass(frozen=True, slots=True)
class LidarSensors:
    """Planar distance readings like a small 2D LiDAR or range finder array.

    Attributes:
        angles_degrees: Beam angles relative to the robot, in degrees. `0.0`
            points forward, negative angles point left, and positive angles
            point right.
        distances_m: Distance for each beam in meters, aligned by index with
            `angles_degrees`. A value equal to `max_distance_m` means no object
            was detected in that direction.
        max_distance_m: Maximum beam range in meters. `math.inf` means the
            simulated beam is intended to report any reachable hit.
        front_m: Convenience reading for the forward beam near `0.0` degrees.
        front_left_m: Convenience reading for the shallow front-left beam near
            `-20.0` degrees.
        front_right_m: Convenience reading for the shallow front-right beam
            near `20.0` degrees.
        left_m: Convenience reading for the left beam near `-90.0` degrees.
        right_m: Convenience reading for the right beam near `90.0` degrees.
    """

    angles_degrees: tuple[float, ...] = DEFAULT_LIDAR_ANGLES_DEGREES
    distances_m: tuple[float, ...] = DEFAULT_LIDAR_DISTANCES_M
    max_distance_m: float = DEFAULT_LIDAR_MAX_DISTANCE_M

    def __post_init__(self) -> None:
        if len(self.angles_degrees) == 0:
            raise ValueError("lidar sensors need at least one beam")
        if len(self.angles_degrees) != len(self.distances_m):
            raise ValueError("lidar angles and distances must have the same length")
        if self.max_distance_m <= 0.0:
            raise ValueError("lidar max distance must be positive")

    @property
    def front_m(self) -> float:
        """Return the distance for the beam nearest straight ahead."""
        return self.distance_at_angle_degrees(0.0)

    @property
    def front_left_m(self) -> float:
        """Return the distance for the beam nearest shallow front-left."""
        return self.distance_at_angle_degrees(-20.0)

    @property
    def front_right_m(self) -> float:
        """Return the distance for the beam nearest shallow front-right."""
        return self.distance_at_angle_degrees(20.0)

    @property
    def left_m(self) -> float:
        """Return the distance for the beam nearest left."""
        return self.distance_at_angle_degrees(-90.0)

    @property
    def right_m(self) -> float:
        """Return the distance for the beam nearest right."""
        return self.distance_at_angle_degrees(90.0)

    def distance_at_angle_degrees(self, angle_degrees: float) -> float:
        """Return the reading whose beam angle is closest to `angle_degrees`."""
        best_index = 0
        best_error = float("inf")
        for index, beam_angle_degrees in enumerate(self.angles_degrees):
            angle_error = abs(beam_angle_degrees - angle_degrees)
            if angle_error < best_error:
                best_index = index
                best_error = angle_error
        return self.distances_m[best_index]


@dataclass(frozen=True, slots=True)
class CameraCompetitorReading:
    """Processed camera reading for one opponent car.

    Attributes:
        distance_m: Distance from this robot to the competitor in meters.
        angle_degrees: Signed angle from this robot's current forward
            direction to the competitor, in degrees. Negative values are left
            of the robot, positive values are right of the robot, and `0.0`
            means straight ahead.
        relative_heading_degrees: Signed angle from this robot's current
            heading to the competitor's current heading, in degrees.
        speed_mps: Competitor forward speed in meters per second.
        closing_speed_mps: This robot's forward speed minus the competitor's
            forward speed. Positive values mean this robot is gaining.
    """

    distance_m: float
    angle_degrees: float
    relative_heading_degrees: float = 0.0
    speed_mps: float = 0.0
    closing_speed_mps: float = 0.0

    def __post_init__(self) -> None:
        if self.distance_m < 0.0:
            raise ValueError("camera competitor distance cannot be negative")


@dataclass(frozen=True, slots=True)
class CameraSensors:
    """Processed track-vision readings, not raw camera pixels.

    These values act like the output of a lane/track detector that an advanced
    robotics class might build from images. CS1 students can use the processed
    numbers directly without doing image processing.

    Attributes:
        visible: Whether processed track-center readings are available.
        center_offset_m: Signed sideways offset from the robot to the track
            center in meters. Positive values mean the center is to the robot's
            right; negative values mean it is to the left.
        heading_error_degrees: Signed angle from the robot's current heading to
            the track direction, in degrees. Positive values mean the desired
            direction turns to the robot's right.
        lookahead_offsets_m: Processed camera offsets to future track center
            points. Each value is in meters, positive to the robot's right.
        lookahead_distances_m: Forward distances in meters corresponding to
            each value in `lookahead_offsets_m`.
        competitors: Up to the three closest opponent cars. Each reading gives
            distance in meters and angle in degrees relative to this robot's
            current forward direction.
    """

    visible: bool = True
    center_offset_m: float = 0.0
    heading_error_degrees: float = 0.0
    lookahead_offsets_m: tuple[float, ...] = DEFAULT_CAMERA_LOOKAHEAD_OFFSETS_M
    lookahead_distances_m: tuple[float, ...] = DEFAULT_CAMERA_LOOKAHEAD_DISTANCES_M
    competitors: tuple[CameraCompetitorReading, ...] = ()

    def __post_init__(self) -> None:
        if len(self.lookahead_offsets_m) != len(self.lookahead_distances_m):
            raise ValueError("camera lookahead offsets and distances must have the same length")
        if len(self.competitors) > MAX_CAMERA_COMPETITORS:
            raise ValueError("camera competitors can include at most three readings")


@dataclass(frozen=True, slots=True)
class ContactSensors:
    """Touch/contact and accumulated damage readings for collisions.

    Attributes:
        wall: Seconds of continuous current contact with a track barrier.
        robot: Seconds of continuous current contact with another robot or blocker.
        any_contact: Seconds of continuous current contact with any object.
        damage: Accumulated vehicle damage from `0.0` to `1.0`, where `1.0`
            means the robot is fully damaged/eliminated.
    """

    wall: float = 0.0
    robot: float = 0.0
    any_contact: float = 0.0
    damage: float = 0.0

    def __post_init__(self) -> None:
        wall = max(0.0, float(self.wall))
        robot = max(0.0, float(self.robot))
        object.__setattr__(self, "wall", wall)
        object.__setattr__(self, "robot", robot)
        object.__setattr__(self, "any_contact", max(0.0, float(self.any_contact), wall, robot))
        object.__setattr__(self, "damage", clamp(self.damage, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class RobotSensors:
    """Student-facing sensor snapshot for one control tick.

    Attributes:
        dt_s: Seconds since the previous controller snapshot.
        tick: Zero-based controller tick number.
        imu: Orientation and acceleration readings from `ImuSensors`.
        odometry: Estimated speed and traveled distance from `OdometrySensors`.
        lidar: Distance readings from `LidarSensors` that detect nearby
            barriers, robots, and blockers.
        wall_lidar: Distance readings from `LidarSensors` that detect track
            barriers only.
        camera: Processed track-vision readings from `CameraSensors`. These are
            detector outputs, not raw image pixels.
        contact: Collision/touch readings from `ContactSensors`.
    """

    dt_s: float = 0.0
    tick: int = 0
    imu: ImuSensors = field(default_factory=ImuSensors)
    odometry: OdometrySensors = field(default_factory=OdometrySensors)
    lidar: LidarSensors = field(default_factory=LidarSensors)
    wall_lidar: LidarSensors = field(default_factory=LidarSensors)
    camera: CameraSensors = field(default_factory=CameraSensors)
    contact: ContactSensors = field(default_factory=ContactSensors)


class RobotController(Protocol):
    """Callable protocol implemented by student robot controllers."""

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        """Return the next remote-control command for one simulation tick."""
        ...


class RobotControllerFactory(Protocol):
    """Factory that returns an independent controller for one car and race."""

    def __call__(self) -> RobotController:
        """Create a fresh runtime controller instance."""
        ...


@dataclass(frozen=True, slots=True)
class StudentControllerSubmission:
    """Loaded student controller plus optional display metadata."""

    controller: RobotController
    display_name: str | None = None
    car_color: ColorRGBA | None = None


def clamp_command(command: RobotCommand) -> RobotCommand:
    """Validate and clamp a robot command to the normalized actuator ranges."""
    values = (command.throttle, command.steer)
    if not all(isfinite(value) for value in values):
        raise ValueError("robot command values must be finite numbers")
    return RobotCommand(
        throttle=clamp(command.throttle, -1.0, 1.0),
        steer=clamp(command.steer, -1.0, 1.0),
    )


def default_student_controller(sensors: RobotSensors) -> RobotCommand:
    """Drive with a small starter strategy that follows the center of the track."""
    if sensors.contact.any_contact:
        open_side_steer = -0.45 if sensors.lidar.left_m > sensors.lidar.right_m else 0.45
        return RobotCommand(throttle=-0.15, steer=open_side_steer)

    if sensors.lidar.front_m < 1.0:
        open_side_steer = -0.55 if sensors.lidar.left_m > sensors.lidar.right_m else 0.55
        return RobotCommand(throttle=-0.8, steer=open_side_steer)

    if not sensors.camera.visible:
        return RobotCommand(throttle=0.08, steer=0.0)

    center_correction = clamp(sensors.camera.center_offset_m * 0.18, -0.45, 0.45)
    heading_correction = clamp(sensors.camera.heading_error_degrees / 90.0, -0.45, 0.45)
    steer = clamp(center_correction + heading_correction, -0.8, 0.8)
    target_speed_mps = 3.0 if abs(sensors.camera.heading_error_degrees) < 25.0 else 1.5
    speed_error = target_speed_mps - sensors.odometry.speed_mps
    throttle = -0.15 if speed_error < -1.0 else clamp(speed_error * 0.20, 0.0, 0.45)
    return RobotCommand(throttle=throttle, steer=steer)


def load_student_controller(
    module_reference: str | Path,
    *,
    function_name: str = "control",
    suppress_prints: bool = False,
) -> RobotController:
    """Load a student controller function from a module path or import name."""
    return load_student_submission(
        module_reference, function_name=function_name, suppress_prints=suppress_prints
    ).controller


def load_student_submission(
    module_reference: str | Path,
    *,
    function_name: str = "control",
    suppress_prints: bool = False,
) -> StudentControllerSubmission:
    """Load a controller and optional ``RACING_NAME``/``RACING_COLOR`` metadata.

    A module-level ``create_controller()`` factory is preferred for the default
    ``control`` function name. The factory lets every car and repeated race get
    independent mutable controller state. Modules without a factory retain the
    original function-based ``control(sensors)`` interface.

    With ``suppress_prints``, direct ``print(...)`` calls in this module are
    skipped, including their arguments, without editing the source file.
    """
    if not function_name:
        raise ValueError("student control function name cannot be empty")

    module = _load_student_module(module_reference, suppress_prints=suppress_prints)
    return StudentControllerSubmission(
        controller=_student_controller_from_module(module=module, function_name=function_name),
        display_name=_student_display_name(module),
        car_color=_student_car_color(module),
    )


def _student_controller_from_module(*, module: ModuleType, function_name: str) -> RobotController:
    if function_name == "control" and hasattr(module, CONTROLLER_FACTORY_FUNCTION):
        return _student_controller_from_factory(module=module)

    raw_function = getattr(module, function_name, None)
    if raw_function is None:
        raise AttributeError(f"student module {module.__name__!r} does not define {function_name!r}")
    if not callable(raw_function):
        raise TypeError(f"student module attribute {function_name!r} must be callable")

    return _ValidatedRobotController(
        raw_controller=raw_function,
        label=f"{module.__name__}.{function_name}",
    )


def _student_controller_from_factory(*, module: ModuleType) -> RobotController:
    raw_factory = getattr(module, CONTROLLER_FACTORY_FUNCTION)
    if not callable(raw_factory):
        raise TypeError(f"student module attribute {CONTROLLER_FACTORY_FUNCTION!r} must be callable")
    factory = cast(Callable[[], object], raw_factory)
    return _controller_created_by_factory(factory=factory, module_name=module.__name__)


def _controller_created_by_factory(*, factory: Callable[[], object], module_name: str) -> RobotController:
    raw_controller = factory()
    if not callable(raw_controller):
        raise TypeError(
            f"student controller factory {module_name}.{CONTROLLER_FACTORY_FUNCTION} must return a callable, "
            f"got {type(raw_controller).__name__}"
        )
    return _ValidatedRobotController(
        raw_controller=raw_controller,
        label=f"{module_name}.{CONTROLLER_FACTORY_FUNCTION}()",
        factory=factory,
        module_name=module_name,
    )


class _ValidatedRobotController:
    """Runtime return-type validation plus optional fresh-controller creation."""

    def __init__(
        self,
        *,
        raw_controller: object,
        label: str,
        factory: Callable[[], object] | None = None,
        module_name: str | None = None,
    ) -> None:
        if not callable(raw_controller):
            raise TypeError(f"student controller {label} must be callable")
        self._controller = cast(Callable[[RobotSensors], object], raw_controller)
        self._label = label
        self._factory = factory
        self._module_name = module_name

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        command = self._controller(sensors)
        if not isinstance(command, RobotCommand):
            raise TypeError(
                f"student controller {self._label} must return racing.RobotCommand, got {type(command).__name__}"
            )
        return command

    def copy_for_car(self) -> RobotController:
        """Return fresh factory/copy state, or this wrapper for a function controller."""
        if self._factory is not None and self._module_name is not None:
            return _controller_created_by_factory(factory=self._factory, module_name=self._module_name)
        copy_for_car = getattr(self._controller, "copy_for_car", None)
        if not callable(copy_for_car):
            return self
        raw_controller = copy_for_car()
        return _ValidatedRobotController(
            raw_controller=raw_controller,
            label=f"{self._label}.copy_for_car()",
        )


def _student_display_name(module: ModuleType) -> str | None:
    if not hasattr(module, RACING_NAME_GLOBAL):
        return None
    raw_name = getattr(module, RACING_NAME_GLOBAL)
    if not isinstance(raw_name, str):
        raise ValueError(f"student module {module.__name__!r} {RACING_NAME_GLOBAL} must be a non-empty string")
    display_name = raw_name.strip()
    if not display_name:
        raise ValueError(f"student module {module.__name__!r} {RACING_NAME_GLOBAL} must be a non-empty string")
    return display_name


def _student_car_color(module: ModuleType) -> ColorRGBA | None:
    if not hasattr(module, RACING_COLOR_GLOBAL):
        return None
    raw_color = getattr(module, RACING_COLOR_GLOBAL)
    try:
        return _parse_student_color(raw_color)
    except ValueError as error:
        raise ValueError(f"student module {module.__name__!r} {RACING_COLOR_GLOBAL} {error}") from error


def _parse_student_color(raw_color: object) -> ColorRGBA:
    if isinstance(raw_color, str):
        return _parse_student_color_text(raw_color)
    if isinstance(raw_color, (tuple, list)):
        return _parse_student_color_channels(cast(tuple[object, ...] | list[object], raw_color))
    raise ValueError("must be a #RRGGBB, #RRGGBBAA, or 3- or 4-channel normalized color")


def _parse_student_color_text(value: str) -> ColorRGBA:
    text = value.strip()
    if text.startswith("#"):
        hex_text = text[1:]
        if len(hex_text) not in (6, 8):
            raise ValueError("must use #RRGGBB or #RRGGBBAA hex color text")
        try:
            channels = tuple(int(hex_text[index : index + 2], 16) / 255 for index in range(0, len(hex_text), 2))
        except ValueError as error:
            raise ValueError("must use hexadecimal digits in color text") from error
        return _color_from_channels(channels)

    parts = tuple(part.strip() for part in text.split(","))
    if len(parts) not in (3, 4):
        raise ValueError("must have three or four comma-separated color channels")
    try:
        channels = tuple(float(part) for part in parts)
    except ValueError as error:
        raise ValueError("must use numeric comma-separated color channels") from error
    return _color_from_channels(channels)


def _parse_student_color_channels(raw_channels: tuple[object, ...] | list[object]) -> ColorRGBA:
    channels: list[float] = []
    for channel in raw_channels:
        if isinstance(channel, bool) or not isinstance(channel, (int, float)):
            raise ValueError("must use numeric color channels")
        channels.append(float(channel))
    return _color_from_channels(channels)


def _color_from_channels(channels: tuple[float, ...] | list[float]) -> ColorRGBA:
    if len(channels) not in (3, 4):
        raise ValueError("must have three or four color channels")
    if any(channel < 0.0 or channel > 1.0 for channel in channels):
        raise ValueError("must use color channels between 0.0 and 1.0")
    if len(channels) == 3:
        return (channels[0], channels[1], channels[2], 1.0)
    return (channels[0], channels[1], channels[2], channels[3])


class _SkipPrintCalls(ast.NodeTransformer):
    """Skip debug output without evaluating arguments or leaving empty blocks."""

    def visit_Call(self, node: ast.Call) -> ast.AST:
        if isinstance(node.func, ast.Name) and node.func.id == "print":
            return ast.copy_location(ast.Constant(value=None), node)
        return self.generic_visit(node)


def _load_student_module(module_reference: str | Path, *, suppress_prints: bool = False) -> ModuleType:
    reference_text = str(module_reference)
    if _looks_like_file_reference(reference_text):
        return _load_student_module_from_path(Path(reference_text), suppress_prints=suppress_prints)
    if suppress_prints:
        spec = util.find_spec(reference_text)
        if spec is None:
            raise ImportError(f"could not find student module {reference_text!r}")
        return _execute_student_module(spec, suppress_prints=True)
    return import_module(reference_text)


def _looks_like_file_reference(reference_text: str) -> bool:
    return reference_text.endswith(".py") or "/" in reference_text or "\\" in reference_text


def _load_student_module_from_path(path: Path, *, suppress_prints: bool = False) -> ModuleType:
    module_path = path.expanduser().resolve()
    if not module_path.is_file():
        raise FileNotFoundError(f"student module file does not exist: {module_path}")

    spec = controller_module_spec(module_path, _student_module_name(module_path))
    if spec is None:
        module_directory = str(module_path.parent)
        if module_directory not in sys.path:
            sys.path.insert(0, module_directory)
        spec = util.spec_from_file_location(_student_module_name(module_path), module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load student module from {module_path}")

    module = _execute_student_module(spec, suppress_prints=suppress_prints)
    parent_name, _, child_name = spec.name.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child_name, module)
    return module


def _execute_student_module(spec: ModuleSpec, *, suppress_prints: bool) -> ModuleType:
    loader = cast(Any, spec.loader)
    code = None
    if suppress_prints:
        source = loader.get_source(spec.name)
        if source is None:
            raise ImportError(f"cannot suppress prints without Python source for {spec.name!r}")
        filename = spec.origin or spec.name
        tree = _SkipPrintCalls().visit(ast.parse(source, filename=filename))
        code = compile(ast.fix_missing_locations(tree), filename, "exec", dont_inherit=True)

    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if code is not None:
        exec(code, module.__dict__)
    else:
        loader.exec_module(module)
    return module


def _student_module_name(module_path: Path) -> str:
    safe_stem = "".join(character if character.isalnum() or character == "_" else "_" for character in module_path.stem)
    digest = blake2s(str(module_path).encode("utf-8"), digest_size=4).hexdigest()
    return f"_racing_student_{safe_stem}_{digest}"
