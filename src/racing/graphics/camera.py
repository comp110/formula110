"""Camera modes and math for framing the car and track."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import atan2, cos, degrees, exp, radians, sin
from typing import Any, TypeAlias

from racing.game.config import CameraView
from racing.graphics.track_rendering import (
    TRACK_EDGE_BUFFER,
    TRACK_SURFACE_Y,
    TRACK_WALL_BASE_Y,
    TRACK_WALL_HEIGHT,
    TRACK_WALL_THICKNESS,
)
from racing.race.progress import TrackProgressModel, project_track_position, track_pose_at_distance
from racing.track.spatial import node_position, track_forward_vector
from racing.track.world import TRACK_SCALE, TRACK_WIDTH, TrackPoint, sampled_track_centerline, track_bounds

TRACK_CAMERA_MARGIN = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS
TRACK_CAMERA_VIEWPORT_FILL = 0.95
DEFAULT_VIEWPORT_ASPECT_RATIO = 16 / 9
TOP_DOWN_CAMERA_HEIGHT = 54 * TRACK_SCALE
THREE_QUARTER_CAMERA_HEADING_DEGREES = 34.0
THREE_QUARTER_CAMERA_PITCH_DEGREES = -40.0
THREE_QUARTER_CAMERA_DISTANCE = 25 * TRACK_SCALE
THREE_QUARTER_CAMERA_HEIGHT = 22 * TRACK_SCALE
THREE_QUARTER_CAMERA_ZOOM = 0.85
TRACK_OVERVIEW_CAMERA_ROTATION_DEGREES = 0.0
FOLLOW_CAMERA_DISTANCE = 7.5
FOLLOW_CAMERA_HEIGHT = 4.0
FOLLOW_CAMERA_FOV = 60
FOLLOW_CAMERA_PITCH_DEGREES = -24.0
FOLLOW_CAMERA_LOOK_AHEAD = 2.8
FOLLOW_CAMERA_LOOK_HEIGHT = 0.5
DRONE_CAMERA_DISTANCE = 14.12
DRONE_CAMERA_HEIGHT = 10.17
DRONE_CAMERA_FOV = 64
DRONE_CAMERA_LOOK_AHEAD = 30.0
DRONE_CAMERA_LOOK_HEIGHT = -15.0
FORMULA_FOLLOW_CAMERA_DISTANCE = 4.0
FORMULA_FOLLOW_CAMERA_HEIGHT = 1.66
FORMULA_FOLLOW_CAMERA_FOV = 64
FORMULA_FOLLOW_CAMERA_LOOK_AHEAD = 12.0
FORMULA_FOLLOW_CAMERA_LOOK_HEIGHT = 0.0
FORMULA_DRONE_CAMERA_DISTANCE = DRONE_CAMERA_DISTANCE
FORMULA_DRONE_CAMERA_HEIGHT = DRONE_CAMERA_HEIGHT
FORMULA_DRONE_CAMERA_FOV = DRONE_CAMERA_FOV
FORMULA_DRONE_CAMERA_LOOK_AHEAD = DRONE_CAMERA_LOOK_AHEAD
FORMULA_DRONE_CAMERA_LOOK_HEIGHT = DRONE_CAMERA_LOOK_HEIGHT
FOLLOW_CAMERA_TRACK_LOOKAHEAD_M = 30.0
FOLLOW_CAMERA_DIRECTION_RESPONSE_SECONDS = 0.16
FOLLOW_CAMERA_AVERAGING_SECONDS = 0.5
MIN_FOLLOW_FORWARD_LENGTH = 0.001
FollowForwardSample: TypeAlias = tuple[float, float, float]


def _new_follow_forward_samples() -> list[FollowForwardSample]:
    return []


@dataclass(slots=True)
class CameraRig:
    """Mutable camera-cycle state for the current scene."""

    view: CameraView = CameraView.TOP_DOWN
    cycle_key_was_down: bool = False
    follow_heading_degrees: float = 0.0
    follow_forward_samples: list[FollowForwardSample] = field(default_factory=_new_follow_forward_samples)
    follow_target_id: int | None = None
    follow_direction_initialized: bool = False
    selected_car_id: str | None = None

    def select_follow_car(self, car_id: str | None) -> None:
        """Follow a stable entrant until another selection replaces it."""
        self.selected_car_id = car_id
        self.view = CameraView.FOLLOW
        self.reset_follow_history()

    def reset_follow_history(self) -> None:
        """Forget recent follow-camera direction samples."""
        self.follow_forward_samples.clear()
        self.follow_target_id = None
        self.follow_direction_initialized = False


@dataclass(frozen=True, slots=True)
class TrackCameraFrame:
    """Track bounds projected into camera-framing coordinates."""

    center_x: float
    center_z: float
    width: float
    length: float


@dataclass(frozen=True, slots=True)
class FollowCameraSettings:
    """Distance and aim point settings for the close chase camera."""

    distance: float = FOLLOW_CAMERA_DISTANCE
    height: float = FOLLOW_CAMERA_HEIGHT
    fov: float = FOLLOW_CAMERA_FOV
    look_ahead: float = FOLLOW_CAMERA_LOOK_AHEAD
    look_height: float = FOLLOW_CAMERA_LOOK_HEIGHT
    track_lookahead_m: float = FOLLOW_CAMERA_TRACK_LOOKAHEAD_M
    direction_response_seconds: float = FOLLOW_CAMERA_DIRECTION_RESPONSE_SECONDS
    uses_track_lead: bool = False


DEFAULT_FOLLOW_CAMERA_SETTINGS = FollowCameraSettings()
DRONE_CAMERA_SETTINGS = FollowCameraSettings(
    distance=DRONE_CAMERA_DISTANCE,
    height=DRONE_CAMERA_HEIGHT,
    fov=DRONE_CAMERA_FOV,
    look_ahead=DRONE_CAMERA_LOOK_AHEAD,
    look_height=DRONE_CAMERA_LOOK_HEIGHT,
    track_lookahead_m=FOLLOW_CAMERA_TRACK_LOOKAHEAD_M,
    direction_response_seconds=FOLLOW_CAMERA_DIRECTION_RESPONSE_SECONDS,
    uses_track_lead=True,
)
FORMULA_FOLLOW_CAMERA_SETTINGS = FollowCameraSettings(
    distance=FORMULA_FOLLOW_CAMERA_DISTANCE,
    height=FORMULA_FOLLOW_CAMERA_HEIGHT,
    fov=FORMULA_FOLLOW_CAMERA_FOV,
    look_ahead=FORMULA_FOLLOW_CAMERA_LOOK_AHEAD,
    look_height=FORMULA_FOLLOW_CAMERA_LOOK_HEIGHT,
    track_lookahead_m=0.0,
    direction_response_seconds=FOLLOW_CAMERA_DIRECTION_RESPONSE_SECONDS,
)
FORMULA_DRONE_CAMERA_SETTINGS = FollowCameraSettings(
    distance=FORMULA_DRONE_CAMERA_DISTANCE,
    height=FORMULA_DRONE_CAMERA_HEIGHT,
    fov=FORMULA_DRONE_CAMERA_FOV,
    look_ahead=FORMULA_DRONE_CAMERA_LOOK_AHEAD,
    look_height=FORMULA_DRONE_CAMERA_LOOK_HEIGHT,
    track_lookahead_m=FOLLOW_CAMERA_TRACK_LOOKAHEAD_M,
    direction_response_seconds=FOLLOW_CAMERA_DIRECTION_RESPONSE_SECONDS,
    uses_track_lead=True,
)


def next_camera_view(view: CameraView, *, include_split: bool = False) -> CameraView:
    """Pick the next view when the player presses the camera-cycle key."""
    if view is CameraView.TOP_DOWN:
        return CameraView.THREE_QUARTER
    if view is CameraView.THREE_QUARTER:
        return CameraView.DRONE
    if view in (CameraView.DRONE, CameraView.FOLLOW_CAR):
        return CameraView.FOLLOW
    if view is CameraView.FOLLOW and include_split:
        return CameraView.SPLIT_FOLLOW
    return CameraView.TOP_DOWN


def update_camera_cycle(rig: CameraRig, *, cycle_key_down: bool, include_split: bool = False) -> None:
    """Advance the camera mode once for each key press."""
    if cycle_key_down and not rig.cycle_key_was_down:
        rig.view = next_camera_view(rig.view, include_split=include_split)
        rig.reset_follow_history()
    rig.cycle_key_was_down = cycle_key_down


def apply_camera_view(
    *,
    ursina: Any,
    view: CameraView,
    target: Any,
    rig: CameraRig | None = None,
    delta_seconds: float = 0.0,
    follow_settings: FollowCameraSettings = DEFAULT_FOLLOW_CAMERA_SETTINGS,
    track_model: TrackProgressModel | None = None,
) -> None:
    """Move the Ursina camera to match the requested simulator view."""
    camera_frame = _track_camera_frame(None if track_model is None else track_model.points)
    viewport_aspect = _viewport_aspect_ratio(ursina)
    ursina.camera.parent = ursina.scene

    if view is CameraView.TOP_DOWN:
        heading_degrees = rotated_overview_camera_heading_degrees(
            top_down_camera_heading_for_viewport(frame=camera_frame, viewport_aspect=viewport_aspect)
        )
        ursina.camera.orthographic = True
        ursina.camera.position = (camera_frame.center_x, TOP_DOWN_CAMERA_HEIGHT, camera_frame.center_z)
        ursina.camera.setHpr(heading_degrees, -90, 0)
        ursina.camera.fov = _track_orthographic_fov(
            frame=camera_frame,
            viewport_aspect=viewport_aspect,
            heading_degrees=heading_degrees,
            pitch_degrees=-90.0,
            max_y=0.0,
        )
        return

    if view is CameraView.THREE_QUARTER:
        heading_degrees = rotated_overview_camera_heading_degrees(THREE_QUARTER_CAMERA_HEADING_DEGREES)
        heading = radians(heading_degrees)
        ursina.camera.orthographic = True
        ursina.camera.position = (
            camera_frame.center_x + sin(heading) * THREE_QUARTER_CAMERA_DISTANCE,
            THREE_QUARTER_CAMERA_HEIGHT,
            camera_frame.center_z - cos(heading) * THREE_QUARTER_CAMERA_DISTANCE,
        )
        ursina.camera.look_at((camera_frame.center_x, TRACK_SURFACE_Y, camera_frame.center_z))
        ursina.camera.setR(0.0)
        ursina.camera.fov = (
            _track_orthographic_fov(
                frame=camera_frame,
                viewport_aspect=viewport_aspect,
                heading_degrees=heading_degrees,
                pitch_degrees=THREE_QUARTER_CAMERA_PITCH_DEGREES,
                max_y=TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT,
            )
            * THREE_QUARTER_CAMERA_ZOOM
        )
        return

    ursina.camera.orthographic = False
    ursina.camera.fov = follow_settings.fov
    apply_follow_camera_view(
        ursina=ursina,
        camera=ursina.camera,
        lens=ursina.camera.perspective_lens,
        target=target,
        rig=rig,
        delta_seconds=delta_seconds,
        follow_settings=follow_settings,
        track_model=track_model,
    )


def apply_follow_camera_view(
    *,
    ursina: Any,
    camera: Any,
    lens: Any,
    target: Any,
    rig: CameraRig | None = None,
    delta_seconds: float = 0.0,
    follow_settings: FollowCameraSettings = DEFAULT_FOLLOW_CAMERA_SETTINGS,
    track_model: TrackProgressModel | None = None,
) -> None:
    """Apply the same follow pose to any scene camera with its own lens and history."""
    target_x, target_y, target_z = node_position(target)
    camera.parent = ursina.scene
    fallback_heading_degrees = rig.follow_heading_degrees if rig is not None else float(target.getH())
    chassis_forward_x, chassis_forward_z, chassis_heading_degrees = follow_camera_forward(
        target=target,
        ursina=ursina,
        fallback_heading_degrees=fallback_heading_degrees,
    )
    if follow_settings.uses_track_lead:
        raw_forward_x, raw_forward_z, raw_heading_degrees = follow_camera_track_forward(
            model=track_model,
            target_x=target_x,
            target_z=target_z,
            lookahead_m=follow_settings.track_lookahead_m,
            fallback_forward_x=chassis_forward_x,
            fallback_forward_z=chassis_forward_z,
            fallback_heading_degrees=chassis_heading_degrees,
        )
    else:
        raw_forward_x, raw_forward_z, raw_heading_degrees = (
            chassis_forward_x,
            chassis_forward_z,
            chassis_heading_degrees,
        )
    if rig is not None:
        target_id = id(target)
        if rig.follow_target_id is not None and rig.follow_target_id != target_id:
            rig.reset_follow_history()
        rig.follow_target_id = target_id
        forward_x, forward_z, heading_degrees = smoothed_follow_forward(
            current_heading_degrees=rig.follow_heading_degrees,
            target_forward_x=raw_forward_x,
            target_forward_z=raw_forward_z,
            sample_duration_s=delta_seconds,
            response_seconds=follow_settings.direction_response_seconds,
            initialized=rig.follow_direction_initialized,
        )
        rig.follow_heading_degrees = heading_degrees
        rig.follow_direction_initialized = True
    else:
        forward_x, forward_z, heading_degrees = raw_forward_x, raw_forward_z, raw_heading_degrees
    camera_x, camera_z = follow_camera_position(
        target_x=target_x,
        target_z=target_z,
        forward_x=forward_x,
        forward_z=forward_z,
        distance=follow_settings.distance,
    )
    lens.setFov(follow_settings.fov)
    camera.position = (
        camera_x,
        target_y + follow_settings.height,
        camera_z,
    )
    camera.look_at(
        (
            target_x + forward_x * follow_settings.look_ahead,
            target_y + follow_settings.look_height,
            target_z + forward_z * follow_settings.look_ahead,
        )
    )
    camera.setR(0.0)


@lru_cache(maxsize=16)
def _track_camera_frame(points: tuple[TrackPoint, ...] | None = None) -> TrackCameraFrame:
    track_points = sampled_track_centerline(samples_per_segment=10) if points is None else points
    bounds = track_bounds(points=track_points, margin=TRACK_CAMERA_MARGIN)
    center_x = (bounds.min_x + bounds.max_x) / 2
    center_z = (bounds.min_z + bounds.max_z) / 2
    return TrackCameraFrame(center_x=center_x, center_z=center_z, width=bounds.width, length=bounds.length)


def _track_orthographic_fov(
    *,
    frame: TrackCameraFrame,
    viewport_aspect: float,
    heading_degrees: float,
    pitch_degrees: float,
    max_y: float,
) -> float:
    projected_width, projected_height = projected_track_size(
        frame=frame,
        heading_degrees=heading_degrees,
        pitch_degrees=pitch_degrees,
        max_y=max_y,
    )
    return orthographic_fov_for_viewport(
        projected_width=projected_width,
        projected_height=projected_height,
        viewport_aspect=viewport_aspect,
        fill=TRACK_CAMERA_VIEWPORT_FILL,
    )


def top_down_camera_heading_for_viewport(*, frame: TrackCameraFrame, viewport_aspect: float) -> float:
    """Choose a top-down angle that fits the whole track in the window."""
    if frame.width <= 0 or frame.length <= 0:
        raise ValueError("track camera frame must be positive")
    if viewport_aspect <= 0:
        raise ValueError("viewport_aspect must be positive")
    if frame.width / frame.length <= viewport_aspect:
        return 0.0

    numerator = frame.width - viewport_aspect * frame.length
    denominator = viewport_aspect * frame.width - frame.length
    if denominator <= 0:
        return 0.0
    return degrees(atan2(numerator, denominator))


def rotated_overview_camera_heading_degrees(heading_degrees: float) -> float:
    """Return the camera heading used for track overview views."""
    return (heading_degrees + TRACK_OVERVIEW_CAMERA_ROTATION_DEGREES) % 360.0


def projected_track_size(
    *,
    frame: TrackCameraFrame,
    heading_degrees: float,
    pitch_degrees: float,
    max_y: float,
) -> tuple[float, float]:
    """Measure how wide and tall the track appears from a camera angle."""
    heading = radians(heading_degrees)
    pitch = radians(abs(pitch_degrees))
    sin_heading = sin(heading)
    cos_heading = cos(heading)
    sin_pitch = sin(pitch)
    cos_pitch = cos(pitch)
    half_width = frame.width / 2
    half_length = frame.length / 2
    projected_x: list[float] = []
    projected_y: list[float] = []

    for dx in (-half_width, half_width):
        for dz in (-half_length, half_length):
            for y in (0.0, max_y):
                projected_x.append(dx * cos_heading + dz * sin_heading)
                projected_y.append((-dx * sin_heading + dz * cos_heading) * sin_pitch + y * cos_pitch)

    return max(projected_x) - min(projected_x), max(projected_y) - min(projected_y)


def follow_camera_position(
    *,
    target_x: float,
    target_z: float,
    forward_x: float,
    forward_z: float,
    distance: float,
) -> tuple[float, float]:
    """Return the X/Z follow-camera position behind a horizontal forward vector."""
    if distance <= 0:
        raise ValueError("distance must be positive")
    return target_x - forward_x * distance, target_z - forward_z * distance


def follow_camera_forward(
    *,
    target: Any,
    ursina: Any,
    fallback_heading_degrees: float,
) -> tuple[float, float, float]:
    """Return a stable horizontal target-forward vector and heading for follow view."""
    fallback_x, fallback_z = track_forward_vector(fallback_heading_degrees)
    try:
        forward = target.getQuat(ursina.scene).xform(ursina.Vec3(0.0, 0.0, 1.0))
        forward_x = float(forward[0])
        forward_z = float(forward[2])
    except (AttributeError, IndexError, TypeError, ValueError):
        return fallback_x, fallback_z, fallback_heading_degrees

    return normalized_follow_forward(
        forward_x=forward_x,
        forward_z=forward_z,
        fallback_heading_degrees=fallback_heading_degrees,
    )


def follow_camera_track_forward(
    *,
    model: TrackProgressModel | None,
    target_x: float,
    target_z: float,
    lookahead_m: float,
    fallback_forward_x: float,
    fallback_forward_z: float,
    fallback_heading_degrees: float,
) -> tuple[float, float, float]:
    """Return the desired follow direction toward a centerline point ahead of the car."""
    if model is None or lookahead_m <= 0.0:
        return fallback_forward_x, fallback_forward_z, fallback_heading_degrees

    projection = project_track_position(model, TrackPoint(target_x, target_z))
    lookahead_pose = track_pose_at_distance(model, projection.progress_distance_m + lookahead_m)
    return normalized_follow_forward(
        forward_x=lookahead_pose.position.x - target_x,
        forward_z=lookahead_pose.position.z - target_z,
        fallback_heading_degrees=fallback_heading_degrees,
    )


def normalized_follow_forward(
    *,
    forward_x: float,
    forward_z: float,
    fallback_heading_degrees: float,
) -> tuple[float, float, float]:
    """Project a possibly rolled or pitched chassis forward vector into the track plane."""
    horizontal_length = (forward_x * forward_x + forward_z * forward_z) ** 0.5
    if horizontal_length < MIN_FOLLOW_FORWARD_LENGTH:
        fallback_x, fallback_z = track_forward_vector(fallback_heading_degrees)
        return fallback_x, fallback_z, fallback_heading_degrees

    normalized_x = forward_x / horizontal_length
    normalized_z = forward_z / horizontal_length
    heading_degrees = degrees(atan2(normalized_x, normalized_z))
    return normalized_x, normalized_z, heading_degrees


def smoothed_follow_forward(
    *,
    current_heading_degrees: float,
    target_forward_x: float,
    target_forward_z: float,
    sample_duration_s: float,
    response_seconds: float,
    initialized: bool,
) -> tuple[float, float, float]:
    """Ease the follow direction toward a target vector without trailing old headings."""
    target_x, target_z, target_heading_degrees = normalized_follow_forward(
        forward_x=target_forward_x,
        forward_z=target_forward_z,
        fallback_heading_degrees=current_heading_degrees,
    )
    if not initialized or sample_duration_s <= 0.0 or response_seconds <= 0.0:
        return target_x, target_z, target_heading_degrees

    current_x, current_z = track_forward_vector(current_heading_degrees)
    blend = 1.0 - exp(-sample_duration_s / response_seconds)
    blended_x = current_x + (target_x - current_x) * blend
    blended_z = current_z + (target_z - current_z) * blend
    return normalized_follow_forward(
        forward_x=blended_x,
        forward_z=blended_z,
        fallback_heading_degrees=target_heading_degrees,
    )


def averaged_follow_forward(
    *,
    samples: list[FollowForwardSample],
    forward_x: float,
    forward_z: float,
    sample_duration_s: float,
    fallback_heading_degrees: float,
    window_seconds: float = FOLLOW_CAMERA_AVERAGING_SECONDS,
) -> tuple[float, float, float]:
    """Return the time-weighted average direction from recent follow samples."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")

    sample_duration_s = max(sample_duration_s, 0.0)
    if sample_duration_s > 0.0:
        samples.append((sample_duration_s, forward_x, forward_z))
        _trim_follow_forward_samples(samples=samples, window_seconds=window_seconds)

    total_duration_s = sum(duration_s for duration_s, _, _ in samples)
    if total_duration_s <= 0.0:
        return forward_x, forward_z, degrees(atan2(forward_x, forward_z))

    averaged_x = sum(duration_s * sample_x for duration_s, sample_x, _ in samples) / total_duration_s
    averaged_z = sum(duration_s * sample_z for duration_s, _, sample_z in samples) / total_duration_s
    return normalized_follow_forward(
        forward_x=averaged_x,
        forward_z=averaged_z,
        fallback_heading_degrees=fallback_heading_degrees,
    )


def _trim_follow_forward_samples(*, samples: list[FollowForwardSample], window_seconds: float) -> None:
    total_duration_s = sum(duration_s for duration_s, _, _ in samples)
    while samples and total_duration_s > window_seconds:
        overflow_s = total_duration_s - window_seconds
        duration_s, forward_x, forward_z = samples[0]
        if overflow_s >= duration_s:
            samples.pop(0)
            total_duration_s -= duration_s
        else:
            samples[0] = (duration_s - overflow_s, forward_x, forward_z)
            total_duration_s = window_seconds


def orthographic_fov_for_viewport(
    *,
    projected_width: float,
    projected_height: float,
    viewport_aspect: float,
    fill: float,
) -> float:
    """Compute an orthographic camera size that keeps a projected area visible."""
    if projected_width <= 0 or projected_height <= 0:
        raise ValueError("projected track size must be positive")
    if viewport_aspect <= 0:
        raise ValueError("viewport_aspect must be positive")
    if not 0 < fill <= 1:
        raise ValueError("fill must be in the interval (0, 1]")
    return max(projected_height, projected_width / viewport_aspect) / fill


def _viewport_aspect_ratio(ursina: Any) -> float:
    window = getattr(ursina, "window", None)
    aspect_ratio = getattr(window, "aspect_ratio", None)
    if isinstance(aspect_ratio, (int, float)) and aspect_ratio > 0:
        return float(aspect_ratio)

    size = getattr(window, "size", None)
    if size is not None:
        try:
            width = float(size[0])
            height = float(size[1])
        except (IndexError, TypeError, ValueError):
            return DEFAULT_VIEWPORT_ASPECT_RATIO
        if width > 0 and height > 0:
            return width / height

    return DEFAULT_VIEWPORT_ASPECT_RATIO
