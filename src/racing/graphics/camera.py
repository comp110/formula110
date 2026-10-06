"""Camera modes and math for framing the car and track."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import atan2, cos, degrees, dist, exp, hypot, log, radians, sin, tan
from typing import Any, TypeAlias

from racing.game.config import CameraView
from racing.graphics.cinematic import CinematicCar, CinematicDirector, CinematicPose
from racing.graphics.leaders import LeadersCamera
from racing.graphics.track_rendering import (
    TRACK_EDGE_BUFFER,
    TRACK_SURFACE_Y,
    TRACK_WALL_BASE_Y,
    TRACK_WALL_HEIGHT,
    TRACK_WALL_THICKNESS,
)
from racing.race.progress import TrackProgressModel, project_track_position, track_pose_at_distance
from racing.race.start import GRID_CAMERA_TRANSITION_SECONDS
from racing.race.timing import TimingStanding
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
# A world-space offset lets the camera pan across corners without orbiting
# with the chassis. Translation deliberately trails the much quicker aim.
HELICOPTER_CAMERA_OFFSET = (24.0, 35.0, -32.0)
HELICOPTER_CAMERA_FOV = 48.0
HELICOPTER_CAMERA_POSITION_RESPONSE_SECONDS = 2.5
HELICOPTER_CAMERA_PAN_RESPONSE_SECONDS = 0.18
HELICOPTER_CAMERA_LOOK_HEIGHT = 0.5
HELICOPTER_CAMERA_MAX_LAG_M = 22.0
HELICOPTER_CAMERA_TELEPORT_DISTANCE_M = 60.0
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
PERSPECTIVE_NEAR_CLIP_MIN_M = 0.1
PERSPECTIVE_NEAR_CLIP_MAX_M = 10.0
PERSPECTIVE_NEAR_CLIP_HEIGHT_FRACTION = 0.05
FollowForwardSample: TypeAlias = tuple[float, float, float]
CAMERA_VIEW_SHORTCUTS: dict[str, CameraView] = {
    "a": CameraView.LEADERS,
    "q": CameraView.CINEMATIC,
    "w": CameraView.TOP_DOWN,
    "e": CameraView.THREE_QUARTER,
    "r": CameraView.HELICOPTER,
    "t": CameraView.DRONE,
    "y": CameraView.SPLIT_FOLLOW,
    "u": CameraView.FOLLOW,
}


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
    helicopter_position: tuple[float, float, float] | None = None
    helicopter_look_at: tuple[float, float, float] | None = None
    helicopter_target_position: tuple[float, float, float] | None = None
    cinematic: CinematicDirector = field(default_factory=CinematicDirector)
    leaders: LeadersCamera = field(default_factory=LeadersCamera)

    def select_follow_car(self, car_id: str | None) -> None:
        """Keep the current focused view; a second click on its car moves closer."""
        view = self.view
        if car_id is not None:
            if view in (CameraView.TOP_DOWN, CameraView.THREE_QUARTER, CameraView.CINEMATIC, CameraView.LEADERS):
                view = CameraView.HELICOPTER
            elif car_id == self.selected_car_id and view is not CameraView.SPLIT_FOLLOW:
                view = CameraView.FOLLOW
        if car_id == self.selected_car_id and view is self.view:
            return
        self.selected_car_id = car_id
        self.view = view
        self.reset_follow_history()

    def reset_follow_history(self) -> None:
        """Forget camera motion history after a view change or race reset."""
        self.follow_forward_samples.clear()
        self.follow_target_id = None
        self.follow_direction_initialized = False
        self.helicopter_position = None
        self.helicopter_look_at = None
        self.helicopter_target_position = None
        self.cinematic = CinematicDirector()
        self.leaders = LeadersCamera()


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
        return CameraView.HELICOPTER
    if view is CameraView.HELICOPTER:
        return CameraView.CINEMATIC
    if view is CameraView.CINEMATIC:
        return CameraView.LEADERS
    if view is CameraView.LEADERS:
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


def select_camera_view_from_key(rig: CameraRig, key: str, *, include_split: bool = False) -> None:
    """Jump to a view without changing the selected car or restarting the same view."""
    view = CAMERA_VIEW_SHORTCUTS.get(key)
    if view is None or view is rig.view or (view is CameraView.SPLIT_FOLLOW and not include_split):
        return
    rig.view = view
    rig.reset_follow_history()


def perspective_near_clip_for_height(camera_height: float) -> float:
    """Preserve depth precision for shallow track layers in distant views.

    The close-follow near plane wastes most of a perspective depth buffer when
    the camera rises above the track. Move it out with altitude, while keeping
    it well short of the ground and restoring close-up clearance near the car.
    """
    return max(
        PERSPECTIVE_NEAR_CLIP_MIN_M,
        min(PERSPECTIVE_NEAR_CLIP_MAX_M, (camera_height - TRACK_SURFACE_Y) * PERSPECTIVE_NEAR_CLIP_HEIGHT_FRACTION),
    )


def apply_camera_view(
    *,
    ursina: Any,
    view: CameraView,
    target: Any,
    rig: CameraRig | None = None,
    delta_seconds: float = 0.0,
    follow_settings: FollowCameraSettings = DEFAULT_FOLLOW_CAMERA_SETTINGS,
    track_model: TrackProgressModel | None = None,
    cinematic_cars: tuple[CinematicCar, ...] = (),
    cinematic_grid: bool = False,
    grid_intro_seconds: float | None = None,
    leader_standings: tuple[TimingStanding, ...] = (),
    leader_left_edge: float = -0.6,
    finishing: bool = False,
    finish_points: tuple[tuple[float, float, float], ...] = (),
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

    if view is CameraView.HELICOPTER:
        apply_helicopter_camera_view(ursina=ursina, target=target, rig=rig, delta_seconds=delta_seconds)
        return

    if view in (CameraView.CINEMATIC, CameraView.LEADERS):
        director = rig.cinematic if rig is not None else CinematicDirector()
        if not cinematic_cars:
            x, y, z = node_position(target)
            projection = None if track_model is None else project_track_position(track_model, TrackPoint(x, z))
            cinematic_cars = (CinematicCar(
                car_id="solo", rank=1, distance_m=0.0, position=(x, y, z),
                track_distance_m=0.0 if projection is None else projection.progress_distance_m,
                heading_degrees=float(target.getH()),
            ),)
        if view is CameraView.LEADERS:
            leaders = rig.leaders if rig is not None else LeadersCamera()
            pose = leaders.update(
                cinematic_cars, leader_standings, delta_seconds=delta_seconds, aspect_ratio=viewport_aspect,
                track_model=track_model, left_edge=leader_left_edge, finishing=finishing, finish_points=finish_points,
            )
        else:
            pose = director.update(
                cinematic_cars, delta_seconds=delta_seconds, aspect_ratio=viewport_aspect, track_model=track_model,
                grid=cinematic_grid,
            )
        if pose is not None:
            if grid_intro_seconds is not None and grid_intro_seconds < GRID_CAMERA_TRANSITION_SECONDS:
                pose = starting_grid_camera_pose(
                    pose, frame=camera_frame, aspect_ratio=viewport_aspect, elapsed_seconds=grid_intro_seconds,
                )
            ursina.camera.orthographic = False
            ursina.camera.perspective_lens.setNear(perspective_near_clip_for_height(pose.position[1]))
            ursina.camera.fov = pose.fov
            ursina.camera.position = pose.position
            ursina.camera.look_at(pose.look_at)
            ursina.camera.setR(0.0)
        else:
            apply_helicopter_camera_view(ursina=ursina, target=target, delta_seconds=delta_seconds)
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


def starting_grid_camera_pose(
    destination: CinematicPose, *, frame: TrackCameraFrame, aspect_ratio: float, elapsed_seconds: float,
) -> CinematicPose:
    """Ease from the whole track overhead to the director, with two lights left to light."""
    progress = max(0.0, min(1.0, elapsed_seconds / GRID_CAMERA_TRANSITION_SECONDS))
    if progress >= 1.0:
        return destination
    blend = progress * progress * (3.0 - 2.0 * progress)
    dx = destination.position[0] - destination.look_at[0]
    dy = destination.position[1] - destination.look_at[1]
    dz = destination.position[2] - destination.look_at[2]
    orbit = atan2(dx, dz)
    width = abs(cos(orbit)) * frame.width + abs(sin(orbit)) * frame.length
    height = abs(sin(orbit)) * frame.width + abs(cos(orbit)) * frame.length
    # The perspective lens uses horizontal FOV. Reserve the tower and lower lights panel.
    half_fov_tangent = tan(radians(destination.fov / 2.0))
    overview_distance = max(width, height * aspect_ratio) / (2.0 * half_fov_tangent * 0.64)
    shift = overview_distance * half_fov_tangent / aspect_ratio * 0.15
    overview_x = frame.center_x + sin(orbit) * shift
    overview_z = frame.center_z + cos(orbit) * shift
    target_distance = max(0.01, dist(destination.position, destination.look_at))
    distance = exp(log(overview_distance) * (1.0 - blend) + log(target_distance) * blend)
    elevation = radians(90.0) * (1.0 - blend) + atan2(dy, hypot(dx, dz)) * blend
    center = (
        overview_x + (destination.look_at[0] - overview_x) * blend,
        TRACK_SURFACE_Y + (destination.look_at[1] - TRACK_SURFACE_Y) * blend,
        overview_z + (destination.look_at[2] - overview_z) * blend,
    )
    # Keep the overhead heading stable after Panda converts positions to float32.
    horizontal = max(0.01, distance * cos(elevation))
    return CinematicPose(
        (center[0] + sin(orbit) * horizontal, center[1] + sin(elevation) * distance,
         center[2] + cos(orbit) * horizontal),
        center, destination.fov,
    )


def apply_helicopter_camera_view(
    *,
    ursina: Any,
    target: Any,
    rig: CameraRig | None = None,
    delta_seconds: float = 0.0,
) -> None:
    """Pan toward the car from a distant camera that slowly translates after it."""
    target_position = node_position(target)
    target_x, target_y, target_z = target_position
    offset_x, offset_y, offset_z = HELICOPTER_CAMERA_OFFSET
    desired_position = (target_x + offset_x, target_y + offset_y, target_z + offset_z)
    desired_look_at = (target_x, target_y + HELICOPTER_CAMERA_LOOK_HEIGHT, target_z)
    position = desired_position
    look_at = desired_look_at
    if rig is not None:
        previous_target = rig.helicopter_target_position
        target_changed = rig.follow_target_id != id(target)
        teleported = (
            previous_target is not None
            and sum(
                (current - previous) ** 2 for current, previous in zip(target_position, previous_target, strict=True)
            )
            > HELICOPTER_CAMERA_TELEPORT_DISTANCE_M**2
        )
        if target_changed or teleported:
            rig.reset_follow_history()
        rig.follow_target_id = id(target)
        rig.helicopter_target_position = target_position
        position = _smoothed_camera_point(
            current=rig.helicopter_position,
            desired=desired_position,
            delta_seconds=delta_seconds,
            response_seconds=HELICOPTER_CAMERA_POSITION_RESPONSE_SECONDS,
        )
        look_at = _smoothed_camera_point(
            current=rig.helicopter_look_at,
            desired=desired_look_at,
            delta_seconds=delta_seconds,
            response_seconds=HELICOPTER_CAMERA_PAN_RESPONSE_SECONDS,
        )
        # Bound the trailing distance so fast cars stay readable and cannot
        # pull the camera directly overhead when driving toward its offset.
        lag_x = position[0] - desired_position[0]
        lag_z = position[2] - desired_position[2]
        lag_length = (lag_x * lag_x + lag_z * lag_z) ** 0.5
        if delta_seconds > 0.0 and lag_length > HELICOPTER_CAMERA_MAX_LAG_M:
            scale = HELICOPTER_CAMERA_MAX_LAG_M / lag_length
            position = (desired_position[0] + lag_x * scale, position[1], desired_position[2] + lag_z * scale)
        rig.helicopter_position = position
        rig.helicopter_look_at = look_at

    ursina.camera.parent = ursina.scene
    ursina.camera.orthographic = False
    ursina.camera.perspective_lens.setNear(perspective_near_clip_for_height(position[1]))
    ursina.camera.fov = HELICOPTER_CAMERA_FOV
    ursina.camera.position = position
    ursina.camera.look_at(look_at)
    ursina.camera.setR(0.0)


def _smoothed_camera_point(
    *,
    current: tuple[float, float, float] | None,
    desired: tuple[float, float, float],
    delta_seconds: float,
    response_seconds: float,
) -> tuple[float, float, float]:
    """Ease a world-space point with a response independent of frame rate."""
    if current is None:
        return desired
    blend = 1.0 - exp(-max(delta_seconds, 0.0) / response_seconds)
    return (
        current[0] + (desired[0] - current[0]) * blend,
        current[1] + (desired[1] - current[1]) * blend,
        current[2] + (desired[2] - current[2]) * blend,
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
    lens.setNear(perspective_near_clip_for_height(target_y + follow_settings.height))
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
