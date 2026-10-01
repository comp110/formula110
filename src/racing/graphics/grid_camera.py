"""Choose a starting-grid viewpoint with clear sightlines through track scenery."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, atan2, cos, degrees, dist, hypot, radians, sin, tan
from typing import TYPE_CHECKING

from racing.graphics.timing_tower import TOWER_WIDTH
from racing.graphics.track_mesh import clean_offset_path
from racing.graphics.track_rendering import (
    START_FINISH_BANNER_CENTER_Y,
    START_FINISH_BANNER_HEIGHT,
    START_FINISH_BANNER_OVERHANG,
    START_FINISH_BANNER_POLE_THICKNESS,
    START_FINISH_BANNER_THICKNESS,
    TRACK_EDGE_BUFFER,
    TRACK_WALL_THICKNESS,
    TRACK_WALL_TOP_SURFACE_Y,
    start_finish_banner_side_distances,
    start_finish_render_pose,
)
from racing.race.progress import TrackProgressModel, track_pose_at_distance
from racing.race.runtime import start_finish_pose_for_progress
from racing.track.world import TRACK_WIDTH

if TYPE_CHECKING:
    from racing.graphics.cinematic import CinematicCar

Point3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class _Screen:
    """A vertical scenery face, used only while choosing the opening camera."""

    start: Point3
    end: Point3
    bottom: float
    top: float


def _scenery(model: TrackProgressModel, front_distance: float) -> tuple[_Screen, ...]:
    screens: list[_Screen] = []
    for side in (-1, 1):
        for thickness in (0.0, TRACK_WALL_THICKNESS):
            path = clean_offset_path(model.points, side * (TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + thickness), 0)
            screens.extend(
                _Screen(point, path[(index + 1) % len(path)], 0.0, TRACK_WALL_TOP_SURFACE_Y)
                for index, point in enumerate(path)
            )
    return (*screens, *_banner_screens(model, front_distance))


def _banner_screens(model: TrackProgressModel, front_distance: float) -> tuple[_Screen, ...]:
    screens: list[_Screen] = []
    start = start_finish_pose_for_progress(model=model, start_progress_distance_m=front_distance)
    banner = start_finish_render_pose(samples=model.points, position=start.position)
    negative, positive = start_finish_banner_side_distances(samples=model.points, position=banner.position)
    heading = radians(banner.heading_degrees)
    forward_x, forward_z = sin(heading), cos(heading)
    left_x, left_z = -forward_z, forward_x

    def point(across: float, forward: float) -> Point3:
        return (
            banner.position.x + left_x * across + forward_x * forward,
            0.0,
            banner.position.z + left_z * across + forward_z * forward,
        )

    bottom = START_FINISH_BANNER_CENTER_Y - START_FINISH_BANNER_HEIGHT / 2
    top = START_FINISH_BANNER_CENTER_Y + START_FINISH_BANNER_HEIGHT / 2
    for face in (-1, 1):
        depth = face * START_FINISH_BANNER_THICKNESS / 2
        screens.append(_Screen(
            point(-negative - START_FINISH_BANNER_OVERHANG / 2, depth),
            point(positive + START_FINISH_BANNER_OVERHANG / 2, depth), bottom, top,
        ))
    for across in (-negative, positive):
        half = START_FINISH_BANNER_POLE_THICKNESS / 2
        corners = tuple(point(across + x * half, z * half) for x, z in ((-1, -1), (1, -1), (1, 1), (-1, 1)))
        screens.extend(_Screen(corner, corners[(i + 1) % 4], 0.0, top) for i, corner in enumerate(corners))
    return tuple(screens)


def grid_banner_corners(*, model: TrackProgressModel, front_distance: float) -> tuple[Point3, ...]:
    """Use the rendered gantry's full width, height, and depth for framing."""
    return tuple(
        (point[0], height, point[2])
        for face in _banner_screens(model, front_distance)[:2]
        for point in (face.start, face.end)
        for height in (face.bottom, face.top)
    )


def _clear_height(position: Point3, targets: tuple[Point3, ...], screens: tuple[_Screen, ...]) -> float:
    """Find the lowest height whose rays pass above or below every scenery face."""
    height = position[1]
    forbidden: list[tuple[float, float]] = []
    for target in targets:
        ray_x, ray_z = position[0] - target[0], position[2] - target[2]
        for screen in screens:
            edge_x, edge_z = screen.end[0] - screen.start[0], screen.end[2] - screen.start[2]
            denominator = ray_x * edge_z - ray_z * edge_x
            if abs(denominator) < 1e-8:
                continue
            offset_x, offset_z = screen.start[0] - target[0], screen.start[2] - target[2]
            along_ray = (offset_x * edge_z - offset_z * edge_x) / denominator
            along_edge = (offset_x * ray_z - offset_z * ray_x) / denominator
            if not (0.0 < along_ray < 1.0 and 0.0 <= along_edge <= 1.0):
                continue
            upper = target[1] + (screen.top + 0.2 - target[1]) / along_ray
            if screen.bottom == 0.0:
                height = max(height, upper)
            else:
                lower = target[1] + (screen.bottom - 0.2 - target[1]) / along_ray
                forbidden.append((lower, upper))
    for lower, upper in sorted(forbidden):
        if lower <= height <= upper:
            height = upper + 0.01
    return height


def _lens_tangent(position: Point3, center: Point3, targets: tuple[Point3, ...], aspect: float) -> float:
    x, y, z = (a - b for a, b in zip(position, center, strict=True))
    distance = dist(position, center)
    orbit, pitch = atan2(-x, -z), atan2(y, hypot(x, z))
    lens = tan(radians(6.0))
    for target in targets:
        x, y, z = (a - b for a, b in zip(target, center, strict=True))
        across = x * cos(orbit) - z * sin(orbit)
        up = (x * sin(orbit) + z * cos(orbit)) * sin(pitch) + y * cos(pitch)
        depth = distance + (x * sin(orbit) + z * cos(orbit)) * cos(pitch) - y * sin(pitch)
        if depth <= 1.0:
            return float("inf")
        lens = max(lens, abs(across) / (depth * 0.62), abs(up) * aspect / (depth * 0.50))
    return lens


def _car_footprints(cars: tuple[CinematicCar, ...], model: TrackProgressModel | None) -> tuple[Point3, ...]:
    targets: list[Point3] = []
    for car in cars:
        heading = radians(
            car.heading_degrees
            if model is None else track_pose_at_distance(model, car.track_distance_m).heading_degrees
        )
        for forward, side in ((-1.3, -1.0), (-1.3, 1.0), (1.3, -1.0), (1.3, 1.0)):
            targets.append((
                car.position[0] + sin(heading) * forward + cos(heading) * side,
                car.position[1],
                car.position[2] + cos(heading) * forward - sin(heading) * side,
            ))
    return tuple(targets)


def grid_framing(
    cars: tuple[CinematicCar, ...], *, model: TrackProgressModel | None,
    position: Point3, center: Point3, aspect: float,
    banner: tuple[Point3, ...] = (),
) -> tuple[Point3, float]:
    """Fit the field and place the complete gantry along the shot's top edge."""
    targets = tuple((x, y + height, z) for x, y, z in _car_footprints(cars, model) for height in (0.0, 1.0))
    if banner:
        return _banner_framing(position, center, targets, banner, aspect)
    for _ in range(2):
        x, y, z = (a - b for a, b in zip(position, center, strict=True))
        orbit, pitch = atan2(-x, -z), atan2(y, hypot(x, z))
        forward = (sin(orbit) * cos(pitch), -sin(pitch), cos(orbit) * cos(pitch))
        right = (cos(orbit), 0.0, -sin(orbit))
        up = (sin(orbit) * sin(pitch), cos(pitch), cos(orbit) * sin(pitch))
        across_values: list[float] = []
        up_values: list[float] = []
        for target in targets:
            ray = tuple(a - b for a, b in zip(target, position, strict=True))
            depth = max(1.0, sum(a * b for a, b in zip(ray, forward, strict=True)))
            across_values.append(sum(a * b for a, b in zip(ray, right, strict=True)) / depth)
            up_values.append(sum(a * b for a, b in zip(ray, up, strict=True)) / depth)
        across_mid = (min(across_values) + max(across_values)) / 2
        up_mid = (min(up_values) + max(up_values)) / 2
        aim = tuple(f + r * across_mid + u * up_mid for f, r, u in zip(forward, right, up, strict=True))
        distance = (center[1] - position[1]) / min(-0.001, aim[1])
        center = (position[0] + aim[0] * distance, center[1], position[2] + aim[2] * distance)
    return center, degrees(2 * atan(_lens_tangent(position, center, targets, aspect)))


def _banner_framing(
    position: Point3, center: Point3, cars: tuple[Point3, ...], banner: tuple[Point3, ...], aspect: float,
) -> tuple[Point3, float]:
    # Center the horizontal angular bounds of both subjects. The side margins
    # leave the full logo clear of the timing tower as well as the frame edges.
    horizontal_fill = min(0.62, max(0.1, 1.0 - (TOWER_WIDTH + 0.055 + 0.06) / aspect))
    rays = tuple(tuple(a - b for a, b in zip(target, position, strict=True)) for target in (*cars, *banner))
    orbit = atan2(center[0] - position[0], center[2] - position[2])
    angles = tuple(atan2(x * cos(orbit) - z * sin(orbit), x * sin(orbit) + z * cos(orbit)) for x, _, z in rays)
    orbit += (min(angles) + max(angles)) / 2
    coordinates = tuple((x * cos(orbit) - z * sin(orbit), y, x * sin(orbit) + z * cos(orbit)) for x, y, z in rays)
    banner_top = max(atan2(y, along) for _, y, along in coordinates[-len(banner):])

    def elevation(lens: float) -> float:
        # A small inset protects the upper edge from rounding and antialiasing.
        return banner_top - atan(0.97 * lens / aspect)

    def fits(lens: float) -> bool:
        angle = elevation(lens)
        for across, y, along in coordinates:
            depth = along * cos(angle) + y * sin(angle)
            up = y * cos(angle) - along * sin(angle)
            if depth <= 0 or abs(across) > depth * lens * horizontal_fill or up < -depth * lens / aspect * 0.50:
                return False
        return True

    lower = tan(radians(6.0))
    upper = lower
    while not fits(upper) and upper < tan(radians(80.0)):
        upper *= 1.2
    for _ in range(24):
        middle = (lower + upper) / 2
        if fits(middle):
            upper = middle
        else:
            lower = middle
    angle = elevation(upper)
    distance = dist(position, center)
    aim = (
        position[0] + sin(orbit) * cos(angle) * distance,
        position[1] + sin(angle) * distance,
        position[2] + cos(orbit) * cos(angle) * distance,
    )
    # From some high viewpoints, distant cars project above the gantry. A wider
    # lens cannot keep that banner at the top; choose a different viewpoint.
    for _, y, along in coordinates[:len(cars)]:
        depth = along * cos(angle) + y * sin(angle)
        up = y * cos(angle) - along * sin(angle)
        if up > depth * upper / aspect * 0.90:
            return aim, float("inf")
    return aim, degrees(2 * atan(upper))


def choose_grid_camera_position(
    cars: tuple[CinematicCar, ...], *, model: TrackProgressModel, center: Point3, preferred: Point3, aspect: float,
) -> Point3:
    """Prefer the corner approach, using another low angle when the grid bends."""
    reference = cars[0].track_distance_m
    ordered = sorted(cars, key=lambda car: (
        car.track_distance_m - reference + model.total_length_m / 2
    ) % model.total_length_m)
    screens = _scenery(model, ordered[-1].track_distance_m)
    banner = grid_banner_corners(model=model, front_distance=ordered[-1].track_distance_m)
    # Check the body footprint, not just a floating label or the car center.
    footprint = tuple((x, y + height, z) for x, y, z in _car_footprints(cars, model) for height in (0.0, 1.0))
    preferred_height = _clear_height(preferred, footprint, screens)
    if preferred_height <= preferred[1]:
        _, fov = grid_framing(cars, model=model, position=preferred, center=center, aspect=aspect, banner=banner)
        if fov <= 56.0:
            return preferred
    front, rear = ordered[-1].position, ordered[0].position
    heading = atan2(front[0] - rear[0], front[2] - rear[2])
    radius = max(28.0, max(hypot(car.position[0] - center[0], car.position[2] - center[2]) for car in cars) + 16.0)
    best = (preferred[0], preferred_height, preferred[2])
    best_score = float("inf")
    candidates = [(0, preferred)]
    for angle in range(-180, 180, 15):
        orbit = heading + radians(angle)
        candidates.append((angle, (center[0] + sin(orbit) * radius, preferred[1], center[2] + cos(orbit) * radius)))
    for angle, position in candidates:
        height = _clear_height(position, footprint, screens)
        position = (position[0], height, position[2])
        _, fov = grid_framing(cars, model=model, position=position, center=center, aspect=aspect, banner=banner)
        # Favor low cameras and a moderate lens; a small preference for seeing
        # the front of the field must never outweigh an obstructed sightline.
        score = height + max(0.0, fov - 50.0) * 0.15 + abs(angle) * 0.008
        if score < best_score:
            best, best_score = position, score
    return best
