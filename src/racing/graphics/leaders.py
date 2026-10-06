"""Drone framing for the leading battle and the stationary finish-line shot."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import cos, exp, radians, sin, tan
from typing import cast

from racing.graphics.cinematic import CinematicCar, CinematicPose, Point3
from racing.race.progress import TrackProgressModel, track_pose_at_distance
from racing.race.timing import TimingStanding

LEADERS_GAP_SECONDS = 3.0
FINISH_HOLD_SECONDS = 1.0
FINISH_CAMERA_TRANSITION_SECONDS = 3.0
LEADERS_FOV = 58.0
LEADERS_ELEVATION = 55.0


def leading_battle_ids(standings: tuple[TimingStanding, ...]) -> tuple[str, ...]:
    """Keep P1/P2 and include P3 only when the whole group spans at most three seconds."""
    leaders = sorted((row for row in standings if not row.eliminated), key=lambda row: row.rank)[:3]
    if len(leaders) == 3 and (leaders[2].gap_seconds is None or leaders[2].gap_seconds > LEADERS_GAP_SECONDS):
        leaders.pop()
    return tuple(row.car_id for row in leaders)


@dataclass(slots=True)
class FinishCameraSequence:
    """Use display time so the finishing shot can complete after physics stops."""

    second_finish_elapsed: float | None = None

    def advance(self, finish_count: int, delta_seconds: float) -> None:
        if self.second_finish_elapsed is None:
            if finish_count >= 2:
                self.second_finish_elapsed = 0.0
        else:
            self.second_finish_elapsed += max(0.0, delta_seconds)

    @property
    def pulling_back(self) -> bool:
        return self.second_finish_elapsed is not None and self.second_finish_elapsed >= FINISH_HOLD_SECONDS - 1e-9

    @property
    def complete(self) -> bool:
        return (self.second_finish_elapsed is not None
                and self.second_finish_elapsed >= FINISH_HOLD_SECONDS + FINISH_CAMERA_TRANSITION_SECONDS - 1e-9)


@dataclass(slots=True)
class LeadersCamera:
    """Follow a stable track heading and fit every selected car clear of the HUD."""

    subject_ids: tuple[str, ...] = ()
    pose: CinematicPose | None = None
    finish_pose: CinematicPose | None = None
    _center: Point3 | None = None
    _heading: float | None = None
    _distance: float = 18.0

    def update(
        self, cars: tuple[CinematicCar, ...], standings: tuple[TimingStanding, ...], *, delta_seconds: float,
        aspect_ratio: float, left_edge: float = -0.6, track_model: TrackProgressModel | None = None,
        finishing: bool = False, finish_points: tuple[Point3, ...] = (),
    ) -> CinematicPose | None:
        if finishing and self.finish_pose is not None:
            return self.finish_pose
        ids = leading_battle_ids(standings) if standings else tuple(car.car_id for car in cars[:2])
        by_id = {car.car_id: car for car in cars if not car.eliminated}
        subjects = tuple(by_id[car_id] for car_id in ids if car_id in by_id)
        self.subject_ids = tuple(car.car_id for car in subjects)
        if not subjects:
            return self.pose
        points = tuple(car.position for car in subjects) + (finish_points if finishing else ())
        desired_center = tuple((min(p[i] for p in points) + max(p[i] for p in points)) / 2 for i in range(3))
        dt = max(0.0, delta_seconds)
        blend = 1.0 - exp(-dt / 0.35)
        previous = self._center or desired_center
        self._center = cast(Point3, tuple(a + (b - a) * blend for a, b in zip(previous, desired_center, strict=True)))
        leader = subjects[0]
        heading = (track_pose_at_distance(track_model, leader.track_distance_m).heading_degrees
                   if track_model is not None else leader.heading_degrees)
        if self._heading is None:
            self._heading = heading
        else:
            change = (heading - self._heading + 180.0) % 360.0 - 180.0
            self._heading += max(-25 * dt, min(25 * dt, change * (1.0 - exp(-dt / 1.2))))
        h, elevation = radians(self._heading), radians(LEADERS_ELEVATION)
        right = (-cos(h), 0.0, sin(h))
        up = (sin(h) * sin(elevation), cos(elevation), cos(h) * sin(elevation))
        forward = (sin(h) * cos(elevation), -sin(elevation), cos(h) * cos(elevation))
        horizontal = tan(radians(LEADERS_FOV / 2))
        vertical = horizontal / max(0.1, aspect_ratio)
        # Reserve the tower on the left and the place cards above the cars.
        x_min, x_max, y_min, y_max = max(-0.88, min(0.4, left_edge)), 0.88, -0.80, 0.65
        cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        required = 18.0
        for point in points:
            for offset in product((-3.0, 3.0), repeat=3):
                relative = tuple(p + o - c for p, o, c in zip(point, offset, self._center, strict=True))
                across, above, depth = (sum(a * b for a, b in zip(relative, axis, strict=True))
                                        for axis in (right, up, forward))
                required = max(
                    required,
                    (across / horizontal - x_max * depth) / (x_max - cx),
                    (x_min * depth - across / horizontal) / (cx - x_min),
                    (above / vertical - y_max * depth) / (y_max - cy),
                    (y_min * depth - above / vertical) / (cy - y_min),
                )
        # Expand immediately to keep all subjects visible; return closer gently.
        self._distance = max(required, self._distance + (required - self._distance) * (1.0 - exp(-dt / 1.8)))
        aim = tuple(c - r * cx * self._distance * horizontal - u * cy * self._distance * vertical
                    for c, r, u in zip(self._center, right, up, strict=True))
        position = tuple(c - f * self._distance for c, f in zip(aim, forward, strict=True))
        self.pose = CinematicPose(cast(Point3, position), cast(Point3, aim), LEADERS_FOV)
        if finishing:
            self.finish_pose = self.pose
        return self.pose
