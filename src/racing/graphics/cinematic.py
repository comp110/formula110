"""A deterministic race director and smooth, engine-independent camera poses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from math import atan, atan2, cos, degrees, dist, exp, hypot, radians, sin, tan

from racing.graphics.grid_camera import choose_grid_camera_position, grid_banner_corners, grid_framing
from racing.graphics.track_rendering import (
    START_FINISH_BANNER_CENTER_Y,
    START_FINISH_BANNER_HEIGHT,
    TRACK_EDGE_BUFFER,
    TRACK_WALL_TOP_SURFACE_Y,
)
from racing.race.progress import TrackProgressModel, track_pose_at_distance
from racing.track.world import TRACK_WIDTH

Point3 = tuple[float, float, float]
MIN_SHOT_SECONDS = 8.0
ANGLE_SHOT_SECONDS = 14.0
STORY_CONFIRM_SECONDS = 1.5
PASS_HOLD_SECONDS = 4.0
MAX_BATTLE_GAP_M = 42.0
MAX_BATTLE_SEPARATION_M = 50.0
SUBJECT_TRANSITION_SECONDS = 5.5
MAX_ORBIT_SPEED_DEG_S = 6.0
MAX_ORBIT_ACCEL_DEG_S2 = 4.0
MAX_ELEVATION_SPEED_DEG_S = 5.0
MAX_ZOOM_OUT_MPS = 10.0
MAX_ZOOM_IN_MPS = 3.0

# Keep the lens and shot heights similar so reframing doesn't feel like a swoop.
SHOT_PROFILES = {
    "chase": (-20.0, 54.0, 40.0),
    "side": (-55.0, 56.0, 44.0),
    "aerial": (-32.0, 62.0, 50.0),
}


class CinematicAngle(Enum):
    GRID = "grid"
    CHASE = "chase"
    SIDE = "side"
    AERIAL = "aerial"


@dataclass(frozen=True, slots=True)
class CinematicCar:
    """One live timing row, with world position and track-relative motion."""

    car_id: str
    rank: int
    distance_m: float
    position: Point3
    speed_mps: float | None = None
    track_distance_m: float = 0.0
    heading_degrees: float = 0.0
    eliminated: bool = False
    recovery_count: int = 0


@dataclass(frozen=True, slots=True)
class CinematicPose:
    position: Point3
    look_at: Point3
    fov: float


@dataclass(frozen=True, slots=True)
class _Story:
    cars: tuple[CinematicCar, ...]
    score: float
    gap_m: float = 0.0

    @property
    def key(self) -> tuple[str, ...]:
        # A pass changes the running order, not the identity of the battle.
        return tuple(sorted(car.car_id for car in self.cars))


def _blend(dt: float, seconds: float) -> float:
    return 1.0 - exp(-max(0.0, dt) / seconds)


def _angle_delta(current: float, desired: float) -> float:
    return (desired - current + 180.0) % 360.0 - 180.0


def _damped_motion(
    value: float,
    velocity: float,
    target: float,
    dt: float,
    *,
    response: float,
    max_speed: float,
    max_acceleration: float,
) -> tuple[float, float]:
    """A critically damped spring with bounded speed and acceleration.

    Small simulation steps keep the limits and damping consistent through a
    dropped render frame, rather than allowing a large camera correction.
    """
    omega = 2.0 / response
    remaining = max(0.0, dt)
    while remaining > 1e-9:
        step = min(remaining, 1 / 120)
        acceleration = omega * omega * (target - value) - 2.0 * omega * velocity
        acceleration = max(-max_acceleration, min(max_acceleration, acceleration))
        next_velocity = max(-max_speed, min(max_speed, velocity + acceleration * step))
        value += (velocity + next_velocity) * 0.5 * step
        velocity = next_velocity
        remaining -= step
    return value, velocity


@dataclass(slots=True)
class CinematicDirector:
    """Prefer podium battles, hold shots, and damp translation, orbit, and zoom."""

    subject_ids: tuple[str, ...] = ()
    angle: CinematicAngle = CinematicAngle.CHASE
    shot_seconds: float = 0.0
    pose: CinematicPose | None = None
    _previous: dict[str, CinematicCar] = field(default_factory=dict[str, CinematicCar])
    _speeds: dict[str, float] = field(default_factory=dict[str, float])
    _passes: dict[tuple[str, ...], float] = field(default_factory=dict[tuple[str, ...], float])
    _center: Point3 | None = None
    _orbit: float | None = None
    _shot_orbit: float = 0.0
    _orbit_velocity: float = 0.0
    _elevation: float = 54.0
    _elevation_velocity: float = 0.0
    _clearance_elevation: float = 0.0
    _distance: float = 40.0
    _distance_velocity: float = 0.0
    _framing_distance: float = 40.0
    _zoom_in_seconds: float = 0.0
    _fov: float = 48.0
    _transition_from: Point3 | None = None
    _transition_seconds: float = 0.0
    _transition_subjects: tuple[str, ...] = ()
    _transition_offset: Point3 = (0.0, 0.0, 0.0)
    _pending_subjects: tuple[str, ...] = ()
    _pending_seconds: float = 0.0
    _pending_angle: CinematicAngle | None = None
    _pending_angle_seconds: float = 0.0
    _grid_position: Point3 | None = None
    _grid_banner: tuple[Point3, ...] = ()

    def update(
        self,
        cars: tuple[CinematicCar, ...],
        *,
        delta_seconds: float,
        aspect_ratio: float = 16 / 9,
        track_model: TrackProgressModel | None = None,
        grid: bool = False,
    ) -> CinematicPose | None:
        """Advance using rendered time; timing distances must be lap-aware."""
        dt = max(0.0, delta_seconds)
        if dt == 0.0 and self.pose is not None:
            return self.pose
        active = tuple(sorted((car for car in cars if not car.eliminated), key=lambda car: car.rank))
        if not active:
            # Retain the last composition when the race has no survivors.
            return self.pose
        discontinuities = self._sample_motion(active, dt)
        stories = (_Story(active, 0.0),) if grid else self._stories(active)
        best = max(stories, key=lambda story: story.score)
        current = next((story for story in stories if story.key == self.subject_ids), None)
        self.shot_seconds += dt
        urgent = current is not None and best.key != current.key and best.score > current.score * 1.3 + 0.65
        if urgent:
            if self._pending_subjects != best.key:
                self._pending_subjects = best.key
                self._pending_seconds = 0.0
            self._pending_seconds += dt
        else:
            self._pending_subjects = ()
            self._pending_seconds = 0.0
        if current is None or (
            self.shot_seconds >= MIN_SHOT_SECONDS and self._pending_seconds >= STORY_CONFIRM_SECONDS
        ):
            current = best
        changed = self.subject_ids != current.key
        if (changed or (self.angle is CinematicAngle.GRID and not grid)) and self._center is not None:
            self._transition_from = self._center
            self._transition_seconds = 0.0
            self._transition_subjects = self.subject_ids
            self._transition_offset = (0.0, 0.0, 0.0)
            if self.angle is CinematicAngle.GRID:
                previous_subjects = tuple(car for car in active if car.car_id in self.subject_ids)
                if previous_subjects:
                    self._transition_offset = (
                        self._center[0] - sum(car.position[0] for car in previous_subjects) / len(previous_subjects),
                        self._center[1]
                        - sum(car.position[1] for car in previous_subjects) / len(previous_subjects) - 0.6,
                        self._center[2] - sum(car.position[2] for car in previous_subjects) / len(previous_subjects),
                    )
        self.subject_ids = current.key
        heading, bend = self._track_direction(current.cars[0], track_model)
        desired_angle = CinematicAngle.GRID if grid else self._choose_angle(current, bend)
        if desired_angle is not self._pending_angle:
            self._pending_angle = desired_angle
            self._pending_angle_seconds = 0.0
        self._pending_angle_seconds += dt
        if changed or grid != (self.angle is CinematicAngle.GRID) or (
            desired_angle is not self.angle
            and self.shot_seconds >= ANGLE_SHOT_SECONDS
            and self._pending_angle_seconds >= STORY_CONFIRM_SECONDS
        ):
            self.angle = desired_angle
            # Hold a world-space angle for the shot. Bends move the cars through
            # the composition instead of whipping the camera around the track.
            if not grid:
                self._shot_orbit = heading + SHOT_PROFILES[self.angle.value][0]
            self.shot_seconds = 0.0
        if any(car_id in discontinuities for car_id in self.subject_ids):
            if any(
                car.car_id in self._previous and dist(car.position, self._previous[car.car_id].position) > 60.0
                for car in current.cars
            ):
                # A distant teleport is a new location, never a long fly-through.
                self._center = None
                self._orbit = None
                self._transition_from = None
            else:
                self._transition_from = self._center
                self._transition_seconds = 0.0
                self._transition_subjects = ()
        self.pose = self._compose(current, active, dt, aspect_ratio, track_model, grid=grid)
        self._previous = {car.car_id: car for car in active}
        return self.pose

    def _sample_motion(self, cars: tuple[CinematicCar, ...], dt: float) -> set[str]:
        discontinuities: set[str] = set()
        self._passes = {key: remaining - dt for key, remaining in self._passes.items() if remaining > dt}
        previous_speeds = self._speeds
        self._speeds = {}
        for car in cars:
            previous = self._previous.get(car.car_id)
            speed = car.speed_mps if car.speed_mps is not None else previous_speeds.get(car.car_id, 0.0)
            if previous is not None and (
                previous.recovery_count != car.recovery_count
                or dist(previous.position, car.position) > max(60.0, abs(speed) * dt * 3.0)
            ):
                discontinuities.add(car.car_id)
            elif car.speed_mps is None and previous is not None and dt > 0.0:
                # A solo viewer has no race timing row; infer motion for its framing.
                speed = dist(previous.position, car.position) / dt
            old_speed = previous_speeds.get(car.car_id, speed) if car.car_id not in discontinuities else speed
            self._speeds[car.car_id] = old_speed + (speed - old_speed) * _blend(dt, 0.6)
        for front, rear in pairwise(cars[:4]):
            old_front = self._previous.get(front.car_id)
            old_rear = self._previous.get(rear.car_id)
            key = tuple(sorted((front.car_id, rear.car_id)))
            if (
                old_front is not None
                and old_rear is not None
                and old_front.rank > old_rear.rank
                and not discontinuities.intersection(key)
                and abs(front.distance_m - rear.distance_m) < MAX_BATTLE_GAP_M
                and dist(front.position, rear.position) < MAX_BATTLE_SEPARATION_M
            ):
                self._passes[key] = PASS_HOLD_SECONDS
        self._passes = {key: seconds for key, seconds in self._passes.items() if not discontinuities.intersection(key)}
        return discontinuities

    def _stories(self, cars: tuple[CinematicCar, ...]) -> tuple[_Story, ...]:
        stories = [_Story((cars[0],), 0.8)]
        # P4 matters only when challenging P3; battles further back cannot take over.
        for front, rear in pairwise(cars[:4]):
            story = self._battle_story(front, rear)
            if story is not None:
                stories.append(story)
        if len(self.subject_ids) in (1, 2) and not any(story.key == self.subject_ids for story in stories):
            subjects = tuple(car for car in cars if car.car_id in self.subject_ids)
            if len(subjects) == len(self.subject_ids) and subjects[0].rank <= 3:
                # Keep watching a battle when a third car slips between its subjects.
                # A looser exit threshold also prevents chatter around the gap limit.
                retained = (
                    self._battle_story(*subjects, retaining=True) if len(subjects) == 2 else _Story(subjects, 0.1)
                )
                if retained is not None:
                    stories.append(retained)
        return tuple(stories)

    def _battle_story(self, front: CinematicCar, rear: CinematicCar, *, retaining: bool = False) -> _Story | None:
        gap = abs(front.distance_m - rear.distance_m)
        margin = 12.0 if retaining else 0.0
        if gap > MAX_BATTLE_GAP_M + margin or dist(front.position, rear.position) > MAX_BATTLE_SEPARATION_M + margin:
            return None
        closing = max(0.0, self._speeds[rear.car_id] - self._speeds[front.car_id])
        catch_seconds = gap / closing if closing > 0.5 else 100.0
        proximity = 2.8 * max(0.0, 1.0 - gap / MAX_BATTLE_GAP_M)
        anticipation = 3.0 * max(0.0, 1.0 - catch_seconds / 6.0)
        key = tuple(sorted((front.car_id, rear.car_id)))
        pass_bonus = 2.4 * self._passes.get(key, 0.0) / PASS_HOLD_SECONDS
        score = proximity + anticipation + pass_bonus + (4 - front.rank) * 0.22
        if max(self._speeds[front.car_id], self._speeds[rear.car_id]) < 1.0:
            score *= 0.2
        return _Story((front, rear), score, gap)

    def _track_direction(self, car: CinematicCar, model: TrackProgressModel | None) -> tuple[float, float]:
        if model is None:
            return car.heading_degrees, 0.0
        near = track_pose_at_distance(model, car.track_distance_m)
        far = track_pose_at_distance(model, car.track_distance_m + 16.0)
        bend = _angle_delta(near.heading_degrees, far.heading_degrees)
        return near.heading_degrees + bend * 0.35, bend

    def _choose_angle(self, story: _Story, bend: float) -> CinematicAngle:
        if abs(bend) > 24.0:
            return CinematicAngle.AERIAL
        if len(story.cars) > 1:
            if story.gap_m < 6.0:
                return CinematicAngle.SIDE
            return CinematicAngle.CHASE
        return CinematicAngle.CHASE

    def _compose(
        self,
        story: _Story,
        cars: tuple[CinematicCar, ...],
        dt: float,
        aspect: float,
        model: TrackProgressModel | None,
        *,
        grid: bool,
    ) -> CinematicPose:
        count = len(story.cars)
        center = (
            sum(car.position[0] for car in story.cars) / count,
            sum(car.position[1] for car in story.cars) / count + 0.6,
            sum(car.position[2] for car in story.cars) / count,
        )
        if self._transition_from is not None:
            self._transition_seconds += dt
            t = min(1.0, self._transition_seconds / SUBJECT_TRANSITION_SECONDS)
            # Quintic easing brings both ends of a subject handoff to rest.
            blend = t * t * t * (t * (6.0 * t - 15.0) + 10.0)
            start = self._transition_from
            previous_subjects = tuple(car for car in cars if car.car_id in self._transition_subjects)
            if previous_subjects:
                # Track both moving groups during the handoff. A frozen starting
                # point makes the camera lag behind and then rush to catch up.
                start = (
                    sum(car.position[0] for car in previous_subjects) / len(previous_subjects)
                    + self._transition_offset[0],
                    sum(car.position[1] for car in previous_subjects) / len(previous_subjects)
                    + 0.6 + self._transition_offset[1],
                    sum(car.position[2] for car in previous_subjects) / len(previous_subjects)
                    + self._transition_offset[2],
                )
            center = (
                start[0] + (center[0] - start[0]) * blend,
                start[1] + (center[1] - start[1]) * blend,
                start[2] + (center[2] - start[2]) * blend,
            )
            if t == 1.0:
                self._transition_from = None
        initial = self._center is None
        if self._center is None:
            self._center = center
        else:
            blend = _blend(dt, 0.3)
            self._center = (
                self._center[0] + (center[0] - self._center[0]) * blend,
                self._center[1] + (center[1] - self._center[1]) * blend,
                self._center[2] + (center[2] - self._center[2]) * blend,
            )
        if grid:
            return self._compose_grid(story, aspect, model, dt)
        # Ease even a wide gantry lens into race coverage without a zoom jump.
        fov_step = (48.0 - self._fov) * _blend(dt, 3.0)
        self._fov += max(-10.0 * dt, min(10.0 * dt, fov_step))
        if self._orbit is None:
            self._orbit = self._shot_orbit
            self._orbit_velocity = 0.0
        else:
            self._orbit, self._orbit_velocity = _damped_motion(
                self._orbit,
                self._orbit_velocity,
                self._orbit + _angle_delta(self._orbit, self._shot_orbit),
                dt,
                response=2.5,
                max_speed=MAX_ORBIT_SPEED_DEG_S,
                max_acceleration=MAX_ORBIT_ACCEL_DEG_S2,
            )
        _, elevation, distance = SHOT_PROFILES[self.angle.value]
        # Fit a bounding sphere around both cars, using the narrower lens axis.
        # Motion padding covers the smoothed center's lag; the extra margin leaves
        # room for car bodies, their labels, and the timing tower.
        radius = max(dist(car.position, story.cars[0].position) for car in story.cars) / 2.0 + 3.0
        radius += max(abs(self._speeds[car.car_id]) for car in story.cars) * 0.3
        half_fov = atan(tan(radians(self._fov / 2.0)) / max(1.0, aspect))
        distance = max(distance, radius / sin(half_fov * 0.78))
        self._clearance_elevation = max(
            self._clearance_elevation - dt * 0.75,
            self._barrier_elevation(story, self._orbit, model),
        )
        elevation = max(elevation, self._clearance_elevation)
        settled_battle = len(story.cars) == 2 and self._transition_from is None
        if settled_battle:
            # Frame the cars in the actual view, including tracking lag. The old
            # speed-padded sphere kept even side-by-side fast cars far away.
            closeness = max(0.0, 1.0 - story.gap_m / 24.0)
            minimum_distance = SHOT_PROFILES[self.angle.value][2] * (1.0 - 0.35 * closeness)
            distance = max(
                minimum_distance,
                self._subject_framing_distance(story, self._orbit, elevation if initial else self._elevation, aspect),
            )
        closing_battle = (
            settled_battle
            and story.gap_m < 24.0
            and (self._speeds[story.cars[1].car_id] - self._speeds[story.cars[0].car_id] > 0.5 or story.gap_m < 6.0)
        )
        if initial:
            self._elevation = elevation
            self._elevation_velocity = 0.0
            self._distance = self._framing_distance = distance
            self._distance_velocity = 0.0
        else:
            # Closing battles can tighten immediately at the slow dolly limit.
            # Other changes must persist, so speed and gap noise do not pump zoom.
            self._zoom_in_seconds = self._zoom_in_seconds + dt if distance < self._framing_distance * 0.8 else 0.0
            if closing_battle or distance > self._framing_distance or self._zoom_in_seconds >= 4.0:
                self._framing_distance = distance
                self._zoom_in_seconds = 0.0
            self._elevation, self._elevation_velocity = _damped_motion(
                self._elevation,
                self._elevation_velocity,
                elevation,
                dt,
                response=2.0,
                max_speed=MAX_ELEVATION_SPEED_DEG_S,
                max_acceleration=4.0,
            )
            self._distance, self._distance_velocity = _damped_motion(
                self._distance,
                self._distance_velocity,
                self._framing_distance,
                dt,
                response=2.0,
                max_speed=MAX_ZOOM_OUT_MPS
                if self._framing_distance > self._distance or self._distance_velocity > 0.0
                else MAX_ZOOM_IN_MPS,
                max_acceleration=6.0,
            )
        orbit = radians(self._orbit)
        horizontal = self._distance * cos(radians(self._elevation))
        position = (
            self._center[0] - sin(orbit) * horizontal,
            self._center[1] + self._distance * sin(radians(self._elevation)),
            self._center[2] - cos(orbit) * horizontal,
        )
        return CinematicPose(position, self._center, self._fov)

    def _compose_grid(
        self, story: _Story, aspect: float, model: TrackProgressModel | None, dt: float,
    ) -> CinematicPose:
        """Watch the full field from a low viewpoint with clear scenery sightlines."""
        assert self._center is not None
        initial = self._grid_position is None
        if initial:
            front = story.cars[0]
            ahead = 30.0
            heading = front.heading_degrees
            if model is not None:
                # Timing order is arbitrary before lights-out, and grids can wrap
                # across lap zero. Locate the physical front of the starting field.
                reference = front.track_distance_m
                front = max(story.cars, key=lambda car: (
                    car.track_distance_m - reference + model.total_length_m / 2
                ) % model.total_length_m)
                # The gantry stays fixed when cars launch during LIGHTS OUT.
                self._grid_banner = grid_banner_corners(model=model, front_distance=front.track_distance_m)
                heading = track_pose_at_distance(model, front.track_distance_m).heading_degrees
                ahead = min(100.0, model.total_length_m / 4)
                for offset in range(2, int(ahead) + 1, 2):
                    corner = track_pose_at_distance(model, front.track_distance_m + offset)
                    if abs(_angle_delta(heading, corner.heading_degrees)) >= 24.0:
                        ahead = max(18.0, offset - 6.0)
                        break
            # Stay on the straight's sightline, just before the bend. A small
            # lateral offset separates the staggered rows without hiding the cars
            # behind the near wall at this low height.
            forward_x, forward_z = sin(radians(heading)), cos(radians(heading))
            self._grid_position = (
                front.position[0] + forward_x * ahead + forward_z * 1.5,
                # Keep the entire sightline below the Formula110 gantry banner.
                START_FINISH_BANNER_CENTER_Y - START_FINISH_BANNER_HEIGHT / 2 - 0.35,
                front.position[2] + forward_z * ahead - forward_x * 1.5,
            )
            if model is not None:
                self._grid_position = choose_grid_camera_position(
                    story.cars, model=model, center=self._center, preferred=self._grid_position, aspect=aspect,
                )
        assert self._grid_position is not None
        self._center, fov = grid_framing(
            story.cars, model=model, position=self._grid_position, center=self._center, aspect=aspect,
            banner=self._grid_banner,
        )
        x, y, z = (a - b for a, b in zip(self._grid_position, self._center, strict=True))
        self._orbit = degrees(atan2(-x, -z))
        self._elevation = degrees(atan2(y, hypot(x, z)))
        self._distance = self._framing_distance = dist(self._grid_position, self._center)
        self._orbit_velocity = self._elevation_velocity = self._distance_velocity = 0.0
        self._fov = fov if initial else self._fov + (fov - self._fov) * _blend(dt, 0.5)
        return CinematicPose(self._grid_position, self._center, self._fov)

    def _subject_framing_distance(
        self,
        story: _Story,
        orbit: float,
        elevation: float,
        aspect: float,
        *,
        horizontal_fill: float = 0.78,
        vertical_fill: float = 0.78,
    ) -> float:
        """Fit car bodies within the lens, with space for tracking and HUD."""
        assert self._center is not None
        heading, pitch = radians(orbit), radians(elevation)
        sin_h, cos_h = sin(heading), cos(heading)
        sin_p, cos_p = sin(pitch), cos(pitch)
        horizontal_tan = tan(radians(self._fov / 2.0)) * horizontal_fill
        vertical_tan = tan(radians(self._fov / 2.0)) * vertical_fill / max(0.1, aspect)
        distance = 0.0
        for car in story.cars:
            x, y, z = (a - b for a, b in zip(car.position, self._center, strict=True))
            across = x * cos_h - z * sin_h
            up = (x * sin_h + z * cos_h) * sin_p + y * cos_p
            depth = (x * sin_h + z * cos_h) * cos_p - y * sin_p
            distance = max(
                distance,
                (abs(across) + 3.0) / horizontal_tan - depth,
                (abs(up) + 3.0) / vertical_tan - depth,
            )
        return distance

    def _barrier_elevation(self, story: _Story, orbit: float, model: TrackProgressModel | None) -> float:
        """Clear the near barrier even when a subject hugs the edge of the road."""
        if model is None:
            return 0.0
        elevation = 0.0
        for car in story.cars:
            pose = track_pose_at_distance(model, car.track_distance_m)
            heading = radians(pose.heading_degrees)
            lateral = (car.position[0] - pose.position.x) * -cos(heading)
            lateral += (car.position[2] - pose.position.z) * sin(heading)
            toward_left = sin(radians(orbit - pose.heading_degrees))
            clearance = TRACK_WIDTH / 2.0 + TRACK_EDGE_BUFFER - (lateral if toward_left > 0 else -lateral)
            ray_distance = max(0.25, clearance) / max(0.01, abs(toward_left))
            rise = max(0.0, TRACK_WALL_TOP_SURFACE_Y - car.position[1] + 0.65)
            elevation = max(elevation, degrees(atan2(rise, ray_distance)))
        return min(72.0, elevation)
