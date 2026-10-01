"""Check real Bullet sightlines, independently of the camera's visibility math."""

from importlib import import_module
from math import cos, radians, sin
from typing import Any, cast

import pytest

from racing.graphics.cinematic import CinematicCar, CinematicDirector
from racing.graphics.track_rendering import (
    START_FINISH_BANNER_CENTER_Y,
    START_FINISH_BANNER_HEIGHT,
    START_FINISH_BANNER_OVERHANG,
    START_FINISH_BANNER_THICKNESS,
    add_racing_scene_collisions,
    start_finish_banner_side_distances,
    start_finish_render_pose,
)
from racing.physics import attach_static_box, create_physics_world
from racing.race.progress import resolve_track
from racing.race.runtime import race_spawn_poses, seeded_race_start_finish_pose


@pytest.mark.parametrize("seed", [1, 100, 110])
@pytest.mark.parametrize("track_id", ["mugello-short", "bahrain"])
@pytest.mark.parametrize("count", [8, 10])
def test_grid_camera_sees_car_bodies_past_walls_and_banner(seed: int, track_id: str, count: int) -> None:
    track = resolve_track(track_id)
    spawns = race_spawn_poses(count, model=track.model, random_seed=seed, shuffle_grid=False)
    cars = tuple(
        CinematicCar(str(index), index + 1, 0, spawn.position, 0, spawn.progress_distance_m)
        for index, spawn in enumerate(spawns)
    )
    pose = CinematicDirector().update(cars, delta_seconds=0, track_model=track.model, grid=True)
    assert pose is not None
    if seed == 100 and track_id == "mugello-short" and count == 8:
        assert pose.position[1] < 8.0
        assert pose.fov < 55.0
    world = create_physics_world()
    core = cast(Any, import_module("panda3d.core"))
    render = core.NodePath("grid-visibility-test")
    add_racing_scene_collisions(physics_world=world, render=render, samples=track.samples)
    start = seeded_race_start_finish_pose(model=track.model, random_seed=seed)
    banner = start_finish_render_pose(samples=track.samples, position=start.position)
    negative, positive = start_finish_banner_side_distances(samples=track.samples, position=start.position)
    heading = radians(banner.heading_degrees)
    offset = (positive - negative) / 2
    attach_static_box(
        world=world, render=render, name="Formula110-banner",
        position=(banner.position.x - cos(heading) * offset, START_FINISH_BANNER_CENTER_Y,
                  banner.position.z + sin(heading) * offset),
        half_extents=((negative + positive + START_FINISH_BANNER_OVERHANG) / 2,
                      START_FINISH_BANNER_HEIGHT / 2, START_FINISH_BANNER_THICKNESS / 2),
        heading_degrees=banner.heading_degrees,
    )
    for spawn in spawns:
        heading = radians(spawn.heading_degrees)
        for forward, side in ((0.0, 0.0), (-1.15, -.9), (-1.15, .9), (1.15, -.9), (1.15, .9)):
            for height in (0.0, 1.0):
                target = core.Point3(
                    spawn.position[0] + sin(heading) * forward + cos(heading) * side,
                    spawn.position[1] + height,
                    spawn.position[2] + cos(heading) * forward - sin(heading) * side,
                )
                hit = world.rayTestClosest(core.Point3(*pose.position), target)
                assert not hit.hasHit(), f"Car hidden by {hit.getNode().getName()} at {target}"
