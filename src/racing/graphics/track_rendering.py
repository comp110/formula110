"""Build the visible track, world floor, barriers, and track-side scenery."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from math import atan, atan2, degrees, hypot
from typing import Any, cast

from racing.graphics.render_assets import (
    CAROLINA_BLUE_COLOR,
    INNER_WALL_LIGHTNESS_SCALE,
    WALL_CONCRETE_DARKNESS_SCALE,
    SceneAssets,
    lit_entity,
)
from racing.graphics.static_batching import batch_static_track_entities
from racing.graphics.track_mesh import (
    clean_offset_path,
    offset_path,
    raised_kerb_chunk_meshes,
    ribbon_chunk_meshes,
    ribbon_mesh,
    wall_collision_segments_for_side,
    wall_collision_thickness,
    wall_face_band_chunk_meshes,
    wall_face_band_mesh,
    wall_offsets_for_side,
    wall_paint_offsets_for_side,
)
from racing.physics import attach_static_box
from racing.track.world import (
    START_POSITION,
    TRACK_SCALE,
    TRACK_WIDTH,
    TrackPoint,
    sampled_track_centerline,
    track_bounds,
)

START_HEADING_DEGREES = 90.0
TRACK_CURB_GAP = 0.02 * TRACK_SCALE
TRACK_CURB_WIDTH = 0.34 * TRACK_SCALE
TRACK_SURFACE_Y = 0.02 * TRACK_SCALE
TRACK_KERB_BASE_Y = TRACK_SURFACE_Y + 0.004 * TRACK_SCALE
TRACK_KERB_RIDGE_HEIGHT = 0.0225 * TRACK_SCALE
TRACK_KERB_LOW_HEIGHT = TRACK_KERB_RIDGE_HEIGHT * 0.50
TRACK_KERB_LOW_COLOR = (0.96, 0.94, 0.88, 1.0)
TRACK_KERB_RIDGE_COLOR = (0.95, 0.025, 0.018, 1.0)
TRACK_SURFACE_FRICTION = 0.80
TRACK_EDGE_BUFFER = 0.70 * TRACK_SCALE
TRACK_KERB_INNER_DISTANCE = TRACK_WIDTH / 2 + TRACK_CURB_GAP
TRACK_KERB_OUTER_DISTANCE = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER
TRACK_KERB_SLAB_LENGTH = 1.0
TRACK_WALL_THICKNESS = 0.225 * TRACK_SCALE
TRACK_WALL_HEIGHT = 0.70 * TRACK_SCALE
TRACK_WALL_BASE_Y = TRACK_SURFACE_Y
TRACK_WALL_COLLISION_THICKNESS = 0.12 * TRACK_SCALE
TRACK_WALL_TOP_SURFACE_Y = TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT + 0.004 * TRACK_SCALE
INCH_TO_METERS = 0.0254
TRACK_WALL_PAINT_BAND_WIDTH = 6.0 * INCH_TO_METERS
TRACK_WALL_PAINT_EXTRUSION = 0.0001
TRACK_WALL_PAINT_DEPTH_BIAS = 1
TRACK_WALL_PAINT_COLOR = CAROLINA_BLUE_COLOR
START_FINISH_BANNER_THICKNESS = 0.30
START_FINISH_BANNER_HEIGHT = 2.05
START_FINISH_BANNER_CENTER_Y = 3.5
START_FINISH_BANNER_FACE_OFFSET = 0.004
START_FINISH_BANNER_POLE_THICKNESS = 0.22
START_FINISH_BANNER_POLE_BOTTOM_Y = 0.1
START_FINISH_BANNER_POLE_OVERLAP_Y = 0.03
START_FINISH_BANNER_POLE_WALL_CLEARANCE = 1.25
START_FINISH_BANNER_OVERHANG = 0.45
START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO = 2172 / 293
START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION = 0.04
START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO = 2048 / 525
START_FINISH_ARGYLE_FLOOR_Y = TRACK_SURFACE_Y + 0.008 * TRACK_SCALE
START_FINISH_ARGYLE_FLOOR_WIDTH = TRACK_WIDTH
START_FINISH_ARGYLE_FLOOR_LENGTH = START_FINISH_ARGYLE_FLOOR_WIDTH / START_FINISH_ARGYLE_TEXTURE_ASPECT_RATIO
WORLD_FLOOR_VISUAL_THICKNESS = 0.08
WORLD_FLOOR_VISUAL_TOP_Y = 0.0
WORLD_FLOOR_COLLISION_HALF_HEIGHT = 0.08
WORLD_FLOOR_COLLISION_CENTER_Y = TRACK_SURFACE_Y - WORLD_FLOOR_COLLISION_HALF_HEIGHT
NIGHT_SKY_COLOR = (0.012, 0.018, 0.038, 1.0)
TRACK_LIGHT_SAMPLE_SPACING = 8
TRACK_LIGHT_NOMINAL_SPACING_M = 14.0
TRACK_LIGHT_POST_HEIGHT = 4.0 * TRACK_SCALE
TRACK_LIGHT_HEAD_CENTER_Y = TRACK_LIGHT_POST_HEIGHT - 0.04 * TRACK_SCALE
TRACK_LIGHT_SIDE_DISTANCE = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS + 0.60 * TRACK_SCALE
TRACK_LIGHT_HEAD_TRACK_OVERHANG = 1.06 * TRACK_SCALE
TRACK_LIGHT_HEAD_SIDE_DISTANCE = TRACK_WIDTH / 2 - TRACK_LIGHT_HEAD_TRACK_OVERHANG
TRACK_LIGHT_ARM_LENGTH_SCALE = 0.30
TRACK_LIGHT_RECEIVER_SEGMENTS_PER_CHUNK = 1
TRACK_LIGHTS_PER_RECEIVER = 4
TRACK_SPOTLIGHTS_PER_VEHICLE = 4
TRACK_SPOTLIGHTS_SCENE_TAG = "racing_track_spotlights"
TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX: dict[int, int] = {
    1: 5,
    2: 4,
    3: 7,
    4: 9,
    5: 6,
    6: 8,
    7: 10,
    8: 11,
    9: 12,
    10: 13,
    11: 14,
    12: 15,
    13: 17,
    14: 19,
    15: 16,
    16: 18,
    17: 21,
    18: 20,
    19: 23,
    20: 22,
    21: 24,
    22: 0,
    23: 1,
    24: 3,
    25: 2,
}
TRACK_REMOVED_STREETLIGHT_NUMBERS = frozenset((3, 4, 5, 10, 11, 13, 15, 17, 19, 22, 23, 24))
TRACK_REMOVED_STREETLIGHT_RENDER_INDICES = frozenset(
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[number] for number in TRACK_REMOVED_STREETLIGHT_NUMBERS
)
TRACK_STREETLIGHT_FRACTION_OVERRIDES_BY_RENDER_INDEX: dict[int, float] = {
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[25]: 0.06,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[1]: 0.18,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[2]: 0.195,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[6]: 0.30,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[8]: 0.42,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[9]: 0.50,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[14]: 0.74,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[20]: 0.92,
}
TRACK_STREETLIGHT_SIDE_OVERRIDES_BY_RENDER_INDEX: dict[int, int] = {
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[12]: -1,
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[20]: 1,
}
TRACK_STREETLIGHT_POST_OFFSETS_BY_RENDER_INDEX: dict[int, tuple[float, float]] = {
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[21]: (0.0, -2.5),
    TRACK_STREETLIGHT_NUMBER_TO_RENDER_INDEX[20]: (-0.34689626973758336, 0.6649533653138013),
}
TRACK_SPOTLIGHT_LUMEN_SCALE = 0.90
TRACK_SPOTLIGHT_INTENSITY_SCALE = 1.25 * 1.25 * TRACK_SPOTLIGHT_LUMEN_SCALE
TRACK_SPOTLIGHT_TARGET_RADIUS_SCALE = 2.0
TRACK_SPOTLIGHT_TARGET_RADIUS = TRACK_WIDTH * TRACK_SPOTLIGHT_TARGET_RADIUS_SCALE
TRACK_SPOTLIGHT_AIM_INWARD_SCALE = 0.25
TRACK_WALL_COLLISION_EXTRA_THICKNESS = 0.0
TRACK_WALL_COLLISION_OVERLAP = 0.0
TRACK_WARM_LIGHT_COLOR = (1.0, 0.86, 0.62, 1.0)
TRACK_SURFACE_LIGHT_WASH_COLOR = (0.24, 0.19, 0.12, 0.22)


@dataclass(frozen=True, slots=True)
class TrackLightLayout:
    """World positions used to render one track light."""

    post: tuple[float, float, float]
    head: tuple[float, float, float]
    target: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class StartFinishGantry:
    """Renderable start/finish banner parts that move with a seeded race."""

    ursina: Any
    root: Any
    negative_side_pole: Any
    positive_side_pole: Any
    banner_backing: Any
    banner_logo: Any
    samples: tuple[TrackPoint, ...]


@dataclass(frozen=True, slots=True)
class StartFinishRenderPose:
    """Start/finish centerpoint and heading aligned to rendered track offsets."""

    position: TrackPoint
    heading_degrees: float


@dataclass(frozen=True, slots=True)
class StartFinishTrackSlice:
    """Cross-track slice through the rendered centerline and outside walls."""

    pose: StartFinishRenderPose
    negative_wall_distance: float
    positive_wall_distance: float


@dataclass(frozen=True, slots=True)
class TrackSpotlight:
    """Panda light node and relevant world points for one lamp."""

    node_path: Any
    source: tuple[float, float, float]
    aim: tuple[float, float, float]
    target: tuple[float, float, float]


def add_world_floor(
    *,
    ursina: Any,
    physics_world: Any,
    assets: SceneAssets,
    include_collision: bool = True,
    samples: tuple[TrackPoint, ...] | None = None,
) -> None:
    """Draw the large ground plane under the whole track."""
    track_samples = sampled_track_centerline(samples_per_segment=10) if samples is None else samples
    bounds = track_bounds(points=track_samples, margin=10 * TRACK_SCALE)
    center_x = (bounds.min_x + bounds.max_x) / 2
    center_z = (bounds.min_z + bounds.max_z) / 2
    floor_padding = 4 * TRACK_SCALE

    lit_entity(
        ursina,
        model="cube",
        position=(center_x, WORLD_FLOOR_VISUAL_TOP_Y - WORLD_FLOOR_VISUAL_THICKNESS / 2, center_z),
        scale=(bounds.width + floor_padding * 2, WORLD_FLOOR_VISUAL_THICKNESS, bounds.length + floor_padding * 2),
        texture=assets.grass_texture,
        material=assets.concrete_material,
        texture_scale=(bounds.width / 6, bounds.length / 6),
        color=(1, 1, 1, 1),
    )
    if include_collision:
        add_world_floor_collision(physics_world=physics_world, render=ursina.scene, points=track_samples)


def add_world_floor_collision(
    *,
    physics_world: Any,
    render: Any,
    points: tuple[TrackPoint, ...] | None = None,
) -> None:
    """Add the flat Bullet collider that keeps cars from falling forever."""
    bounds = track_bounds(
        points=sampled_track_centerline(samples_per_segment=10) if points is None else points,
        margin=10 * TRACK_SCALE,
    )
    center_x = (bounds.min_x + bounds.max_x) / 2
    center_z = (bounds.min_z + bounds.max_z) / 2
    half_x = bounds.width / 2
    half_z = bounds.length / 2
    floor_padding = 4 * TRACK_SCALE

    attach_static_box(
        world=physics_world,
        render=render,
        name="grass-and-track-floor",
        position=(center_x, WORLD_FLOOR_COLLISION_CENTER_Y, center_z),
        half_extents=(half_x + floor_padding, WORLD_FLOOR_COLLISION_HALF_HEIGHT, half_z + floor_padding),
        friction=TRACK_SURFACE_FRICTION,
    )


def add_track(
    *,
    ursina: Any,
    physics_world: Any,
    assets: SceneAssets,
    samples: tuple[TrackPoint, ...],
    start_line_position: TrackPoint = START_POSITION,
    start_line_heading_degrees: float = START_HEADING_DEGREES,
    include_collision: bool = True,
    legacy_lighting: bool = False,
) -> Any:
    """Draw a centerline-driven racing track and optional colliders."""
    wall_inside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER
    track_light_receivers: list[Any] = []
    static_entities: list[Any] = []

    track_light_receivers.extend(
        _add_lit_ribbon_chunks(
            ursina,
            samples=samples,
            inner_offset=-TRACK_WIDTH / 2,
            outer_offset=TRACK_WIDTH / 2,
            y=TRACK_SURFACE_Y,
            uv_scale=5.4 * TRACK_SCALE,
            texture=assets.asphalt_texture,
            material=assets.asphalt_material,
            color=(1, 1, 1, 1),
            double_sided=True,
        )
    )
    static_entities.extend(
        _add_unlit_ribbon_chunks(
            ursina,
            samples=samples,
            inner_offset=-TRACK_WIDTH / 2 + TRACK_CURB_WIDTH * 0.45,
            outer_offset=TRACK_WIDTH / 2 - TRACK_CURB_WIDTH * 0.45,
            y=TRACK_SURFACE_Y + 0.006 * TRACK_SCALE,
            uv_scale=5.4 * TRACK_SCALE,
            color=TRACK_SURFACE_LIGHT_WASH_COLOR,
        )
    )
    for side in (-1, 1):
        wall_inside_offset, wall_outside_offset, _ = wall_offsets_for_side(
            side,
            inside_distance=wall_inside_distance,
            thickness=TRACK_WALL_THICKNESS,
        )
        track_light_receivers.extend(
            _add_lit_ribbon_chunks(
                ursina,
                samples=samples,
                inner_offset=side * (TRACK_WIDTH / 2),
                outer_offset=wall_inside_offset,
                y=0.024 * TRACK_SCALE,
                uv_scale=3.4 * TRACK_SCALE,
                texture=assets.gravel_texture,
                material=assets.concrete_material,
                color=(1, 1, 1, 1),
                double_sided=True,
            )
        )
        track_light_receivers.extend(
            _add_lit_raised_kerb_chunks(
                ursina,
                samples=samples,
                inner_offset=side * TRACK_KERB_INNER_DISTANCE,
                outer_offset=side * TRACK_KERB_OUTER_DISTANCE,
                uv_scale=1.4 * TRACK_SCALE,
                texture=assets.curb_texture,
                material=assets.kerb_material,
            )
        )
        track_light_receivers.extend(
            _add_wall_concrete_faces(
                ursina=ursina,
                samples=samples,
                wall_inside_offset=wall_inside_offset,
                wall_outside_offset=wall_outside_offset,
                assets=assets,
            )
        )
        track_light_receivers.extend(
            _add_wall_edge_paint(
                ursina=ursina,
                samples=samples,
                side=side,
                wall_inside_offset=wall_inside_offset,
                wall_outside_offset=wall_outside_offset,
                assets=assets,
            )
        )
        static_entities.append(
            lit_entity(
                ursina,
                model=ribbon_mesh(
                    ursina,
                    samples,
                    inner_offset=wall_inside_offset - side * 0.18 * TRACK_SCALE,
                    outer_offset=wall_inside_offset,
                    y=0.056 * TRACK_SCALE,
                    uv_scale=2.5 * TRACK_SCALE,
                ),
                color=(0.03, 0.035, 0.03, 0.38),
                double_sided=True,
                unlit=True,
            )
        )
    static_entities.extend(
        _add_track_night_lights(
            ursina=ursina,
            samples=samples,
            receivers=tuple(track_light_receivers),
            legacy_layout=legacy_lighting,
        )
    )
    batch_static_track_entities(ursina=ursina, entities=(*track_light_receivers, *static_entities))

    if include_collision:
        add_mugello_short_track_collisions(physics_world=physics_world, render=ursina.scene)

    return _add_track_start_line(
        ursina=ursina,
        assets=assets,
        position=start_line_position,
        heading_degrees=start_line_heading_degrees,
    )


def add_mugello_short_track(
    *,
    ursina: Any,
    physics_world: Any,
    assets: SceneAssets,
    start_line_position: TrackPoint = START_POSITION,
    start_line_heading_degrees: float = START_HEADING_DEGREES,
    include_collision: bool = True,
) -> Any:
    """Draw the default Mugello-inspired track and optional colliders."""
    return add_track(
        ursina=ursina,
        physics_world=physics_world,
        assets=assets,
        samples=sampled_track_centerline(samples_per_segment=10),
        start_line_position=start_line_position,
        start_line_heading_degrees=start_line_heading_degrees,
        include_collision=include_collision,
        legacy_lighting=True,
    )


def _add_lit_ribbon_chunks(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    inner_offset: float,
    outer_offset: float,
    y: float,
    uv_scale: float,
    texture: Any | None,
    material: Any,
    color: tuple[float, float, float, float],
    double_sided: bool,
) -> tuple[Any, ...]:
    entities: list[Any] = []
    for mesh in ribbon_chunk_meshes(
        ursina,
        samples,
        inner_offset=inner_offset,
        outer_offset=outer_offset,
        y=y,
        uv_scale=uv_scale,
        segments_per_chunk=TRACK_LIGHT_RECEIVER_SEGMENTS_PER_CHUNK,
    ):
        kwargs: dict[str, Any] = {
            "model": mesh,
            "material": material,
            "color": color,
            "double_sided": double_sided,
        }
        if texture is not None:
            kwargs["texture"] = texture
        entities.append(
            lit_entity(
                ursina,
                **kwargs,
            )
        )
    return tuple(entities)


def _add_wall_edge_paint(
    *,
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    side: int,
    wall_inside_offset: float,
    wall_outside_offset: float,
    assets: SceneAssets,
) -> tuple[Any, ...]:
    paint_offsets = wall_paint_offsets_for_side(
        side,
        inside_offset=wall_inside_offset,
        outside_offset=wall_outside_offset,
        band_width=TRACK_WALL_PAINT_BAND_WIDTH,
        extrusion=TRACK_WALL_PAINT_EXTRUSION,
    )
    wall_top_y = TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT
    paint_top_y = TRACK_WALL_TOP_SURFACE_Y + TRACK_WALL_PAINT_EXTRUSION
    band_bottom_y = wall_top_y - TRACK_WALL_PAINT_BAND_WIDTH
    wall_top_width = abs(paint_offsets.top_outer_offset - paint_offsets.top_inner_offset)
    entities: list[Any] = []
    entities.append(
        _add_lit_wall_face_band(
            ursina,
            samples=samples,
            face_offset=paint_offsets.inner_face_offset,
            bottom_y=band_bottom_y,
            top_y=wall_top_y,
            flip_normal=False,
            material=assets.wall_paint_material,
        )
    )
    entities.extend(
        _add_lit_wall_top_band_chunks(
            ursina,
            samples=samples,
            edge_offset=paint_offsets.top_inner_offset,
            side=side,
            band_width=wall_top_width,
            y=paint_top_y,
            uv_scale=1.0 * TRACK_SCALE,
            extends_outside=True,
            material=assets.wall_paint_material,
            texture=None,
            color=TRACK_WALL_PAINT_COLOR,
        )
    )
    return tuple(entities)


def _add_lit_wall_face_band(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    face_offset: float,
    bottom_y: float,
    top_y: float,
    flip_normal: bool,
    material: Any,
    texture: Any | None = None,
    color: tuple[float, float, float, float] = TRACK_WALL_PAINT_COLOR,
) -> Any:
    kwargs: dict[str, Any] = {
        "model": wall_face_band_mesh(
            ursina,
            samples,
            face_offset=face_offset,
            bottom_y=bottom_y,
            top_y=top_y,
            uv_scale=1.0 * TRACK_SCALE,
            flip_normal=flip_normal,
        ),
        "material": material,
        "color": color,
        "double_sided": True,
    }
    if texture is not None:
        kwargs["texture"] = texture
    entity = lit_entity(ursina, **kwargs)
    _apply_wall_paint_depth_bias(entity)
    return entity


def _add_lit_wall_face_band_chunks(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    face_offset: float,
    bottom_y: float,
    top_y: float,
    flip_normal: bool,
    material: Any,
    texture: Any | None = None,
    color: tuple[float, float, float, float] = TRACK_WALL_PAINT_COLOR,
) -> tuple[Any, ...]:
    entities: list[Any] = []
    for mesh in wall_face_band_chunk_meshes(
        ursina,
        samples,
        face_offset=face_offset,
        bottom_y=bottom_y,
        top_y=top_y,
        uv_scale=1.0 * TRACK_SCALE,
        flip_normal=flip_normal,
        segments_per_chunk=TRACK_LIGHT_RECEIVER_SEGMENTS_PER_CHUNK,
    ):
        kwargs: dict[str, Any] = {
            "model": mesh,
            "material": material,
            "color": color,
            "double_sided": True,
        }
        if texture is not None:
            kwargs["texture"] = texture
        entities.append(
            lit_entity(
                ursina,
                **kwargs,
            )
        )
    return tuple(entities)


def _add_wall_concrete_faces(
    *,
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    wall_inside_offset: float,
    wall_outside_offset: float,
    assets: SceneAssets,
) -> tuple[Any, ...]:
    entities: list[Any] = []
    entities.extend(
        _add_lit_wall_face_band_chunks(
            ursina,
            samples=samples,
            face_offset=wall_inside_offset,
            bottom_y=TRACK_WALL_BASE_Y,
            top_y=TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT,
            flip_normal=False,
            material=assets.inner_wall_material,
            texture=assets.wall_texture,
            color=(
                INNER_WALL_LIGHTNESS_SCALE * WALL_CONCRETE_DARKNESS_SCALE,
                INNER_WALL_LIGHTNESS_SCALE * WALL_CONCRETE_DARKNESS_SCALE,
                INNER_WALL_LIGHTNESS_SCALE * WALL_CONCRETE_DARKNESS_SCALE,
                1,
            ),
        )
    )
    entities.extend(
        _add_lit_wall_face_band_chunks(
            ursina,
            samples=samples,
            face_offset=wall_outside_offset,
            bottom_y=TRACK_WALL_BASE_Y,
            top_y=TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT,
            flip_normal=True,
            material=assets.concrete_material,
            texture=assets.wall_texture,
            color=(
                WALL_CONCRETE_DARKNESS_SCALE,
                WALL_CONCRETE_DARKNESS_SCALE,
                WALL_CONCRETE_DARKNESS_SCALE,
                1,
            ),
        )
    )
    return tuple(entities)


def _add_lit_wall_top_band_chunks(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    edge_offset: float,
    side: int,
    band_width: float,
    y: float,
    uv_scale: float,
    extends_outside: bool,
    material: Any,
    texture: Any | None,
    color: tuple[float, float, float, float],
) -> tuple[Any, ...]:
    across_offset = edge_offset + side * band_width if extends_outside else edge_offset - side * band_width
    kwargs: dict[str, Any] = {
        "model": ribbon_mesh(
            ursina,
            samples,
            inner_offset=edge_offset,
            outer_offset=across_offset,
            y=y,
            uv_scale=uv_scale,
        ),
        "material": material,
        "color": color,
        "double_sided": True,
    }
    if texture is not None:
        kwargs["texture"] = texture
    entity = lit_entity(ursina, **kwargs)
    _apply_wall_paint_depth_bias(entity)
    return (entity,)


def _apply_wall_paint_depth_bias(entity: Any) -> None:
    if hasattr(entity, "setDepthOffset"):
        entity.setDepthOffset(TRACK_WALL_PAINT_DEPTH_BIAS)


def _add_lit_raised_kerb_chunks(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    inner_offset: float,
    outer_offset: float,
    uv_scale: float,
    texture: Any,
    material: Any,
) -> tuple[Any, ...]:
    entities: list[Any] = []
    for index, mesh in enumerate(
        raised_kerb_chunk_meshes(
            ursina,
            samples,
            inner_offset=inner_offset,
            outer_offset=outer_offset,
            base_y=TRACK_KERB_BASE_Y,
            low_height=TRACK_KERB_LOW_HEIGHT,
            ridge_height=TRACK_KERB_RIDGE_HEIGHT,
            uv_scale=uv_scale,
            slab_length=TRACK_KERB_SLAB_LENGTH,
        )
    ):
        entities.append(
            lit_entity(
                ursina,
                model=mesh,
                texture=texture,
                material=material,
                color=_kerb_slab_color(index),
                double_sided=True,
            )
        )
    return tuple(entities)


def _kerb_slab_color(index: int) -> tuple[float, float, float, float]:
    return TRACK_KERB_RIDGE_COLOR if index % 2 == 0 else TRACK_KERB_LOW_COLOR


def _add_unlit_ribbon_chunks(
    ursina: Any,
    *,
    samples: tuple[TrackPoint, ...],
    inner_offset: float,
    outer_offset: float,
    y: float,
    uv_scale: float,
    color: tuple[float, float, float, float],
) -> tuple[Any, ...]:
    entities: list[Any] = []
    for mesh in ribbon_chunk_meshes(
        ursina,
        samples,
        inner_offset=inner_offset,
        outer_offset=outer_offset,
        y=y,
        uv_scale=uv_scale,
        segments_per_chunk=TRACK_LIGHT_RECEIVER_SEGMENTS_PER_CHUNK,
    ):
        entities.append(
            lit_entity(
                ursina,
                model=mesh,
                color=color,
                double_sided=True,
                unlit=True,
            )
        )
    return tuple(entities)


def _add_track_start_line(
    *,
    ursina: Any,
    assets: SceneAssets,
    position: TrackPoint,
    heading_degrees: float,
) -> Any:
    root = ursina.Entity(name="start-finish-track-line")
    set_start_finish_pose(root, position=position, heading_degrees=heading_degrees)
    lit_entity(
        ursina,
        parent=root,
        model=_start_finish_floor_argyle_mesh(ursina),
        position=(0.0, START_FINISH_ARGYLE_FLOOR_Y, 0.0),
        texture=assets.argyle_banner_texture,
        material=assets.argyle_banner_material,
        color=(1, 1, 1, 1),
        double_sided=True,
        unlit=True,
    )
    return root


def set_start_finish_pose(entity: Any, *, position: TrackPoint, heading_degrees: float) -> None:
    """Place a start/finish visual root on the racing line."""
    entity.position = (position.x, 0.0, position.z)
    entity.rotation_y = _start_finish_rotation_y(heading_degrees)


def _start_finish_rotation_y(heading_degrees: float) -> float:
    return START_HEADING_DEGREES - heading_degrees


def add_mugello_short_track_collisions(
    *,
    physics_world: Any,
    render: Any,
    samples: tuple[TrackPoint, ...] | None = None,
) -> None:
    """Add mugello short track collisions."""
    track_samples = sampled_track_centerline(samples_per_segment=10) if samples is None else samples
    wall_inside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER
    collision_thickness = wall_collision_thickness(
        visual_thickness=TRACK_WALL_COLLISION_THICKNESS,
        extra_thickness=TRACK_WALL_COLLISION_EXTRA_THICKNESS,
    )
    for side in (-1, 1):
        for center_x, center_z, length, heading in wall_collision_segments_for_side(
            track_samples,
            side=side,
            inside_distance=wall_inside_distance,
            visual_thickness=TRACK_WALL_COLLISION_THICKNESS,
            extra_thickness=TRACK_WALL_COLLISION_EXTRA_THICKNESS,
        ):
            attach_static_box(
                world=physics_world,
                render=render,
                name=f"track-barrier-{side}",
                position=(
                    center_x,
                    TRACK_WALL_BASE_Y + TRACK_WALL_HEIGHT / 2,
                    center_z,
                ),
                half_extents=(
                    length / 2 + TRACK_WALL_COLLISION_OVERLAP,
                    TRACK_WALL_HEIGHT / 2,
                    collision_thickness / 2,
                ),
                heading_degrees=-heading,
            )


def add_racing_scene_collisions(
    *,
    physics_world: Any,
    render: Any,
    samples: tuple[TrackPoint, ...] | None = None,
) -> None:
    """Add the canonical static Bullet colliders for the playable racing scene."""
    add_world_floor_collision(physics_world=physics_world, render=render, points=samples)
    add_mugello_short_track_collisions(physics_world=physics_world, render=render, samples=samples)


def _add_track_night_lights(
    *,
    ursina: Any,
    samples: tuple[TrackPoint, ...],
    receivers: tuple[Any, ...],
    legacy_layout: bool,
) -> tuple[Any, ...]:
    spotlights: list[TrackSpotlight] = []
    entities: list[Any] = []
    for render_index, layout in enumerate(track_light_layouts(samples, legacy_layout=legacy_layout)):
        if legacy_layout and render_index in TRACK_REMOVED_STREETLIGHT_RENDER_INDICES:
            continue
        lamp_x, _, lamp_z = layout.post
        head_x, _, head_z = layout.head
        entities.append(
            lit_entity(
                ursina,
                model="cube",
                position=(lamp_x, TRACK_LIGHT_POST_HEIGHT / 2, lamp_z),
                scale=(0.10 * TRACK_SCALE, TRACK_LIGHT_POST_HEIGHT, 0.10 * TRACK_SCALE),
                color=(0.055, 0.058, 0.060, 1),
                unlit=True,
            )
        )
        entities.append(
            lit_entity(
                ursina,
                model=_lamp_arm_mesh(
                    ursina,
                    start=(lamp_x, TRACK_LIGHT_POST_HEIGHT, lamp_z),
                    end=(head_x, TRACK_LIGHT_POST_HEIGHT, head_z),
                    thickness=0.08 * TRACK_SCALE,
                ),
                color=(0.055, 0.058, 0.060, 1),
                unlit=True,
            )
        )
        entities.append(
            lit_entity(
                ursina,
                model="sphere",
                position=(head_x, TRACK_LIGHT_HEAD_CENTER_Y, head_z),
                scale=(0.32 * TRACK_SCALE, 0.14 * TRACK_SCALE, 0.32 * TRACK_SCALE),
                color=(1.0, 0.86, 0.52, 1),
                unlit=True,
            )
        )
        spotlights.append(_add_track_spotlight(ursina=ursina, layout=layout))

    _register_track_spotlights(ursina=ursina, spotlights=tuple(spotlights))
    _bind_track_spotlights_to_receivers(receivers=receivers, spotlights=tuple(spotlights))
    return tuple(entities)


def _register_track_spotlights(*, ursina: Any, spotlights: tuple[TrackSpotlight, ...]) -> None:
    ursina.scene.setPythonTag(TRACK_SPOTLIGHTS_SCENE_TAG, tuple(spotlight.node_path for spotlight in spotlights))


def track_spotlight_node_paths(ursina: Any) -> tuple[Any, ...]:
    """Return track streetlight NodePaths registered in the current scene."""
    scene = getattr(ursina, "scene", None)
    if scene is None or not hasattr(scene, "hasPythonTag") or not scene.hasPythonTag(TRACK_SPOTLIGHTS_SCENE_TAG):
        return ()
    return cast(tuple[Any, ...], scene.getPythonTag(TRACK_SPOTLIGHTS_SCENE_TAG))


def bind_track_spotlights_to_node(*, ursina: Any, node: Any) -> None:
    """Bind all track streetlights to a render node and its descendants."""
    for spotlight_node_path in track_spotlight_node_paths(ursina):
        node.setLight(spotlight_node_path)


def bind_nearest_track_spotlights_to_node(
    *,
    ursina: Any,
    node: Any,
    max_lights: int = TRACK_SPOTLIGHTS_PER_VEHICLE,
) -> None:
    """Bind the closest track streetlights to a render node."""
    spotlights = track_spotlight_node_paths(ursina)
    if len(spotlights) == 0:
        return

    for spotlight_node_path in spotlights:
        node.clearLight(spotlight_node_path)

    node_x, _, node_z = node.getPos(ursina.scene)
    nearest_spotlights = sorted(
        spotlights,
        key=lambda spotlight: track_spotlight_relevance_distance_squared(
            float(node_x),
            float(node_z),
            source=cast(tuple[float, float, float], spotlight.getPythonTag("track_source_point")),
            aim=cast(tuple[float, float, float], spotlight.getPythonTag("track_aim_point")),
            target=cast(tuple[float, float, float], spotlight.getPythonTag("track_target_point")),
        ),
    )[:max_lights]
    for spotlight_node_path in nearest_spotlights:
        node.setLight(spotlight_node_path)


def add_track_spotlight_binding_updater(*, ursina: Any, node: Any) -> None:
    """Keep a moving render node bound to nearby track streetlights."""
    bind_nearest_track_spotlights_to_node(ursina=ursina, node=node)

    def update() -> None:
        """Refresh the nearby lights for this moving node."""
        bind_nearest_track_spotlights_to_node(ursina=ursina, node=node)

    ursina.Entity(name=f"track_spotlight_binding_{id(node)}", update=update, ignore_paused=True)


def _add_track_spotlight(*, ursina: Any, layout: TrackLightLayout) -> TrackSpotlight:
    core = cast(Any, import_module("panda3d.core"))
    light_node = core.Spotlight("track-spotlight")
    light_node.setColor(core.VBase4(*track_spotlight_color()))
    light_node.setAttenuation(core.LVector3(1.0, 0.020, 0.003))
    light_node.setExponent(0.30)
    light_node.getLens().setFov(track_spotlight_fov_degrees(layout))
    light_node.getLens().setNearFar(0.1, 20.0 * TRACK_SCALE)

    source = track_light_source_position(layout)
    aim = track_light_aim_position(layout)
    target = track_light_target_position(layout)
    light_np = ursina.scene.attachNewNode(light_node)
    light_np.setPos(core.Point3(*source))
    light_np.lookAt(core.Point3(*aim), core.Vec3(1.0, 0.0, 0.0))
    light_np.setPythonTag("track_source_point", source)
    light_np.setPythonTag("track_aim_point", aim)
    light_np.setPythonTag("track_target_point", target)
    return TrackSpotlight(node_path=light_np, source=source, aim=aim, target=target)


def _bind_track_spotlights_to_receivers(
    *,
    receivers: tuple[Any, ...],
    spotlights: tuple[TrackSpotlight, ...],
) -> None:
    for receiver in receivers:
        center_x, center_z = _receiver_center_xz(receiver)
        nearest_spotlights = sorted(
            spotlights,
            key=lambda spotlight: track_spotlight_relevance_distance_squared(
                center_x,
                center_z,
                source=spotlight.source,
                aim=spotlight.aim,
                target=spotlight.target,
            ),
        )[:TRACK_LIGHTS_PER_RECEIVER]
        for spotlight in nearest_spotlights:
            receiver.setLight(spotlight.node_path)


def _receiver_center_xz(receiver: Any) -> tuple[float, float]:
    bounds = receiver.get_tight_bounds()
    if bounds is None:
        return 0.0, 0.0
    minimum, maximum = bounds
    return (float(minimum[0] + maximum[0]) / 2, float(minimum[2] + maximum[2]) / 2)


def _xz_distance_squared(first_x: float, first_z: float, second_x: float, second_z: float) -> float:
    delta_x = first_x - second_x
    delta_z = first_z - second_z
    return delta_x * delta_x + delta_z * delta_z


def track_spotlight_relevance_distance_squared(
    x: float,
    z: float,
    *,
    source: tuple[float, float, float],
    aim: tuple[float, float, float],
    target: tuple[float, float, float],
) -> float:
    """Return the closest X/Z distance to the lamp source, beam aim, or target."""
    return min(
        _xz_distance_squared(x, z, source[0], source[2]),
        _xz_distance_squared(x, z, aim[0], aim[2]),
        _xz_distance_squared(x, z, target[0], target[2]),
    )


def track_light_source_position(layout: TrackLightLayout) -> tuple[float, float, float]:
    """Return the world-space source point for a rendered track light."""
    head_x, _, head_z = layout.head
    return (head_x, TRACK_LIGHT_HEAD_CENTER_Y, head_z)


def track_light_target_position(layout: TrackLightLayout) -> tuple[float, float, float]:
    """Return the track-surface point used to bind this lamp to road receivers."""
    target_x, _, target_z = layout.target
    return (target_x, TRACK_SURFACE_Y, target_z)


def track_light_aim_position(layout: TrackLightLayout) -> tuple[float, float, float]:
    """Return the slightly inward aim point for the rendered spotlight cone."""
    head_x, _, head_z = layout.head
    target_x, _, target_z = layout.target
    return (
        head_x + (target_x - head_x) * TRACK_SPOTLIGHT_AIM_INWARD_SCALE,
        TRACK_SURFACE_Y,
        head_z + (target_z - head_z) * TRACK_SPOTLIGHT_AIM_INWARD_SCALE,
    )


def track_spotlight_color() -> tuple[float, float, float, float]:
    """Return the color used by track spotlights after intensity scaling."""
    red, green, blue, alpha = TRACK_WARM_LIGHT_COLOR
    return (
        red * TRACK_SPOTLIGHT_INTENSITY_SCALE,
        green * TRACK_SPOTLIGHT_INTENSITY_SCALE,
        blue * TRACK_SPOTLIGHT_INTENSITY_SCALE,
        alpha,
    )


def track_spotlight_fov_degrees(layout: TrackLightLayout) -> float:
    """Return the spotlight cone angle needed for the requested target radius."""
    source_x, source_y, source_z = track_light_source_position(layout)
    target_x, target_y, target_z = track_light_target_position(layout)
    horizontal_distance = hypot(target_x - source_x, target_z - source_z)
    target_distance = hypot(horizontal_distance, target_y - source_y)
    if target_distance == 0:
        return 0.0
    return degrees(2 * atan(TRACK_SPOTLIGHT_TARGET_RADIUS / target_distance))


def track_light_layouts(
    samples: tuple[TrackPoint, ...],
    *,
    legacy_layout: bool = True,
) -> tuple[TrackLightLayout, ...]:
    """Return evenly spaced alternating lamp layouts around cleaned offset paths."""
    if len(samples) == 0:
        return ()

    post_paths = {side: clean_offset_path(samples, side * TRACK_LIGHT_SIDE_DISTANCE, 0.0) for side in (-1, 1)}
    center_path = tuple((sample.x, 0.0, sample.z) for sample in samples)
    if legacy_layout:
        placement_fractions = tuple(
            sample_index / len(samples) for sample_index in track_light_sample_indices(len(samples))
        )
    else:
        centerline_length = sum(
            hypot(
                samples[(index + 1) % len(samples)].x - sample.x,
                samples[(index + 1) % len(samples)].z - sample.z,
            )
            for index, sample in enumerate(samples)
        )
        light_count = max(4, 2 * round(centerline_length / (2 * TRACK_LIGHT_NOMINAL_SPACING_M)))
        placement_fractions = tuple(index / light_count for index in range(light_count))

    layouts: list[TrackLightLayout] = []
    for render_index, default_fraction in enumerate(placement_fractions):
        default_side = -1 if render_index % 2 == 0 else 1
        side = (
            TRACK_STREETLIGHT_SIDE_OVERRIDES_BY_RENDER_INDEX.get(render_index, default_side)
            if legacy_layout
            else default_side
        )
        fraction = (
            TRACK_STREETLIGHT_FRACTION_OVERRIDES_BY_RENDER_INDEX.get(render_index, default_fraction)
            if legacy_layout
            else default_fraction
        )
        if legacy_layout:
            post = _path_point_at_fraction(post_paths[side], fraction)
            offset_x, offset_z = TRACK_STREETLIGHT_POST_OFFSETS_BY_RENDER_INDEX.get(render_index, (0.0, 0.0))
            if offset_x != 0.0 or offset_z != 0.0:
                post = (post[0] + offset_x, post[1], post[2] + offset_z)
            center_x, center_z, distance_to_centerline, tangent_x, tangent_z = _nearest_centerline_frame(
                samples,
                x=post[0],
                z=post[2],
            )
        else:
            center, tangent_x, tangent_z = _path_frame_at_fraction(center_path, fraction)
            center_x, _, center_z = center
            post = (
                center_x - tangent_z * side * TRACK_LIGHT_SIDE_DISTANCE,
                0.0,
                center_z + tangent_x * side * TRACK_LIGHT_SIDE_DISTANCE,
            )
            distance_to_centerline = TRACK_LIGHT_SIDE_DISTANCE
        inward_x = -tangent_z
        inward_z = tangent_x
        if inward_x * (center_x - post[0]) + inward_z * (center_z - post[2]) < 0:
            inward_x = -inward_x
            inward_z = -inward_z
        head_offset = distance_to_centerline
        scaled_head_offset = head_offset * TRACK_LIGHT_ARM_LENGTH_SCALE
        layouts.append(
            TrackLightLayout(
                post=post,
                head=(post[0] + inward_x * scaled_head_offset, 0.0, post[2] + inward_z * scaled_head_offset),
                target=(center_x, 0.0, center_z),
            )
        )
    return tuple(layouts)


def _nearest_centerline_frame(
    samples: tuple[TrackPoint, ...],
    *,
    x: float,
    z: float,
) -> tuple[float, float, float, float, float]:
    best_x = samples[0].x
    best_z = samples[0].z
    best_distance = float("inf")
    best_tangent_x = 0.0
    best_tangent_z = 1.0
    for index, start in enumerate(samples):
        end = samples[(index + 1) % len(samples)]
        segment_x = end.x - start.x
        segment_z = end.z - start.z
        segment_length_squared = segment_x * segment_x + segment_z * segment_z
        if segment_length_squared == 0:
            candidate_x = start.x
            candidate_z = start.z
        else:
            t = ((x - start.x) * segment_x + (z - start.z) * segment_z) / segment_length_squared
            clamped_t = min(max(t, 0.0), 1.0)
            candidate_x = start.x + segment_x * clamped_t
            candidate_z = start.z + segment_z * clamped_t
        distance = hypot(x - candidate_x, z - candidate_z)
        if distance < best_distance:
            segment_length = hypot(segment_x, segment_z)
            best_x = candidate_x
            best_z = candidate_z
            best_distance = distance
            best_tangent_x = segment_x / segment_length if segment_length > 0 else 0.0
            best_tangent_z = segment_z / segment_length if segment_length > 0 else 1.0
    return best_x, best_z, best_distance, best_tangent_x, best_tangent_z


def _path_point_at_fraction(
    points: tuple[tuple[float, float, float], ...],
    fraction: float,
) -> tuple[float, float, float]:
    point, _, _ = _path_frame_at_fraction(points, fraction)
    return point


def _path_frame_at_fraction(
    points: tuple[tuple[float, float, float], ...],
    fraction: float,
) -> tuple[tuple[float, float, float], float, float]:
    if len(points) == 0:
        raise ValueError("path must contain at least one point")
    segment_lengths = tuple(
        hypot(
            points[(index + 1) % len(points)][0] - point[0],
            points[(index + 1) % len(points)][2] - point[2],
        )
        for index, point in enumerate(points)
    )
    total_length = sum(segment_lengths)
    if total_length == 0:
        return points[0], 0.0, 1.0

    target_distance = (fraction % 1.0) * total_length
    walked_distance = 0.0
    for index, segment_length in enumerate(segment_lengths):
        next_point = points[(index + 1) % len(points)]
        if walked_distance + segment_length >= target_distance:
            point = points[index]
            t = 0.0 if segment_length == 0 else (target_distance - walked_distance) / segment_length
            interpolated = (
                point[0] + (next_point[0] - point[0]) * t,
                point[1] + (next_point[1] - point[1]) * t,
                point[2] + (next_point[2] - point[2]) * t,
            )
            if segment_length == 0:
                return interpolated, 0.0, 1.0
            return (
                interpolated,
                (next_point[0] - point[0]) / segment_length,
                (next_point[2] - point[2]) / segment_length,
            )
        walked_distance += segment_length

    last_point = points[-1]
    first_point = points[0]
    last_length = hypot(first_point[0] - last_point[0], first_point[2] - last_point[2])
    if last_length == 0:
        return last_point, 0.0, 1.0
    return last_point, (first_point[0] - last_point[0]) / last_length, (first_point[2] - last_point[2]) / last_length


def _lamp_arm_mesh(
    ursina: Any,
    *,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    thickness: float,
) -> Any:
    start_x, center_y, start_z = start
    end_x, _, end_z = end
    direction_x = end_x - start_x
    direction_z = end_z - start_z
    length = hypot(direction_x, direction_z)
    if length == 0:
        raise ValueError("lamp arm endpoints must be distinct")

    unit_x = direction_x / length
    unit_z = direction_z / length
    perpendicular_x = -unit_z
    perpendicular_z = unit_x
    half_width = thickness / 2
    half_height = thickness / 2

    vertices: list[tuple[float, float, float]] = []
    for x, z in ((start_x, start_z), (end_x, end_z)):
        vertices.extend(
            (
                (x + perpendicular_x * half_width, center_y - half_height, z + perpendicular_z * half_width),
                (x - perpendicular_x * half_width, center_y - half_height, z - perpendicular_z * half_width),
                (x - perpendicular_x * half_width, center_y + half_height, z - perpendicular_z * half_width),
                (x + perpendicular_x * half_width, center_y + half_height, z + perpendicular_z * half_width),
            )
        )

    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 7, 6),
        (4, 6, 5),
        (0, 4, 5),
        (0, 5, 1),
        (3, 2, 6),
        (3, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
    ]
    return ursina.Mesh(vertices=vertices, triangles=triangles, static=True)


def track_light_sample_indices(sample_count: int, *, spacing: int = TRACK_LIGHT_SAMPLE_SPACING) -> tuple[int, ...]:
    """Choose evenly spaced centerline samples for lamp placement."""
    if sample_count <= 0:
        return ()
    if spacing < 1:
        raise ValueError("spacing must be at least one")
    return tuple(range(0, sample_count, spacing))


def add_trackside_scenery(
    *,
    ursina: Any,
    assets: SceneAssets,
    start_line_position: TrackPoint = START_POSITION,
    start_line_heading_degrees: float = START_HEADING_DEGREES,
    samples: tuple[TrackPoint, ...] | None = None,
) -> Any:
    """Add trees, lamps, and the start/finish gantry around the track."""
    track_samples = sampled_track_centerline(samples_per_segment=10) if samples is None else samples
    for index, position in enumerate(trackside_scenery_positions(samples=track_samples)):
        x, y, z = position
        lit_entity(
            ursina,
            model="cube",
            position=(x, y + 0.75, z),
            scale=(0.35, 1.5, 0.35),
            texture=assets.wall_texture,
            material=assets.concrete_material,
            color=(0.72, 0.48, 0.28, 1),
        )
        lit_entity(
            ursina,
            model="sphere",
            position=(x, y + 1.75, z),
            scale=(1.35, 1.0, 1.35),
            texture=assets.grass_texture,
            material=assets.concrete_material,
            color=(0.45, 0.85, 0.44, 1),
        )
        lit_entity(
            ursina,
            model="cube",
            position=(x + 1.2, y + 0.65, z - 0.8),
            scale=(0.92, 0.55, 0.18),
            rotation_y=45,
            texture=assets.wall_texture,
            material=assets.concrete_material,
            color=((0.95, 0.82, 0.23, 1), (0.24, 0.47, 0.78, 1))[index % 2],
        )
        for tire_index in range(3):
            lit_entity(
                ursina,
                model=ursina.Cylinder(resolution=24, radius=0.5, start=-0.5),
                position=(x - 1.15, y + 0.23 + tire_index * 0.22, z + 0.85),
                rotation_x=90,
                scale=(0.72, 0.22, 0.72),
                texture=assets.tire_texture,
                material=assets.rubber_material,
                color=(0.12, 0.12, 0.12, 1),
            )

    return _add_start_finish_gantry(
        ursina=ursina,
        assets=assets,
        position=start_line_position,
        heading_degrees=start_line_heading_degrees,
        samples=track_samples,
    )


def _add_start_finish_gantry(
    *,
    ursina: Any,
    assets: SceneAssets,
    position: TrackPoint,
    heading_degrees: float,
    samples: tuple[TrackPoint, ...] | None = None,
) -> StartFinishGantry:
    track_samples = sampled_track_centerline(samples_per_segment=10) if samples is None else samples
    root = ursina.Entity(name="start-finish-gantry")
    negative_side_pole = _add_start_finish_banner_pole(
        ursina=ursina,
        assets=assets,
        parent=root,
        side_offset=-_start_finish_banner_pole_side_distance(),
    )
    positive_side_pole = _add_start_finish_banner_pole(
        ursina=ursina,
        assets=assets,
        parent=root,
        side_offset=_start_finish_banner_pole_side_distance(),
    )
    banner_backing, banner_logo = _add_start_finish_banner(
        ursina=ursina,
        assets=assets,
        parent=root,
        width=_start_finish_banner_pole_span() + START_FINISH_BANNER_OVERHANG,
    )
    gantry = StartFinishGantry(
        ursina=ursina,
        root=root,
        negative_side_pole=negative_side_pole,
        positive_side_pole=positive_side_pole,
        banner_backing=banner_backing,
        banner_logo=banner_logo,
        samples=track_samples,
    )
    set_start_finish_gantry_pose(gantry, position=position, heading_degrees=heading_degrees)
    return gantry


def set_start_finish_gantry_pose(
    gantry: StartFinishGantry,
    *,
    position: TrackPoint,
    heading_degrees: float,
) -> None:
    """Place a start/finish gantry and resize it to clear the rendered walls."""
    render_pose = start_finish_render_pose(samples=gantry.samples, position=position)
    set_start_finish_pose(
        gantry.root,
        position=render_pose.position,
        heading_degrees=render_pose.heading_degrees,
    )
    negative_distance, positive_distance = start_finish_banner_side_distances(
        samples=gantry.samples,
        position=render_pose.position,
    )
    pole_center_y = _start_finish_banner_pole_center_y()
    gantry.negative_side_pole.position = (0.0, pole_center_y, -negative_distance)
    gantry.positive_side_pole.position = (0.0, pole_center_y, positive_distance)

    banner_width = negative_distance + positive_distance + START_FINISH_BANNER_OVERHANG
    banner_center_z = (positive_distance - negative_distance) / 2
    banner_position = (0.0, START_FINISH_BANNER_CENTER_Y, banner_center_z)
    gantry.banner_backing.position = banner_position
    gantry.banner_backing.scale = (START_FINISH_BANNER_THICKNESS, START_FINISH_BANNER_HEIGHT, banner_width)
    gantry.banner_logo.position = banner_position
    gantry.banner_logo.model = _start_finish_banner_logo_mesh(
        gantry.ursina,
        width=banner_width,
        height=START_FINISH_BANNER_HEIGHT,
        thickness=START_FINISH_BANNER_THICKNESS + START_FINISH_BANNER_FACE_OFFSET,
    )


def _add_start_finish_banner(
    *,
    ursina: Any,
    assets: SceneAssets,
    parent: Any,
    width: float,
) -> tuple[Any, Any]:
    backing = lit_entity(
        ursina,
        parent=parent,
        model="cube",
        position=(0.0, START_FINISH_BANNER_CENTER_Y, 0.0),
        scale=(START_FINISH_BANNER_THICKNESS, START_FINISH_BANNER_HEIGHT, width),
        material=assets.formula_banner_material,
        color=(1.0, 1.0, 1.0, 1.0),
        unlit=True,
    )
    logo = lit_entity(
        ursina,
        parent=parent,
        model=_start_finish_banner_logo_mesh(
            ursina,
            width=width,
            height=START_FINISH_BANNER_HEIGHT,
            thickness=START_FINISH_BANNER_THICKNESS + START_FINISH_BANNER_FACE_OFFSET,
        ),
        position=(0.0, START_FINISH_BANNER_CENTER_Y, 0.0),
        texture=assets.formula_banner_texture,
        material=assets.formula_banner_material,
        color=(1.0, 1.0, 1.0, 1.0),
        double_sided=True,
        unlit=True,
    )
    return backing, logo


def _add_start_finish_banner_pole(*, ursina: Any, assets: SceneAssets, parent: Any, side_offset: float) -> Any:
    pole_height = _start_finish_banner_pole_height()
    return lit_entity(
        ursina,
        parent=parent,
        model=ursina.Cylinder(resolution=24, radius=0.5, start=-0.5),
        position=(0.0, _start_finish_banner_pole_center_y(), side_offset),
        scale=(START_FINISH_BANNER_POLE_THICKNESS, pole_height, START_FINISH_BANNER_POLE_THICKNESS),
        material=assets.concrete_material,
        color=(0.95, 0.79, 0.18, 1),
        unlit=True,
    )


def _start_finish_banner_pole_span() -> float:
    return _start_finish_banner_pole_side_distance() * 2


def _start_finish_banner_pole_side_distance() -> float:
    wall_outside_distance = TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS
    return wall_outside_distance + START_FINISH_BANNER_POLE_THICKNESS / 2 + START_FINISH_BANNER_POLE_WALL_CLEARANCE


def start_finish_banner_side_distances(
    *,
    samples: tuple[TrackPoint, ...],
    position: TrackPoint,
) -> tuple[float, float]:
    """Find gantry pole offsets wide enough to clear both outside walls."""
    track_slice = start_finish_track_slice(samples=samples, position=position)
    return (
        _start_finish_banner_side_distance(wall_distance=track_slice.negative_wall_distance),
        _start_finish_banner_side_distance(wall_distance=track_slice.positive_wall_distance),
    )


def start_finish_render_pose(
    *,
    position: TrackPoint,
    samples: tuple[TrackPoint, ...] | None = None,
) -> StartFinishRenderPose:
    """Align the start/finish art to the nearest rendered track slice."""
    if samples is None:
        samples = sampled_track_centerline(samples_per_segment=10)
    return start_finish_track_slice(samples=samples, position=position).pose


def start_finish_track_slice(
    *,
    samples: tuple[TrackPoint, ...],
    position: TrackPoint,
) -> StartFinishTrackSlice:
    """Find the rendered track slice closest to a requested centerline point."""
    segment_index, fraction = _nearest_start_finish_sample_segment_fraction(samples=samples, position=position)
    center_x, center_z = _start_finish_centerline_point(
        samples=samples,
        segment_index=segment_index,
        fraction=fraction,
    )
    negative_x, negative_z = _start_finish_offset_point(
        samples=samples,
        segment_index=segment_index,
        fraction=fraction,
        offset=_start_finish_outside_wall_offset(side=-1),
    )
    positive_x, positive_z = _start_finish_offset_point(
        samples=samples,
        segment_index=segment_index,
        fraction=fraction,
        offset=_start_finish_outside_wall_offset(side=1),
    )
    across_x = positive_x - negative_x
    across_z = positive_z - negative_z
    return StartFinishTrackSlice(
        pose=StartFinishRenderPose(
            position=TrackPoint(center_x, center_z),
            heading_degrees=_start_finish_heading_for_across_vector(across_x=across_x, across_z=across_z),
        ),
        negative_wall_distance=hypot(center_x - negative_x, center_z - negative_z),
        positive_wall_distance=hypot(positive_x - center_x, positive_z - center_z),
    )


def _start_finish_banner_side_distance(*, wall_distance: float) -> float:
    nominal_distance = _start_finish_banner_pole_side_distance()
    pole_radius = START_FINISH_BANNER_POLE_THICKNESS / 2
    return max(nominal_distance, wall_distance + pole_radius + START_FINISH_BANNER_POLE_WALL_CLEARANCE)


def _nearest_start_finish_sample_segment_fraction(
    *,
    samples: tuple[TrackPoint, ...],
    position: TrackPoint,
) -> tuple[int, float]:
    if len(samples) < 2:
        raise ValueError("start/finish placement requires at least two track samples")

    best_segment_index = 0
    best_fraction = 0.0
    best_distance_squared = float("inf")
    for index, sample in enumerate(samples):
        next_sample = samples[(index + 1) % len(samples)]
        segment_x = next_sample.x - sample.x
        segment_z = next_sample.z - sample.z
        segment_length_squared = segment_x * segment_x + segment_z * segment_z
        if segment_length_squared <= 0.0:
            continue

        fraction = ((position.x - sample.x) * segment_x + (position.z - sample.z) * segment_z) / segment_length_squared
        fraction = max(0.0, min(1.0, fraction))
        nearest_x = sample.x + segment_x * fraction
        nearest_z = sample.z + segment_z * fraction
        distance_squared = (position.x - nearest_x) ** 2 + (position.z - nearest_z) ** 2
        if distance_squared < best_distance_squared:
            best_segment_index = index
            best_fraction = fraction
            best_distance_squared = distance_squared

    return best_segment_index, best_fraction


def _start_finish_centerline_point(
    *,
    samples: tuple[TrackPoint, ...],
    segment_index: int,
    fraction: float,
) -> tuple[float, float]:
    start = samples[segment_index]
    end = samples[(segment_index + 1) % len(samples)]
    return (
        start.x + (end.x - start.x) * fraction,
        start.z + (end.z - start.z) * fraction,
    )


def _start_finish_offset_point(
    *,
    samples: tuple[TrackPoint, ...],
    segment_index: int,
    fraction: float,
    offset: float,
) -> tuple[float, float]:
    points = offset_path(samples, offset, 0.0)
    start = points[segment_index]
    end = points[(segment_index + 1) % len(points)]
    return (
        start[0] + (end[0] - start[0]) * fraction,
        start[2] + (end[2] - start[2]) * fraction,
    )


def _start_finish_outside_wall_offset(*, side: int) -> float:
    _, outside_offset, _ = wall_offsets_for_side(
        side,
        inside_distance=TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER,
        thickness=TRACK_WALL_THICKNESS,
    )
    return outside_offset


def _start_finish_heading_for_across_vector(*, across_x: float, across_z: float) -> float:
    if hypot(across_x, across_z) <= 0.0:
        return START_HEADING_DEGREES
    return degrees(atan2(across_x, across_z)) + 90.0


def _start_finish_banner_pole_center_y() -> float:
    return START_FINISH_BANNER_POLE_BOTTOM_Y + _start_finish_banner_pole_height() / 2


def _start_finish_banner_pole_height() -> float:
    pole_top_y = _start_finish_banner_bottom_y() + START_FINISH_BANNER_POLE_OVERLAP_Y
    return pole_top_y - START_FINISH_BANNER_POLE_BOTTOM_Y


def _start_finish_banner_bottom_y() -> float:
    return START_FINISH_BANNER_CENTER_Y - START_FINISH_BANNER_HEIGHT / 2


def _start_finish_formula_logo_size(*, width: float, height: float) -> tuple[float, float]:
    available_height = height * (1.0 - 2 * START_FINISH_FORMULA_VERTICAL_MARGIN_FRACTION)
    full_width_height = width / START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO
    if full_width_height <= available_height:
        return width, full_width_height
    return available_height * START_FINISH_FORMULA_TEXTURE_ASPECT_RATIO, available_height


def _start_finish_banner_logo_mesh(
    ursina: Any,
    *,
    width: float,
    height: float,
    thickness: float,
) -> Any:
    half_width = width / 2
    half_height = height / 2
    half_thickness = thickness / 2
    logo_width, logo_height = _start_finish_formula_logo_size(width=width, height=height)
    u_min = 0.5 - width / (2 * logo_width)
    u_max = 0.5 + width / (2 * logo_width)
    v_min = 0.5 - height / (2 * logo_height)
    v_max = 0.5 + height / (2 * logo_height)
    vertices = [
        (half_thickness, -half_height, -half_width),
        (half_thickness, -half_height, half_width),
        (half_thickness, half_height, half_width),
        (half_thickness, half_height, -half_width),
        (-half_thickness, -half_height, half_width),
        (-half_thickness, -half_height, -half_width),
        (-half_thickness, half_height, -half_width),
        (-half_thickness, half_height, half_width),
    ]
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
        (4, 5, 6),
        (4, 6, 7),
    ]
    uvs = [
        (u_max, v_min),
        (u_min, v_min),
        (u_min, v_max),
        (u_max, v_max),
        (u_max, v_min),
        (u_min, v_min),
        (u_min, v_max),
        (u_max, v_max),
    ]
    normals = [
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
    ]
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def _start_finish_floor_argyle_mesh(ursina: Any) -> Any:
    half_length = START_FINISH_ARGYLE_FLOOR_LENGTH / 2
    half_width = START_FINISH_ARGYLE_FLOOR_WIDTH / 2
    vertices = [
        (-half_length, 0.0, -half_width),
        (-half_length, 0.0, half_width),
        (half_length, 0.0, half_width),
        (half_length, 0.0, -half_width),
    ]
    triangles = [
        (0, 1, 2),
        (0, 2, 3),
    ]
    uvs = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]
    normals = [
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)


def trackside_scenery_positions(
    *,
    samples: tuple[TrackPoint, ...] | None = None,
) -> tuple[tuple[float, float, float], ...]:
    """List fixed decorative positions around the track."""
    track_samples = sampled_track_centerline(samples_per_segment=10) if samples is None else samples
    bounds = track_bounds(
        points=track_samples,
        margin=TRACK_LIGHT_SIDE_DISTANCE + 4.0 * TRACK_SCALE,
    )
    inset = 2.0 * TRACK_SCALE
    return (
        (bounds.min_x + inset, 0.0, bounds.min_z + inset),
        (bounds.min_x + inset, 0.0, bounds.max_z - inset),
        (bounds.max_x - inset, 0.0, bounds.max_z - inset),
        (bounds.max_x - inset, 0.0, bounds.min_z + inset),
    )
