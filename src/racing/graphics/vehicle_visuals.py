"""Draw formula-style cars from physics state or preview-only nodes."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from math import atan2, cos, degrees, hypot, pi, radians, sin
from typing import Any

from racing.game.config import CarShowcaseView
from racing.graphics.colors import DEFAULT_FORMULA_TEAM_COLOR, ColorRGBA
from racing.graphics.mesh_utils import mesh_from_quads
from racing.graphics.render_assets import SceneAssets, lit_entity
from racing.graphics.track_rendering import add_track_spotlight_binding_updater
from racing.physics import (
    FORMULA_VEHICLE_PHYSICS_CONFIG,
    REFERENCE_VEHICLE_LOWER_CHASSIS_HALF_WIDTH,
    VehiclePhysicsConfig,
    wheel_axis_points,
)


@dataclass(slots=True)
class RobotVisualRig:
    """Renderable stand-in for a car that does not need Bullet physics.

    Attributes:
        chassis_np: Root node for the rendered vehicle body.
        wheel_nodes: Child nodes used as wheel attachment points.
        config: Physics dimensions mirrored by the visual model.
    """

    chassis_np: Any
    wheel_nodes: tuple[Any, ...]
    config: VehiclePhysicsConfig
    team_paint_entities: tuple[Any, ...] = ()


def vehicle_visual_scale(config: VehiclePhysicsConfig) -> float:
    """Scale the car art to match a physics configuration."""
    lower_chassis_half_width = config.lower_chassis_half_extents[0]
    if lower_chassis_half_width <= 0:
        raise ValueError("lower chassis half width must be positive")
    return lower_chassis_half_width / REFERENCE_VEHICLE_LOWER_CHASSIS_HALF_WIDTH


def wheel_visual_diameter(config: VehiclePhysicsConfig) -> float:
    """Compute the visible wheel diameter from the physics wheel radius."""
    if config.wheel_radius <= 0:
        raise ValueError("wheel_radius must be positive")
    return config.wheel_radius * 2


WHEEL_VISUAL_AXLE_ROTATION_Z_DEGREES = 90.0
WHEEL_RIM_FACE_OFFSET_FRACTION = 0.504
WHEEL_HUB_FACE_OFFSET_FRACTION = 0.510
WHEEL_SIDEWALL_MARK_FACE_OFFSET_FRACTION = 0.512
COCKPIT_INSTRUMENT_BASE_POSITION = (0.0, 0.520, -0.415)
COCKPIT_INSTRUMENT_BASE_SCALE = (0.115, 0.046, 0.080)
COCKPIT_INSTRUMENT_MAST_SCALE = (0.014, 0.150, 0.014)
COCKPIT_INSTRUMENT_MAST_POSITION = (
    0.0,
    COCKPIT_INSTRUMENT_BASE_POSITION[1] + COCKPIT_INSTRUMENT_BASE_SCALE[1] / 2 + COCKPIT_INSTRUMENT_MAST_SCALE[1] / 2,
    -0.405,
)


def add_showcase_floor(*, ursina: Any) -> None:
    """Draw the simple floor and backdrop for the car preview scene."""
    lit_entity(
        ursina,
        model="cube",
        position=(0.0, -0.045, 0.0),
        scale=(7.0, 0.08, 5.8),
        color=(1.0, 1.0, 0.985, 1),
        unlit=True,
    )
    lit_entity(
        ursina,
        model="cube",
        position=(0.0, 1.15, -2.62),
        scale=(7.0, 4.4, 0.08),
        color=(1.0, 1.0, 0.985, 1),
        unlit=True,
    )
    lit_entity(
        ursina,
        model="cube",
        position=(-3.46, 1.15, 0.0),
        scale=(0.08, 4.4, 5.8),
        color=(1.0, 1.0, 0.985, 1),
        unlit=True,
    )
    lit_entity(
        ursina,
        model="cube",
        position=(3.46, 1.15, 0.0),
        scale=(0.08, 4.4, 5.8),
        color=(1.0, 1.0, 0.985, 1),
        unlit=True,
    )


def create_showcase_robot(ursina: Any, config: VehiclePhysicsConfig | None = None) -> RobotVisualRig:
    """Create a preview-only car root and wheel nodes."""
    config = FORMULA_VEHICLE_PHYSICS_CONFIG if config is None else config
    chassis_np = ursina.Entity(name="showcase-formula-car-root", position=(0.0, config.wheel_radius, 0.0))
    wheel_nodes = tuple(
        ursina.Entity(parent=chassis_np, name=f"showcase-wheel-{index}", position=connection)
        for index, connection in enumerate(wheel_axis_points(config), start=0)
    )
    return RobotVisualRig(chassis_np=chassis_np, wheel_nodes=wheel_nodes, config=config)


def pose_showcase_car(chassis_np: Any, view: CarShowcaseView) -> None:
    """Rotate the preview car for the selected showcase angle."""
    heading_degrees = 180.0 if view in (CarShowcaseView.REAR, CarShowcaseView.REAR_THREE_QUARTER) else 0.0
    pitch_degrees = 180.0 if view is CarShowcaseView.UPSIDE_DOWN else 0.0
    chassis_np.setHpr(heading_degrees, pitch_degrees, 0.0)


def apply_showcase_camera(
    *,
    ursina: Any,
    view: CarShowcaseView,
) -> None:
    """Move the camera for the selected car preview angle."""
    ursina.camera.parent = ursina.scene
    ursina.camera.orthographic = True
    ursina.camera.fov = 3.30
    if view is CarShowcaseView.TOP:
        ursina.camera.position = (0.0, 5.4, 0.0)
        ursina.camera.fov = 3.75
        ursina.camera.setHpr(0.0, -90.0, 0.0)
        return

    if view is CarShowcaseView.SIDE:
        ursina.camera.position = (4.2, 1.05, 0.0)
        ursina.camera.fov = 3.20
    elif view is CarShowcaseView.FRONT or view is CarShowcaseView.REAR:
        ursina.camera.position = (0.0, 0.72, 4.4)
        ursina.camera.fov = 2.95
    elif view is CarShowcaseView.UPSIDE_DOWN:
        ursina.camera.position = (2.8, 1.95, 3.2)
        ursina.camera.fov = 3.35
    elif view is CarShowcaseView.REAR_THREE_QUARTER:
        ursina.camera.position = (2.95, 2.35, 3.35)
        ursina.camera.fov = 3.35
    else:
        ursina.camera.position = (2.95, 2.35, 3.35)
        ursina.camera.fov = 3.35

    ursina.camera.look_at(ursina.Vec3(0.0, 0.42, 0.0))


def add_robot_visuals(
    *,
    ursina: Any,
    robot: Any,
    assets: SceneAssets,
    team_color: ColorRGBA = DEFAULT_FORMULA_TEAM_COLOR,
) -> None:
    """Attach formula-style visual pieces to a physics robot."""
    _add_formula_visuals(ursina=ursina, robot=robot, assets=assets, team_color=team_color)
    add_track_spotlight_binding_updater(ursina=ursina, node=robot.chassis_np)


def apply_robot_team_color(
    *,
    robot: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
) -> None:
    """Repaint the reusable team-color pieces on a car."""
    paint_entities = _robot_team_paint_entities(robot)
    if not paint_entities:
        paint_entities = tuple(
            entity
            for entity in _visual_descendants(robot.chassis_np)
            if _entity_uses_material(entity, assets.team_paint_material)
        )
    for entity in paint_entities:
        _set_entity_color(entity, team_color)


def _robot_team_paint_entities(robot: Any) -> tuple[Any, ...]:
    entities = getattr(robot, "team_paint_entities", ())
    return tuple(entities)


def _add_formula_team_paint_entity(
    *,
    ursina: Any,
    team_paint_entities: list[Any],
    assets: SceneAssets,
    team_color: ColorRGBA,
    **kwargs: Any,
) -> Any:
    entity = lit_entity(
        ursina,
        material=assets.team_paint_material,
        color=team_color,
        **kwargs,
    )
    team_paint_entities.append(entity)
    return entity


def _record_team_paint_entity(team_paint_entities: list[Any], entity: Any | None) -> None:
    if entity is not None:
        team_paint_entities.append(entity)


def _visual_descendants(root: Any) -> tuple[Any, ...]:
    descendants: list[Any] = []
    stack = list(_visual_children(root))
    while stack:
        entity = stack.pop()
        descendants.append(entity)
        stack.extend(_visual_children(entity))
    return tuple(descendants)


def _visual_children(entity: Any) -> tuple[Any, ...]:
    children = getattr(entity, "children", None)
    if children is not None:
        return tuple(children)
    if hasattr(entity, "getChildren"):
        return tuple(entity.getChildren())
    return ()


def _entity_uses_material(entity: Any, material: Any) -> bool:
    entity_material = getattr(entity, "material", None)
    if entity_material is material:
        return True
    if hasattr(entity, "getMaterial"):
        node_material = entity.getMaterial()
        return node_material is material or node_material == material
    return False


def _set_entity_color(entity: Any, color: ColorRGBA) -> None:
    if hasattr(entity, "color"):
        entity.color = color
        return
    if hasattr(entity, "setColor"):
        entity.setColor(*color)


def _add_formula_visuals(
    *,
    ursina: Any,
    robot: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
) -> None:
    config = robot.config
    wheel_diameter = wheel_visual_diameter(config)
    carbon_color = (0.008, 0.009, 0.010, 1)
    shadow_carbon = (0.014, 0.015, 0.016, 1)
    team_paint_entities: list[Any] = []

    lit_entity(
        ursina,
        parent=robot.chassis_np,
        model="cube",
        position=(0.0, 0.043, -0.05),
        scale=(0.98, 0.022, 1.70),
        material=assets.black_plastic_material,
        color=carbon_color,
    )
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=robot.chassis_np,
        model=_formula_body_mesh(ursina),
        double_sided=True,
    )
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=robot.chassis_np,
        model=_formula_engine_cover_mesh(ursina),
        double_sided=True,
    )
    _add_formula_center_fin(
        ursina=ursina,
        parent=robot.chassis_np,
        assets=assets,
        team_color=team_color,
        team_paint_entities=team_paint_entities,
    )

    for side in (-1, 1):
        _add_formula_sidepod(
            ursina=ursina,
            parent=robot.chassis_np,
            assets=assets,
            side=side,
            team_color=team_color,
            team_paint_entities=team_paint_entities,
        )
        _add_formula_suspension(
            ursina=ursina,
            parent=robot.chassis_np,
            assets=assets,
            side=side,
            config=config,
        )
        _add_formula_floor_edge(
            ursina=ursina,
            parent=robot.chassis_np,
            assets=assets,
            side=side,
            color=shadow_carbon,
        )

    _add_xz_slab(
        ursina=ursina,
        parent=robot.chassis_np,
        points=((-0.125, -0.335), (0.125, -0.335), (0.155, -0.075), (0.095, 0.080), (-0.095, 0.080), (-0.155, -0.075)),
        y=0.306,
        thickness=0.014,
        material=assets.black_plastic_material,
        color=(0.004, 0.004, 0.005, 1),
    )
    _add_formula_cockpit_details(
        ursina=ursina,
        parent=robot.chassis_np,
        assets=assets,
        team_color=team_color,
        team_paint_entities=team_paint_entities,
    )
    _add_formula_front_wing(
        ursina=ursina,
        parent=robot.chassis_np,
        assets=assets,
        team_color=team_color,
        team_paint_entities=team_paint_entities,
    )
    _add_formula_rear_wing(
        ursina=ursina,
        parent=robot.chassis_np,
        assets=assets,
        team_color=team_color,
        team_paint_entities=team_paint_entities,
    )
    _add_formula_diffuser(
        ursina=ursina,
        parent=robot.chassis_np,
        assets=assets,
        team_color=team_color,
        team_paint_entities=team_paint_entities,
    )

    for index, wheel_node in enumerate(robot.wheel_nodes):
        is_rear = index in (2, 3)
        _add_formula_wheel_visual(
            ursina=ursina,
            wheel_node=wheel_node,
            diameter=wheel_diameter * (1.18 if is_rear else 1.08),
            width=config.wheel_width * (1.36 if is_rear else 1.05),
            assets=assets,
            team_color=team_color,
            team_paint_entities=team_paint_entities,
        )
    robot.team_paint_entities = tuple(team_paint_entities)


def _add_formula_sidepod(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    side: int,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=parent,
        model=_formula_sidepod_mesh(ursina, side=side),
        double_sided=True,
    )
    lit_entity(
        ursina,
        parent=parent,
        model="cube",
        position=(side * 0.282, 0.202, 0.150),
        scale=(0.205, 0.078, 0.034),
        material=assets.black_plastic_material,
        color=(0.006, 0.007, 0.008, 1),
    )
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=parent,
        model="cube",
        position=(side * 0.410, 0.102, -0.250),
        scale=(0.046, 0.054, 0.585),
    )


def _add_formula_cockpit_details(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    lit_entity(
        ursina,
        parent=parent,
        model="cube",
        position=COCKPIT_INSTRUMENT_BASE_POSITION,
        scale=COCKPIT_INSTRUMENT_BASE_SCALE,
        material=assets.black_plastic_material,
        color=(0.004, 0.005, 0.006, 1),
    )
    lit_entity(
        ursina,
        parent=parent,
        model="cube",
        position=COCKPIT_INSTRUMENT_MAST_POSITION,
        scale=COCKPIT_INSTRUMENT_MAST_SCALE,
        material=assets.black_plastic_material,
        color=(0.004, 0.005, 0.006, 1),
    )
    for side in (-1, 1):
        _add_xz_bar_between(
            ursina=ursina,
            parent=parent,
            assets=assets,
            start=(side * 0.095, 0.310, 0.060),
            end=(side * 0.255, 0.335, 0.170),
            thickness=0.014,
            color=(0.006, 0.007, 0.008, 1),
        )
        _add_formula_team_paint_entity(
            ursina=ursina,
            team_paint_entities=team_paint_entities,
            assets=assets,
            team_color=team_color,
            parent=parent,
            model="cube",
            position=(side * 0.292, 0.347, 0.176),
            scale=(0.070, 0.030, 0.035),
            rotation_y=side * 12,
        )


def _add_formula_floor_edge(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    side: int,
    color: ColorRGBA,
) -> None:
    _add_xz_slab(
        ursina=ursina,
        parent=parent,
        points=(
            (side * 0.205, 0.600),
            (side * 0.585, 0.460),
            (side * 0.585, -0.745),
            (side * 0.302, -0.850),
        ),
        y=0.074,
        thickness=0.018,
        material=assets.black_plastic_material,
        color=color,
    )


def _add_formula_suspension(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    side: int,
    config: VehiclePhysicsConfig,
) -> None:
    color = (0.009, 0.010, 0.011, 1)
    front_wheel_inner_x = side * (config.wheel_track_half_width - config.wheel_width * 1.05 / 2)
    rear_wheel_inner_x = side * (config.wheel_track_half_width - config.wheel_width * 1.36 / 2)
    front_upright = (front_wheel_inner_x, 0.115, config.wheelbase_half_length)
    rear_upright = (rear_wheel_inner_x, 0.112, -config.wheelbase_half_length)
    bars = (
        ((side * 0.055, 0.085, 0.485), (front_upright[0], 0.082, front_upright[2] - 0.020)),
        ((side * 0.070, 0.135, 0.390), (front_upright[0], 0.158, front_upright[2] - 0.018)),
        ((side * 0.115, 0.062, 0.785), (front_upright[0], 0.078, front_upright[2] + 0.024)),
        ((side * 0.155, 0.135, -0.410), (rear_upright[0], 0.085, rear_upright[2] + 0.020)),
        ((side * 0.205, 0.186, -0.555), (rear_upright[0], 0.154, rear_upright[2] + 0.018)),
        ((side * 0.250, 0.062, -0.790), (rear_upright[0], 0.080, rear_upright[2] - 0.024)),
    )
    for start, end in bars:
        _add_xz_bar_between(
            ursina=ursina,
            parent=parent,
            assets=assets,
            start=start,
            end=end,
            thickness=0.018,
            color=color,
        )
    for upright in (front_upright, rear_upright):
        lit_entity(
            ursina,
            parent=parent,
            model="cube",
            position=upright,
            scale=(0.026, 0.170, 0.052),
            material=assets.black_plastic_material,
            color=color,
        )


def _add_formula_center_fin(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    _record_team_paint_entity(
        team_paint_entities,
        _add_vertical_yz_slab(
            ursina,
            parent=parent,
            assets=assets,
            x=0.0,
            thickness=0.018,
            points=((0.270, 0.035), (0.500, -0.170), (0.520, -0.555), (0.338, -0.720), (0.285, -0.320)),
            material=assets.team_paint_material,
            color=team_color,
            reverse=True,
        ),
    )


def _add_formula_front_wing(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=parent,
        model=_formula_curved_wing_mesh(
            ursina,
            sections=(
                (0.920, 0.126, 0.350, 0.034),
                (0.965, 0.108, 0.440, 0.040),
                (1.025, 0.086, 0.500, 0.046),
                (1.090, 0.069, 0.525, 0.050),
                (1.145, 0.066, 0.515, 0.044),
            ),
        ),
        double_sided=True,
    )
    for side in (-1, 1):
        _record_team_paint_entity(
            team_paint_entities,
            _add_vertical_yz_slab(
                ursina,
                parent=parent,
                assets=assets,
                x=side * 0.526,
                thickness=0.042,
                points=((0.044, 0.935), (0.054, 1.150), (0.220, 1.114), (0.206, 0.948)),
                material=assets.team_paint_material,
                color=team_color,
                reverse=side < 0,
            ),
        )


def _add_formula_rear_wing(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    _add_formula_team_paint_entity(
        ursina=ursina,
        team_paint_entities=team_paint_entities,
        assets=assets,
        team_color=team_color,
        parent=parent,
        model=_formula_curved_wing_mesh(
            ursina,
            sections=(
                (-1.050, 0.505, 0.520, 0.054),
                (-0.990, 0.520, 0.535, 0.060),
                (-0.925, 0.505, 0.530, 0.062),
                (-0.860, 0.455, 0.492, 0.056),
                (-0.805, 0.382, 0.420, 0.046),
            ),
        ),
        double_sided=True,
    )
    for side in (-1, 1):
        _record_team_paint_entity(
            team_paint_entities,
            _add_vertical_yz_slab(
                ursina,
                parent=parent,
                assets=assets,
                x=side * 0.520,
                thickness=0.052,
                points=(
                    (0.250, -1.058),
                    (0.590, -1.035),
                    (0.584, -0.925),
                    (0.535, -0.802),
                    (0.302, -0.817),
                    (0.240, -0.910),
                ),
                material=assets.team_paint_material,
                color=team_color,
                reverse=side < 0,
            ),
        )
    for x in (-0.120, 0.120):
        _add_formula_team_paint_entity(
            ursina=ursina,
            team_paint_entities=team_paint_entities,
            assets=assets,
            team_color=team_color,
            parent=parent,
            model="cube",
            position=(x, 0.300, -0.835),
            scale=(0.036, 0.360, 0.036),
            rotation_y=8 if x < 0 else -8,
        )


def _add_formula_diffuser(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    for side in (-1, 0, 1):
        _add_formula_team_paint_entity(
            ursina=ursina,
            team_paint_entities=team_paint_entities,
            assets=assets,
            team_color=team_color,
            parent=parent,
            model="cube",
            position=(side * 0.165, 0.118, -0.820),
            scale=(0.032, 0.150, 0.320),
            rotation_y=side * 10,
        )


def _add_formula_wheel_visual(
    *,
    ursina: Any,
    wheel_node: Any,
    diameter: float,
    width: float,
    assets: SceneAssets,
    team_color: ColorRGBA,
    team_paint_entities: list[Any],
) -> None:
    radius = diameter / 2
    tire_root = ursina.Entity(parent=wheel_node, rotation_z=WHEEL_VISUAL_AXLE_ROTATION_Z_DEGREES)
    lit_entity(
        ursina,
        parent=tire_root,
        model=ursina.Cylinder(resolution=96, radius=0.5, start=-0.5),
        scale=(diameter, width, diameter),
        texture=assets.tire_texture,
        material=assets.rubber_material,
        color=(0.018, 0.018, 0.017, 1),
    )
    for side_sign in (-1, 1):
        lit_entity(
            ursina,
            parent=tire_root,
            model=_rim_disc_mesh(ursina, resolution=72, vertical_sign=side_sign),
            position=(0.0, side_sign * width * WHEEL_RIM_FACE_OFFSET_FRACTION, 0.0),
            scale=(diameter * 0.66, 1.0, diameter * 0.66),
            material=assets.black_plastic_material,
            color=(0.006, 0.007, 0.008, 1),
            double_sided=True,
        )
        _add_formula_team_paint_entity(
            ursina=ursina,
            team_paint_entities=team_paint_entities,
            assets=assets,
            team_color=team_color,
            parent=tire_root,
            model=_rim_disc_mesh(ursina, resolution=48, vertical_sign=side_sign),
            position=(0.0, side_sign * width * WHEEL_HUB_FACE_OFFSET_FRACTION, 0.0),
            scale=(diameter * 0.22, 1.0, diameter * 0.22),
            double_sided=True,
        )
        for mark_index in range(10):
            angle = mark_index * 36
            lit_entity(
                ursina,
                parent=tire_root,
                model="cube",
                position=(
                    sin(radians(angle)) * radius * 0.84,
                    side_sign * width * WHEEL_SIDEWALL_MARK_FACE_OFFSET_FRACTION,
                    cos(radians(angle)) * radius * 0.84,
                ),
                rotation_y=angle,
                scale=(diameter * 0.035, width * 0.012, diameter * 0.105),
                material=assets.white_decal_material,
                color=(0.86, 0.86, 0.82, 1),
            )


def _add_xz_bar_between(
    *,
    ursina: Any,
    parent: Any,
    assets: SceneAssets,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    thickness: float,
    color: ColorRGBA,
) -> None:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    dz = end[2] - start[2]
    horizontal_length = hypot(dx, dz)
    length = hypot(horizontal_length, dy)
    if length <= 0.001:
        return
    lit_entity(
        ursina,
        parent=parent,
        model="cube",
        position=((start[0] + end[0]) / 2, (start[1] + end[1]) / 2, (start[2] + end[2]) / 2),
        rotation_x=-degrees(atan2(dy, horizontal_length)),
        rotation_y=degrees(atan2(dx, dz)),
        scale=(thickness, thickness, length),
        material=assets.black_plastic_material,
        color=color,
    )


def _add_vertical_yz_slab(
    ursina: Any,
    *,
    parent: Any,
    assets: SceneAssets,
    x: float,
    thickness: float,
    points: tuple[tuple[float, float], ...],
    color: ColorRGBA,
    reverse: bool = False,
    material: Any | None = None,
) -> Any | None:
    if len(points) < 3:
        return None

    half_thickness = thickness / 2
    positive_face = tuple((x + half_thickness, y, z) for y, z in points)
    negative_face = tuple((x - half_thickness, y, z) for y, z in points)
    quads: list[tuple[tuple[float, float, float], ...]] = [
        positive_face,
        tuple(reversed(negative_face)),
    ]
    for index, positive_point in enumerate(positive_face):
        next_index = 0 if index == len(positive_face) - 1 else index + 1
        quads.append(
            (
                positive_point,
                positive_face[next_index],
                negative_face[next_index],
                negative_face[index],
            )
        )
    if reverse:
        quads = [tuple(reversed(quad)) for quad in quads]
    return lit_entity(
        ursina,
        parent=parent,
        model=_outward_mesh_from_quads(ursina, tuple(quads)),
        material=assets.black_plastic_material if material is None else material,
        color=color,
        double_sided=True,
    )


def _add_xz_slab(
    *,
    ursina: Any,
    parent: Any,
    points: tuple[tuple[float, float], ...],
    y: float,
    thickness: float,
    color: ColorRGBA,
    material: Any | None = None,
    texture: Any | None = None,
) -> Any | None:
    if len(points) < 3:
        return None

    half_thickness = thickness / 2
    top_face = tuple((x, y + half_thickness, z) for x, z in points)
    bottom_face = tuple((x, y - half_thickness, z) for x, z in points)
    quads: list[tuple[tuple[float, float, float], ...]] = [
        top_face,
        tuple(reversed(bottom_face)),
    ]
    for index, top_point in enumerate(top_face):
        next_index = 0 if index == len(top_face) - 1 else index + 1
        quads.append((top_point, top_face[next_index], bottom_face[next_index], bottom_face[index]))
    return lit_entity(
        ursina,
        parent=parent,
        model=_outward_mesh_from_quads(ursina, tuple(quads)),
        texture=texture,
        material=material,
        color=color,
        double_sided=True,
    )


def _outward_mesh_from_quads(
    ursina: Any,
    quads: tuple[tuple[tuple[float, float, float], ...], ...],
) -> Any:
    return mesh_from_quads(ursina, _quads_oriented_away_from_center(quads))


def _quads_oriented_away_from_center(
    quads: tuple[tuple[tuple[float, float, float], ...], ...],
) -> tuple[tuple[tuple[float, float, float], ...], ...]:
    points = tuple(point for quad in quads for point in quad)
    if len(points) == 0:
        return quads
    center = (
        (min(point[0] for point in points) + max(point[0] for point in points)) / 2,
        (min(point[1] for point in points) + max(point[1] for point in points)) / 2,
        (min(point[2] for point in points) + max(point[2] for point in points)) / 2,
    )
    oriented: list[tuple[tuple[float, float, float], ...]] = []
    for quad in quads:
        if len(quad) < 3:
            oriented.append(quad)
            continue
        face_center = _quad_center(quad)
        center_to_face = (
            face_center[0] - center[0],
            face_center[1] - center[1],
            face_center[2] - center[2],
        )
        normal = _quad_face_normal(quad[0], quad[1], quad[2])
        if _dot(normal, center_to_face) < 0:
            oriented.append(tuple(reversed(quad)))
        else:
            oriented.append(quad)
    return tuple(oriented)


def _quad_center(points: tuple[tuple[float, float, float], ...]) -> tuple[float, float, float]:
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
        sum(point[2] for point in points) / len(points),
    )


def _quad_face_normal(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    third: tuple[float, float, float],
) -> tuple[float, float, float]:
    first_edge = (second[0] - first[0], second[1] - first[1], second[2] - first[2])
    second_edge = (third[0] - first[0], third[1] - first[1], third[2] - first[2])
    normal = (
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    )
    length = hypot(normal[0], hypot(normal[1], normal[2]))
    if length == 0:
        return (0.0, 1.0, 0.0)
    return (normal[0] / length, normal[1] / length, normal[2] / length)


def _dot(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _formula_curved_wing_mesh(
    ursina: Any,
    *,
    sections: tuple[tuple[float, float, float, float], ...],
) -> Any:
    section_vertices = tuple(
        (
            (-half_width, center_y + thickness * 0.5, z),
            (half_width, center_y + thickness * 0.5, z),
            (half_width, center_y - thickness * 0.5, z),
            (-half_width, center_y - thickness * 0.5, z),
        )
        for z, center_y, half_width, thickness in sections
    )
    quads: list[tuple[tuple[float, float, float], ...]] = []
    for first, second in pairwise(section_vertices):
        quads.append((first[0], second[0], second[1], first[1]))
        quads.append((first[1], second[1], second[2], first[2]))
        quads.append((first[2], second[2], second[3], first[3]))
        quads.append((first[3], second[3], second[0], first[0]))
    quads.append(section_vertices[0])
    quads.append(tuple(reversed(section_vertices[-1])))
    return mesh_from_quads(ursina, tuple(quads))


def _formula_body_mesh(ursina: Any) -> Any:
    sections = (
        (-0.760, 0.075, 0.230, 0.205),
        (-0.560, 0.080, 0.285, 0.245),
        (-0.320, 0.082, 0.330, 0.210),
        (-0.040, 0.074, 0.292, 0.155),
        (0.240, 0.064, 0.220, 0.115),
        (0.610, 0.056, 0.165, 0.074),
        (0.860, 0.050, 0.120, 0.044),
        (1.080, 0.047, 0.090, 0.030),
    )
    return _formula_closed_section_mesh(ursina, sections)


def _formula_engine_cover_mesh(ursina: Any) -> Any:
    sections = (
        (-0.700, 0.230, 0.385, 0.022),
        (-0.540, 0.276, 0.505, 0.046),
        (-0.350, 0.304, 0.515, 0.064),
        (-0.130, 0.282, 0.408, 0.046),
        (0.060, 0.238, 0.292, 0.022),
    )
    return _formula_closed_section_mesh(ursina, sections)


def _formula_sidepod_mesh(ursina: Any, *, side: int) -> Any:
    sections = (
        (-0.705, 0.060, 0.166, 0.170, 0.300, 0.218),
        (-0.505, 0.064, 0.218, 0.162, 0.405, 0.310),
        (-0.235, 0.068, 0.248, 0.150, 0.432, 0.332),
        (0.060, 0.078, 0.212, 0.128, 0.350, 0.258),
        (0.260, 0.082, 0.146, 0.102, 0.210, 0.154),
    )
    quads: list[tuple[tuple[float, float, float], ...]] = []
    section_vertices = tuple(
        (
            (side * inner_x, bottom_y, z),
            (side * outer_lower_x, bottom_y, z),
            (side * outer_lower_x, bottom_y + (top_y - bottom_y) * 0.58, z),
            (side * outer_upper_x, top_y, z),
            (side * (inner_x * 0.88), top_y * 0.97, z),
        )
        for z, bottom_y, top_y, inner_x, outer_lower_x, outer_upper_x in sections
    )
    for first, second in pairwise(section_vertices):
        for index in range(len(first)):
            next_index = 0 if index == len(first) - 1 else index + 1
            quads.append((first[index], first[next_index], second[next_index], second[index]))
    quads.append(tuple(reversed(section_vertices[0])))
    quads.append(section_vertices[-1])
    if side < 0:
        quads = [tuple(reversed(quad)) for quad in quads]
    return _outward_mesh_from_quads(ursina, tuple(quads))


def _formula_closed_section_mesh(
    ursina: Any,
    sections: tuple[tuple[float, float, float, float], ...],
) -> Any:
    profile = (-1.0, -0.82, -0.62, -0.42, -0.24, -0.10, 0.0, 0.10, 0.24, 0.42, 0.62, 0.82, 1.0)
    quads: list[tuple[tuple[float, float, float], ...]] = []
    section_vertices: list[tuple[tuple[float, float, float], ...]] = []
    for z, bottom_y, top_y, half_width in sections:
        vertices: list[tuple[float, float, float]] = []
        for u in profile:
            crown = 1.0 - abs(u) ** 1.65
            shoulder = 0.018 * max(0.0, 1.0 - abs(abs(u) - 0.62) / 0.24)
            y = bottom_y + (top_y - bottom_y) * crown + shoulder
            vertices.append((u * half_width, y, z))
        section_vertices.append(tuple(vertices))

    for first, second in pairwise(tuple(section_vertices)):
        for index in range(len(profile) - 1):
            quads.append((first[index], second[index], second[index + 1], first[index + 1]))
        quads.append((first[-1], second[-1], second[0], first[0]))

    quads.append(section_vertices[0])
    quads.append(tuple(reversed(section_vertices[-1])))
    return _outward_mesh_from_quads(ursina, tuple(quads))


def _rim_disc_mesh(ursina: Any, *, resolution: int, vertical_sign: int) -> Any:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 0.0)]
    triangles: list[tuple[int, int, int]] = []
    uvs: list[tuple[float, float]] = [(0.5, 0.5)]
    normals: list[tuple[float, float, float]] = [(0.0, float(vertical_sign), 0.0)]
    for index in range(resolution):
        angle = index * 2 * pi / resolution
        x = cos(angle) * 0.5
        z = sin(angle) * 0.5
        vertices.append((x, 0.0, z))
        uvs.append((x + 0.5, z + 0.5))
        normals.append((0.0, float(vertical_sign), 0.0))

    for index in range(1, resolution + 1):
        next_index = 1 if index == resolution else index + 1
        if vertical_sign > 0:
            triangles.append((0, index, next_index))
        else:
            triangles.append((0, next_index, index))
    return ursina.Mesh(vertices=vertices, triangles=triangles, uvs=uvs, normals=normals, static=True)
