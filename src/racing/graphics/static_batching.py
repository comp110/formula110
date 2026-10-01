"""Combine static track visuals without changing their lighting or colors."""

from __future__ import annotations

from importlib import import_module
from math import floor
from typing import Any, cast

STATIC_TRACK_CELL_SIZE = 20.0


def batch_static_track_entities(*, ursina: Any, entities: tuple[Any, ...]) -> Any:
    """Consume finalized track visuals into spatial batches owned by one entity.

    Call after assigning materials and local lights. Only pass static visual
    entities: their individual transforms and Ursina handles are retired here.
    Cars, colliders, lights, and the movable start line/gantry stay independent.
    """
    core = cast(Any, import_module("panda3d.core"))
    root = ursina.Entity(name="static-track", ignore=True)
    cells: dict[tuple[int, int], Any] = {}
    for entity in entities:
        minimum, maximum = entity.getTightBounds(ursina.scene)
        center = (minimum + maximum) * 0.5
        key = (floor(center.x / STATIC_TRACK_CELL_SIZE), floor(center.z / STATIC_TRACK_CELL_SIZE))
        if key not in cells:
            cells[key] = root.attachNewNode(f"track-cell-{key[0]}-{key[1]}")
        entity.world_parent = cells[key]
        # Ursina's per-entity raycast tag otherwise prevents nodes from merging.
        entity.clearPythonTag("Entity")

    entity_ids = {id(entity) for entity in entities}
    ursina.scene.entities[:] = [entity for entity in ursina.scene.entities if id(entity) not in entity_ids]

    reducer = core.SceneGraphReducer()
    for cell in cells.values():
        # flattenStrong() also bakes ColorScale into vertex colors, which changes
        # material-lit kerbs and wall paint. Bake transforms only; the reducer
        # then merges geometry only where the original render states agree.
        reducer.applyAttribs(cell.node(), core.SceneGraphReducer.TTTransform)
        reducer.flatten(
            cell.node(),
            core.SceneGraphReducer.CSRecurse | core.SceneGraphReducer.CSGeomNode | core.SceneGraphReducer.CSOther,
        )
        reducer.collectVertexData(cell.node(), 0)
        reducer.unify(cell.node(), False)
    return root
