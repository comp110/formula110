from __future__ import annotations

from collections.abc import Iterator
from importlib import import_module
from typing import Any, cast

import pytest

from racing.graphics.static_batching import batch_static_track_entities


@pytest.fixture
def ursina_scene(monkeypatch: pytest.MonkeyPatch) -> Iterator[Any]:
    """Exercise real Panda geometry and Ursina ownership without a GPU window."""
    ursina = cast(Any, import_module("ursina"))
    entity_module = import_module("ursina.entity")
    monkeypatch.setattr(entity_module, "_warn_if_ursina_not_instantiated", False)
    previous = {id(entity) for entity in ursina.scene.entities}
    yield ursina
    for entity in tuple(ursina.scene.entities):
        if id(entity) not in previous:
            ursina.destroy(entity)


def _triangle(ursina: Any, x: float, color: tuple[float, float, float, float]) -> Any:
    return ursina.Entity(
        model=ursina.Mesh(
            vertices=[(0, 0, 0), (1, 0, 0), (0, 0, 1)],
            triangles=[(0, 1, 2)],
            normals=[(0, 1, 0)] * 3,
            static=True,
        ),
        position=(x, 0, 0),
        color=color,
    )


def _geoms(root: Any) -> tuple[tuple[Any, Any], ...]:
    return tuple(
        (path.node().getGeom(index), path.getNetState().compose(path.node().getGeomState(index)))
        for path in root.findAllMatches("**/+GeomNode")
        for index in range(path.node().getNumGeoms())
    )


def test_batching_merges_geometry_and_retires_only_static_entities(ursina_scene: Any) -> None:
    ursina = ursina_scene
    entities = tuple(_triangle(ursina, x, (1, 0, 0, 1)) for x in (1, 3, 5))
    movable_line = _triangle(ursina, 8, (1, 1, 1, 1))

    root = batch_static_track_entities(ursina=ursina, entities=entities)

    assert len(_geoms(root)) == 1
    assert _geoms(root)[0][0].getPrimitive(0).getNumPrimitives() == 3
    minimum, maximum = root.getTightBounds(ursina.scene)
    assert tuple(minimum) == pytest.approx((1, 0, 0))
    assert tuple(maximum) == pytest.approx((6, 0, 1))
    assert root in ursina.scene.entities
    assert root.ignore
    assert all(entity not in ursina.scene.entities for entity in entities)
    assert all(entity not in ursina.scene.children for entity in entities)
    assert movable_line in ursina.scene.entities
    movable_line.x = 12
    assert movable_line.x == 12
    ursina.destroy(root)
    assert not ursina.scene.find("**/static-track")
    assert movable_line in ursina.scene.entities


def test_batching_preserves_color_material_light_and_depth_states(ursina_scene: Any) -> None:
    ursina = ursina_scene
    core = cast(Any, import_module("panda3d.core"))
    material = core.Material("kerb")
    material.setDiffuse((0.8, 0.8, 0.8, 1))
    light = ursina.scene.attachNewNode(core.Spotlight("test-track-light"))
    entities = (
        _triangle(ursina, 1, (1, 0, 0, 1)),
        _triangle(ursina, 3, (1, 1, 1, 1)),
        _triangle(ursina, 5, (1, 0, 0, 1)),
    )
    for entity in entities:
        entity.setMaterial(material, 1)
        entity.setDepthOffset(1)
    entities[0].setLight(light)
    before = {state for entity in entities for _, state in _geoms(entity)}
    try:
        root = batch_static_track_entities(ursina=ursina, entities=entities)
        after = {state for _, state in _geoms(root)}

        assert after == before
        assert len(_geoms(root)) == 3  # Different colors/lights must remain separate draws.
    finally:
        light.removeNode()


def test_distant_geometry_keeps_separate_culling_bounds(ursina_scene: Any) -> None:
    ursina = ursina_scene
    entities = tuple(_triangle(ursina, x, (1, 1, 1, 1)) for x in (-25, 1, 45))

    root = batch_static_track_entities(ursina=ursina, entities=entities)

    assert root.getNumChildren() == 3
    for cell in root.getChildren():
        minimum, maximum = cell.getTightBounds(ursina.scene)
        assert maximum.x - minimum.x == pytest.approx(1)
