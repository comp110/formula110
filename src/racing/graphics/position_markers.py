"""Broadcast leader cards with translucent pointers to visible top-three cars."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from racing.graphics.colors import ColorRGBA

Point = tuple[float, float]
Rect = tuple[float, float, float, float]  # left, bottom, right, top
CARD_WIDTH = 0.64
CARD_HEIGHT = 0.14
POSITION_WIDTH = 0.12
CARD_TOP = 0.95
CARD_GAP = 0.05
VIEWPORT_MARGIN = 0.04
POINTER_CAR_GAP = 0.014


def marker_text_color(color: ColorRGBA) -> ColorRGBA:
    """Choose black or white by relative luminance for the stronger contrast."""
    linear = tuple(
        channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in color[:3]
    )
    luminance = sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))
    return (0.0, 0.0, 0.0, 1.0) if luminance > 0.179 else (1.0, 1.0, 1.0, 1.0)


@dataclass(frozen=True, slots=True)
class PositionTarget:
    car_id: str
    rank: int
    color: ColorRGBA
    name: str
    position: Point
    pointer_position: Point


@dataclass(frozen=True, slots=True)
class LeaderCard:
    center: Point
    scale: float
    pointer: tuple[Point, Point, Point]

    @property
    def bounds(self) -> Rect:
        x, y = self.center
        return (
            x - CARD_WIDTH * self.scale / 2,
            y - CARD_HEIGHT * self.scale / 2,
            x + CARD_WIDTH * self.scale / 2,
            y + CARD_HEIGHT * self.scale / 2,
        )


def leader_card_layout(
    targets: tuple[PositionTarget, ...], aspect: float, obstacles: tuple[Rect, ...], *, viewport: Rect | None = None,
    focused_car_id: str | None = None,
) -> dict[str, LeaderCard]:
    """Reserve top-three slots, or one centered card for a split pane's focus."""
    view_left, view_bottom, view_right, view_top = viewport or (-aspect, -1.0, aspect, 1.0)
    card_top = view_top - (1.0 - CARD_TOP)
    intervals = [(view_left + VIEWPORT_MARGIN, view_right - VIEWPORT_MARGIN)]
    for left, bottom, right, top in obstacles:
        if bottom > card_top or top < card_top - CARD_HEIGHT:
            continue
        remaining: list[tuple[float, float]] = []
        for start, end in intervals:
            if right + CARD_GAP <= start or left - CARD_GAP >= end:
                remaining.append((start, end))
            else:
                if start < left - CARD_GAP:
                    remaining.append((start, left - CARD_GAP))
                if right + CARD_GAP < end:
                    remaining.append((right + CARD_GAP, end))
        intervals = remaining
    if not intervals:
        return {}
    left, right = max(intervals, key=lambda interval: interval[1] - interval[0])
    columns = 3 if focused_car_id is None else 1
    scale = min(1.0, (right - left) / (columns * CARD_WIDTH + (columns - 1) * CARD_GAP))
    if scale < 0.5:
        return {}
    row_center = (left + right) / 2
    bottom = card_top - CARD_HEIGHT * scale
    cards: dict[str, LeaderCard] = {}
    for target in targets:
        if focused_car_id is not None and target.car_id != focused_car_id:
            continue
        column = target.rank - 1 if focused_car_id is None else 0
        x, y = target.position
        tip_x, tip_y = target.pointer_position
        if (
            target.rank < 1 or (focused_car_id is None and target.rank > 3)
            or not view_left <= x <= view_right
            or not view_bottom <= y <= view_top
            or not view_left <= tip_x <= view_right
            or not view_bottom <= tip_y < bottom - POINTER_CAR_GAP
            or bottom < view_bottom + VIEWPORT_MARGIN
            or any(a <= x <= c and b <= y <= d for a, b, c, d in obstacles)
        ):
            continue
        center_x = row_center + (column - (columns - 1) / 2) * (CARD_WIDTH + CARD_GAP) * scale
        card_left = center_x - CARD_WIDTH * scale / 2
        cards[target.car_id] = LeaderCard(
            center=(center_x, card_top - CARD_HEIGHT * scale / 2),
            scale=scale,
            pointer=((card_left, bottom), (card_left + POSITION_WIDTH * scale, bottom), target.pointer_position),
        )
    return cards


@dataclass(slots=True)
class _CardWidgets:
    root: Any
    rank_background: Any
    stripe: Any
    rank: Any
    name: Any
    pointer: Any
    vertices: Any


class PositionMarkers:
    """Keep each visible leader's name, rank, paint, and live pointer together."""

    def __init__(self, ursina: Any) -> None:
        self._ursina = ursina
        self._core = cast(Any, import_module("panda3d.core"))
        self._y_up = self._core.getDefaultCoordinateSystem() in (self._core.CSYupRight, self._core.CSYupLeft)
        self._root = ursina.application.base.aspect2d.attachNewNode("race-position-markers")
        self._root.setDepthTest(False, 100)
        self._root.setDepthWrite(False, 100)
        self._root.setLightOff(100)
        self._root.setShaderOff(100)
        self._root.setTransparency(self._core.TransparencyAttrib.MAlpha)
        font_path = Path(str(ursina.__file__)).parent / "fonts" / "OpenSans-Regular.ttf"
        self._font = ursina.application.base.loader.loadFont(str(font_path)) if font_path.is_file() else None
        self._markers: dict[str, _CardWidgets] = {}
        self.positions: dict[str, Point] = {}

    def hide(self) -> None:
        self._root.hide()
        self.positions.clear()

    def update(
        self, targets: tuple[PositionTarget, ...], obstacles: tuple[Rect, ...], *, viewport: Rect | None = None,
        focused_car_id: str | None = None,
    ) -> None:
        cards = leader_card_layout(
            targets, float(self._ursina.camera.aspect_ratio), obstacles,
            viewport=viewport, focused_car_id=focused_car_id,
        )
        self.positions = {car_id: card.center for car_id, card in cards.items()}
        self._root.show()
        for car_id, widgets in self._markers.items():
            if car_id not in cards:
                widgets.root.hide()
                widgets.pointer.hide()
        for target in targets:
            card = cards.get(target.car_id)
            if card is None:
                continue
            if target.car_id not in self._markers:
                self._markers[target.car_id] = self._create_marker(target.car_id)
            widgets = self._markers[target.car_id]
            widgets.root.setPos(*self._position(*card.center))
            widgets.root.setScale(card.scale)
            widgets.rank_background.setColor(*target.color[:3], 1.0)
            widgets.stripe.setColor(*target.color[:3], 1.0)
            rank = f"P{target.rank}"
            if widgets.rank.node().getText() != rank:
                widgets.rank.node().setText(rank)
            widgets.rank.node().setTextColor(*marker_text_color(target.color))
            name = " ".join(target.name.split())
            node = widgets.name.node()
            if float(node.calcWidth(name)) * 0.038 > CARD_WIDTH - 0.18:
                while name and float(node.calcWidth(name + "…")) * 0.038 > CARD_WIDTH - 0.18:
                    name = name[:-1]
                name = name.rstrip() + "…"
            if node.getText() != name:
                node.setText(name)
            writer = self._core.GeomVertexWriter(widgets.vertices, "vertex")
            for point in card.pointer:
                writer.setData3f(*self._position(*point))
            widgets.pointer.setColor(*target.color[:3], 0.36)
            widgets.root.show()
            widgets.pointer.show()

    def _position(self, x: float, y: float) -> tuple[float, float, float]:
        return (x, y, 0.0) if self._y_up else (x, 0.0, y)

    def _card(self, parent: Any, name: str, bounds: Rect, color: ColorRGBA, order: int) -> Any:
        left, bottom, right, top = bounds
        maker = self._core.CardMaker(name)
        maker.setFrame(
            *(self._core.Point3(*self._position(x, y)) for x, y in (
                (left, bottom), (right, bottom), (right, top), (left, top)
            ))
        )
        path = parent.attachNewNode(maker.generate())
        path.setColor(*color)
        path.setTwoSided(True)
        path.setBin("fixed", order, 100)
        return path

    def _text(self, parent: Any, name: str, x: float, scale: float) -> Any:
        node = self._core.TextNode(name)
        if self._font is not None:
            node.setFont(self._font)
        node.setTextColor(0.96, 0.97, 0.99, 1.0)
        path = parent.attachNewNode(node)
        path.setPos(*self._position(x, -0.012))
        path.setScale(scale)
        path.setBin("fixed", 104, 100)
        return path

    def _create_marker(self, car_id: str) -> _CardWidgets:
        core = self._core
        root = self._root.attachNewNode(f"leader-card-{car_id}")
        left, right, bottom, top = -CARD_WIDTH / 2, CARD_WIDTH / 2, -CARD_HEIGHT / 2, CARD_HEIGHT / 2
        self._card(root, "leader-background", (left, bottom, right, top), (0.025, 0.034, 0.050, 0.97), 102)
        rank_background = self._card(
            root, "leader-color", (left, bottom, left + POSITION_WIDTH, top), (1, 1, 1, 1), 103
        )
        stripe = self._card(root, "leader-stripe", (left, bottom, right, bottom + 0.009), (1, 1, 1, 1), 103)
        rank = self._text(root, "leader-rank", left + POSITION_WIDTH / 2, 0.044)
        rank.node().setAlign(core.TextNode.ACenter)
        name = self._text(root, "leader-name", left + 0.15, 0.038)
        vertices = core.GeomVertexData(f"leader-pointer-{car_id}", core.GeomVertexFormat.getV3(), core.Geom.UHDynamic)
        vertices.setNumRows(3)
        triangle = core.GeomTriangles(core.Geom.UHStatic)
        triangle.addVertices(0, 1, 2)
        geometry = core.Geom(vertices)
        geometry.addPrimitive(triangle)
        node = core.GeomNode(f"leader-pointer-{car_id}")
        node.addGeom(geometry)
        pointer = self._root.attachNewNode(node)
        pointer.setTwoSided(True)
        pointer.setBin("fixed", 98, 100)
        return _CardWidgets(root, rank_background, stripe, rank, name, pointer, vertices)
