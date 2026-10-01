"""Five red starting lights in the race viewer's shared screen-space HUD."""

from importlib import import_module
from math import cos, pi, sin
from typing import Any, cast

from racing.race.start import START_LIGHT_COUNT, RaceStartSequence


class StartLights:
    """Keep the start signal visible in every camera, including split-screen."""

    def __init__(self, ursina: Any) -> None:
        self._base = ursina.application.base
        self._core = cast(Any, import_module("panda3d.core"))
        self._y_up = self._core.getDefaultCoordinateSystem() in (self._core.CSYupRight, self._core.CSYupLeft)
        self.root = self._base.aspect2d.attachNewNode("race-start-lights")
        self.root.setDepthTest(False)
        self.root.setDepthWrite(False)
        self.root.setLightOff(1)
        self.root.setTransparency(self._core.TransparencyAttrib.MAlpha)
        panel = self._core.CardMaker("start-lights-housing")
        panel.setFrame(-0.68, 0.68, -0.18, 0.18)
        housing = self.root.attachNewNode(panel.generate())
        housing.setColor(0.018, 0.023, 0.032, 0.97)
        housing.setBin("fixed", 600)
        self._lamps: list[Any] = []
        self._glows: list[Any] = []
        for index in range(START_LIGHT_COUNT):
            x = (index - 2) * 0.245
            self._disc(f"socket-{index}", x, 0.023, 0.106, 601).setColor(0.12, 0.14, 0.17, 1.0)
            self._glows.append(self._disc(f"glow-{index}", x, 0.023, 0.100, 602))
            self._lamps.append(self._disc(f"lamp-{index}", x, 0.023, 0.083, 603))
        label_node = self._core.TextNode("start-lights-label")
        label_node.setAlign(self._core.TextNode.ACenter)
        label_node.setTextColor(0.84, 0.87, 0.92, 1.0)
        self._label = self.root.attachNewNode(label_node)
        self._label.setPos(*self._position(0.0, -0.137))
        self._label.setScale(0.035)
        self._label.setBin("fixed", 604)
        self.update(RaceStartSequence())

    def update(self, sequence: RaceStartSequence, *, left_edge: float | None = None, center_y: float = 0.74) -> None:
        """Light left to right, extinguish together, then remove the housing."""
        if not sequence.visible:
            self.root.hide()
            return
        self.root.show()
        aspect = float(self._base.win.getXSize()) / max(1, int(self._base.win.getYSize()))
        left = max(-aspect + 0.04, left_edge if left_edge is not None else -aspect + 0.04)
        right = aspect - 0.04
        scale = min(1.0, max(0.1, (right - left) / 1.36))
        center = max(left + 0.68 * scale, min(0.0, right - 0.68 * scale))
        self.root.setScale(scale)
        self.root.setPos(*self._position(center, center_y))
        for index, (lamp, glow) in enumerate(zip(self._lamps, self._glows, strict=True)):
            lit = index < sequence.light_count
            lamp.setColor(*((1.0, 0.035, 0.012, 1.0) if lit else (0.12, 0.025, 0.025, 1.0)))
            glow.setColor(1.0, 0.015, 0.005, 0.32 if lit else 0.0)
        self._label.node().setText("LIGHTS OUT" if sequence.racing else "GET READY")

    def _position(self, x: float, y: float) -> tuple[float, float, float]:
        return (x, y, 0.0) if self._y_up else (x, 0.0, y)

    def _disc(self, name: str, x: float, y: float, radius: float, order: int) -> Any:
        core = self._core
        data = core.GeomVertexData(name, core.GeomVertexFormat.getV3(), core.Geom.UHStatic)
        vertices = core.GeomVertexWriter(data, "vertex")
        vertices.addData3f(*self._position(x, y))
        segments = 64
        for i in range(segments):
            angle = 2.0 * pi * i / segments
            vertices.addData3f(*self._position(x + radius * cos(angle), y + radius * sin(angle)))
        triangles = core.GeomTriangles(core.Geom.UHStatic)
        for i in range(segments):
            triangles.addVertices(0, i + 1, (i + 1) % segments + 1)
        geometry = core.Geom(data)
        geometry.addPrimitive(triangles)
        node = core.GeomNode(name)
        node.addGeom(geometry)
        disc = self.root.attachNewNode(node)
        disc.setTwoSided(True)
        disc.setBin("fixed", order)
        return disc
