"""A compact, clickable race timing tower anchored to Panda's 2D viewport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import ceil, isfinite
from pathlib import Path
from typing import Any, cast

from racing.graphics.colors import ColorRGBA

TOWER_WIDTH = 0.76
TOWER_HEADER_HEIGHT = 0.14
TOWER_CLOCK_HEIGHT = 0.085
TOWER_ROW_HEIGHT = 0.084
_WHITE: ColorRGBA = (0.96, 0.97, 0.99, 1.0)
_MUTED: ColorRGBA = (0.61, 0.66, 0.73, 1.0)
_BLUE: ColorRGBA = (75 / 255, 156 / 255, 211 / 255, 1.0)
_DARK: ColorRGBA = (0.035, 0.045, 0.060, 0.97)
_SELECTED: ColorRGBA = (0.10, 0.22, 0.32, 0.99)


@dataclass(frozen=True, slots=True)
class TimingTowerRow:
    car_id: str
    name: str
    color: ColorRGBA
    rank: int
    gap_seconds: float | None
    eliminated: bool = False


def format_timing_gap(row: TimingTowerRow) -> str:
    """Keep an unavailable gap distinct from the leader and eliminated cars."""
    if row.eliminated:
        return "OUT"
    if row.rank == 1:
        return "LEADER"
    if row.gap_seconds is None or not isfinite(row.gap_seconds):
        return "—"
    return f"+{max(0.0, row.gap_seconds):.3f}"


def format_race_countdown(remaining_seconds: float) -> str:
    """Round up the live countdown so a newly started second remains visible."""
    seconds = max(0, ceil(remaining_seconds)) if isfinite(remaining_seconds) else 0
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"


@dataclass(slots=True)
class _RowWidgets:
    button: Any
    background: Any
    stripe: Any
    rank: Any
    name: Any
    gap: Any


class TimingTower:
    """Display live standings and select a car by its stable runtime ID.

    Panda DirectButton owns pointer events; native TextNodes and cards use the
    application's coordinate system, including Formula110's Y-up viewport.
    """

    def __init__(
        self,
        ursina: Any,
        on_select: Callable[[str | None], None],
        logo_path: Path,
        *,
        on_toggle: Callable[[bool], None] | None = None,
    ) -> None:
        self._base = ursina.application.base
        self._core = cast(Any, import_module("panda3d.core"))
        self._button_type = cast(Any, import_module("direct.gui.DirectButton")).DirectButton
        self._on_select = on_select
        self._on_toggle = on_toggle
        font_root = Path(str(getattr(ursina, "__file__", ""))).parent / "fonts"
        font_path = font_root / "OpenSans-Regular.ttf"
        self._font = self._base.loader.loadFont(str(font_path)) if font_path.is_file() else None
        self._visible = True
        self._destroyed = False
        self._rows: dict[str, _RowWidgets] = {}
        self._row_count = 0
        self._layout_key: tuple[int, int, int, bool] | None = None
        self._y_up = self._core.getDefaultCoordinateSystem() in (self._core.CSYupRight, self._core.CSYupLeft)
        self._root = self._base.aspect2d.attachNewNode("timing-tower")
        self._set_hud_render_state(self._root, 500)
        self._panel = self._root.attachNewNode("timing-tower-panel")
        self._card(self._panel, "header", 0.0, 0.0, TOWER_WIDTH, TOWER_HEADER_HEIGHT, (1.0, 1.0, 1.0, 1.0))
        self._add_logo(logo_path)

        self._toggle = self._button(self._root, 0.245, 0.066, self._toggle_visible)
        self._toggle_background = self._card(
            self._toggle, "toggle-background", 0.0, 0.0, 0.245, 0.066, (1.0, 1.0, 1.0, 1.0)
        )
        self._text(self._toggle, "Timing [L]", 0.1225, -0.044, 0.031, (0.14, 0.22, 0.29, 1.0), align="center")

        self._card(self._panel, "clock-background", 0.0, -TOWER_HEADER_HEIGHT, TOWER_WIDTH, TOWER_CLOCK_HEIGHT, _DARK)
        self._clock = self._text(self._panel, "00:00", 0.034, -0.196, 0.045, _WHITE)
        self._race_label = self._text(self._panel, "RACE 1/1", 0.382, -0.194, 0.027, _MUTED, align="center")
        self._auto = self._button(self._panel, 0.145, TOWER_CLOCK_HEIGHT, lambda: self._on_select(None))
        self._auto.setPos(*self._position(TOWER_WIDTH - 0.16, -TOWER_HEADER_HEIGHT))
        self._auto_text = self._text(self._auto, "AUTO", 0.12, -0.054, 0.028, _BLUE, align="right")
        self._update_layout()

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def right_edge(self) -> float:
        """Right edge in aspect2d coordinates, for neighboring HUD elements."""
        return float(self._root.getX() + TOWER_WIDTH * self._root.getSx())

    @property
    def row_buttons(self) -> dict[str, Any]:
        """Expose stable car IDs and their GUI controls for interaction checks."""
        return {car_id: widgets.button for car_id, widgets in self._rows.items()}

    @property
    def toggle_button(self) -> Any:
        return self._toggle

    @property
    def auto_button(self) -> Any:
        return self._auto

    def set_visible(self, visible: bool) -> None:
        """Collapse the rows while keeping the pointer toggle available."""
        if self._destroyed or self._visible == visible:
            return
        self._visible = visible
        if visible:
            self._panel.show()
        else:
            self._panel.hide()
        self._update_layout()
        if self._on_toggle is not None:
            self._on_toggle(visible)

    def update(
        self,
        rows: tuple[TimingTowerRow, ...],
        remaining_seconds: float,
        race_index: int,
        race_count: int,
        selected_car_id: str | None,
    ) -> None:
        """Refresh row order without changing what an existing row click selects."""
        if self._destroyed:
            return
        current_ids = {row.car_id for row in rows}
        for car_id in tuple(self._rows):
            if car_id not in current_ids:
                self._rows.pop(car_id).button.destroy()
        for index, row in enumerate(rows):
            widgets = self._rows.get(row.car_id)
            if widgets is None:
                widgets = self._create_row(row.car_id)
                self._rows[row.car_id] = widgets
            widgets.button.setPos(
                *self._position(0.0, -TOWER_HEADER_HEIGHT - TOWER_CLOCK_HEIGHT - index * TOWER_ROW_HEIGHT)
            )
            selected = row.car_id == selected_car_id
            background = _SELECTED if selected else (_DARK if index % 2 == 0 else (0.055, 0.066, 0.082, 0.97))
            widgets.background.setColor(*background)
            widgets.stripe.setColor(*row.color)
            self._set_text(widgets.rank, str(row.rank))
            self._set_text(widgets.name, self._fit_name(row.name, widgets.name.node(), 0.39, 0.034))
            self._set_text(widgets.gap, format_timing_gap(row))
            color = _MUTED if row.eliminated else _WHITE
            widgets.rank.node().setTextColor(*color)
            widgets.name.node().setTextColor(*color)
            widgets.gap.node().setTextColor(*(_BLUE if selected and not row.eliminated else color))
        self._row_count = len(rows)
        self._set_text(self._clock, format_race_countdown(remaining_seconds))
        self._set_text(self._race_label, f"RACE {race_index}/{race_count}")
        self._auto_text.node().setTextColor(*(_BLUE if selected_car_id is None else _MUTED))
        self._update_layout()

    def destroy(self) -> None:
        """Remove GUI event registrations as well as the visible scene nodes."""
        if self._destroyed:
            return
        for widgets in self._rows.values():
            widgets.button.destroy()
        self._rows.clear()
        self._auto.destroy()
        self._toggle.destroy()
        self._root.removeNode()
        self._destroyed = True

    def _create_row(self, car_id: str) -> _RowWidgets:
        button = self._button(self._panel, TOWER_WIDTH, TOWER_ROW_HEIGHT, lambda: self._on_select(car_id))
        background = self._card(button, "row-background", 0.0, 0.0, TOWER_WIDTH, TOWER_ROW_HEIGHT, _DARK)
        stripe = self._card(button, "team-stripe", 0.105, -0.019, 0.008, 0.046, _BLUE, bin_order=510)
        self._card(
            button,
            "row-divider",
            0.0,
            -TOWER_ROW_HEIGHT + 0.002,
            TOWER_WIDTH,
            0.002,
            (0.2, 0.23, 0.28, 0.5),
            bin_order=510,
        )
        rank = self._text(button, "", 0.048, -0.055, 0.036, _WHITE, align="center")
        name = self._text(button, "", 0.135, -0.053, 0.034, _WHITE)
        gap = self._text(button, "", TOWER_WIDTH - 0.026, -0.053, 0.032, _WHITE, align="right")
        return _RowWidgets(button, background, stripe, rank, name, gap)

    def _button(self, parent: Any, width: float, height: float, command: Callable[[], None]) -> Any:
        button = self._button_type(
            parent=parent,
            frameSize=(0.0, width, -height, 0.0),
            relief=None,
            pressEffect=False,
            command=command,
            rolloverSound=None,
            clickSound=None,
            sortOrder=1000,
        )
        self._set_hud_render_state(button, 500)
        return button

    def _add_logo(self, path: Path) -> None:
        texture = self._base.loader.loadTexture(str(path.resolve()))
        if texture is None:
            self._text(self._panel, "F110", 0.035, -0.095, 0.095, _BLUE)
            return
        width = 0.365
        height = width * int(texture.getYSize()) / max(1, int(texture.getXSize()))
        logo = self._card(
            self._panel,
            "f110-logo",
            0.034,
            -(TOWER_HEADER_HEIGHT - height) / 2,
            width,
            height,
            (1, 1, 1, 1),
            bin_order=510,
        )
        logo.setTexture(texture, 1)

    def _card(
        self,
        parent: Any,
        name: str,
        left: float,
        top: float,
        width: float,
        height: float,
        color: ColorRGBA,
        *,
        bin_order: int = 500,
    ) -> Any:
        maker = self._core.CardMaker(f"timing-tower-{name}")
        maker.setFrame(
            self._core.Point3(*self._position(left, top - height)),
            self._core.Point3(*self._position(left + width, top - height)),
            self._core.Point3(*self._position(left + width, top)),
            self._core.Point3(*self._position(left, top)),
        )
        card = parent.attachNewNode(maker.generate())
        card.setColor(*color)
        card.setTransparency(self._core.TransparencyAttrib.MAlpha)
        card.setTwoSided(True)
        self._set_hud_render_state(card, bin_order)
        return card

    def _text(
        self,
        parent: Any,
        value: str,
        x: float,
        y: float,
        scale: float,
        color: ColorRGBA,
        *,
        align: str = "left",
    ) -> Any:
        node = self._core.TextNode("timing-tower-text")
        if self._font is not None:
            node.setFont(self._font)
        node.setText(value)
        node.setTextColor(*color)
        node.setAlign(
            {
                "left": self._core.TextNode.ALeft,
                "center": self._core.TextNode.ACenter,
                "right": self._core.TextNode.ARight,
            }[align]
        )
        path = parent.attachNewNode(node)
        path.setPos(*self._position(x, y))
        path.setScale(scale)
        self._set_hud_render_state(path, 520)
        return path

    @staticmethod
    def _set_hud_render_state(path: Any, bin_order: int) -> None:
        # PGItem and generated TextNode geometry may carry their own render
        # attributes. These overlays must never read/write scene depth, and
        # logo/text must draw after their opaque backgrounds in every view.
        path.setDepthTest(False, 100)
        path.setDepthWrite(False, 100)
        path.setLightOff(100)
        path.setShaderOff(100)
        path.setBin("fixed", bin_order, 100)

    def _position(self, x: float, y: float) -> tuple[float, float, float]:
        return (x, y, 0.0) if self._y_up else (x, 0.0, y)

    def _toggle_visible(self) -> None:
        self.set_visible(not self._visible)

    def _update_layout(self) -> None:
        width = int(self._base.win.getXSize())
        height = int(self._base.win.getYSize())
        if width <= 0 or height <= 0:
            return
        key = (width, height, self._row_count, self._visible)
        if key == self._layout_key:
            return
        self._layout_key = key
        aspect = width / height
        content_height = TOWER_HEADER_HEIGHT + TOWER_CLOCK_HEIGHT + self._row_count * TOWER_ROW_HEIGHT
        scale = min(1.0, (2 * aspect - 0.11) / TOWER_WIDTH, 1.86 / content_height if self._visible else 1.0)
        self._root.setPos(*self._position(-aspect + 0.055, 0.95))
        self._root.setScale(max(0.1, scale))
        self._toggle.setPos(
            *self._position(TOWER_WIDTH - 0.26, -0.037) if self._visible else self._position(0.0, -0.22)
        )

    @staticmethod
    def _set_text(path: Any, value: str) -> None:
        if path.node().getText() != value:
            path.node().setText(value)

    @staticmethod
    def _fit_name(name: str, node: Any, width: float, scale: float) -> str:
        value = " ".join(name.split())
        if node.calcWidth(value) * scale <= width:
            return value
        while value and node.calcWidth(value + "…") * scale > width:
            value = value[:-1]
        return value.rstrip() + "…"
