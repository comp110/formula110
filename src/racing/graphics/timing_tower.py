"""A compact, clickable race timing tower anchored to Panda's 2D viewport."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from math import ceil, isfinite
from pathlib import Path
from typing import Any, cast

from racing.graphics.colors import ColorRGBA

TOWER_WIDTH = 0.43
TOWER_HEADER_HEIGHT = 0.14
TOWER_CLOCK_HEIGHT = 0.085
TOWER_ROW_HEIGHT = 0.084
TOWER_NAME_X = 0.102
TOWER_NAME_WIDTH = 0.15
GRID_LOGO_WIDTH = 1.16
RESULTS_FADE_SECONDS = 1.5
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
    finished: bool = False
    dnf: bool = False


def format_timing_gap(row: TimingTowerRow) -> str:
    """Keep an unavailable gap distinct from the leader and eliminated cars."""
    if row.dnf:
        return "DNF"
    if row.finished and row.rank == 1:
        return "FINISH"
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
        self._grid_root = self._base.aspect2d.attachNewNode("starting-grid")
        self._grid_root.setTransparency(self._core.TransparencyAttrib.MAlpha)
        self._results_elapsed: float | None = None
        self._grid_background = self._card(
            self._grid_root, "grid-backdrop", -1.0, 1.0, 2.0, 2.0, (0.020, 0.028, 0.042, 1.0), bin_order=700,
        )
        self._grid_content = self._grid_root.attachNewNode("starting-grid-content")
        self._grid_branding = self._grid_root.attachNewNode("starting-grid-branding")
        self._grid_content_width = 1.34
        self._grid_content_height = 1.47
        self._grid_root.hide()
        self._toggle = self._button(self._root, TOWER_WIDTH, TOWER_HEADER_HEIGHT, self._toggle_visible)
        self._card(
            self._toggle, "header", 0.0, 0.0, TOWER_WIDTH, TOWER_HEADER_HEIGHT, (1.0, 1.0, 1.0, 1.0)
        )
        self._add_logo(logo_path)

        self._card(self._panel, "clock-background", 0.0, -TOWER_HEADER_HEIGHT, TOWER_WIDTH, TOWER_CLOCK_HEIGHT, _DARK)
        self._auto = self._button(self._panel, TOWER_WIDTH, TOWER_CLOCK_HEIGHT, lambda: self._on_select(None))
        self._auto.setPos(*self._position(0.0, -TOWER_HEADER_HEIGHT))
        self._clock = self._text(self._auto, "00:00", TOWER_WIDTH / 2, -0.056, 0.042, _WHITE, align="center")
        self._update_layout()

    @property
    def visible(self) -> bool:
        return self._visible

    @property
    def right_edge(self) -> float:
        """Right edge in aspect2d coordinates, for neighboring HUD elements."""
        return float(self._root.getX() + TOWER_WIDTH * self._root.getSx())

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Visible HUD rectangle in aspect2d coordinates: left, bottom, right, top."""
        left = float(self._root.getX())
        top = float(self._root.getY() if self._y_up else self._root.getZ())
        scale = float(self._root.getSx())
        if not self._visible:
            return left, top - (0.22 + TOWER_HEADER_HEIGHT) * scale, self.right_edge, top - 0.22 * scale
        height = TOWER_HEADER_HEIGHT + TOWER_CLOCK_HEIGHT + self._row_count * TOWER_ROW_HEIGHT
        return left, top - height * scale, self.right_edge, top

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
        """Collapse the rows while keeping the clickable logo available."""
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

    def show_starting_grid(
        self, rows: tuple[TimingTowerRow, ...], participant_names: dict[str, str],
        race_title: str = "STARTING GRID",
    ) -> None:
        """Replace the compact tower with a full-screen, input-gated lineup."""
        self._results_elapsed = None
        self._show_race_table(rows, participant_names, race_title, "STARTING GRID")
        self._grid_root.setColorScale(1, 1, 1, 1)
        self._root.hide()

    def show_results(
        self, rows: tuple[TimingTowerRow, ...], participant_names: dict[str, str],
        race_title: str, details: dict[str, str], *, heading: str = "FINAL RESULTS",
    ) -> None:
        """Fade the title-screen layout in over the final race scene."""
        self._show_race_table(rows, participant_names, race_title, heading, details)
        self._results_elapsed = 0.0
        self._grid_root.setColorScale(1, 1, 1, 0)
        # Keep the tower visible beneath the fade, but prevent invisible clicks.
        for button in (self._toggle, self._auto, *self.row_buttons.values()):
            button["state"] = "disabled"

    def advance_results(self, delta_seconds: float) -> None:
        if self._results_elapsed is None:
            return
        self._results_elapsed = min(RESULTS_FADE_SECONDS, self._results_elapsed + max(0.0, delta_seconds))
        progress = self._results_elapsed / RESULTS_FADE_SECONDS
        opacity = progress * progress * (3.0 - 2.0 * progress)
        self._grid_root.setColorScale(1, 1, 1, opacity)
        if progress >= 1.0:
            self._root.hide()
        self._update_layout()

    def _show_race_table(
        self, rows: tuple[TimingTowerRow, ...], participant_names: dict[str, str],
        race_title: str, heading: str, details: dict[str, str] | None = None,
    ) -> None:
        self._grid_content.getChildren().detach()
        self._grid_branding.getChildren().detach()

        def text(
            value: str, x: float, y: float, scale: float, color: ColorRGBA = _WHITE, *, align: str = "left",
        ) -> Any:
            node = self._text(self._grid_content, value, x, y, scale, color, align=align, bin_order=720)
            return node

        if self._logo_texture is not None:
            width = GRID_LOGO_WIDTH
            height = width * int(self._logo_texture.getYSize()) / max(1, int(self._logo_texture.getXSize()))
            logo = self._card(
                self._grid_branding, "grid-f110-logo", -width / 2, height / 2, width, height,
                (1.0, 1.0, 1.0, 1.0), bin_order=720,
            )
            logo.setTexture(self._logo_texture, 1)
        else:
            self._text(self._grid_branding, "F110", 0.0, -0.08, 0.24, _BLUE, align="center", bin_order=720)
        metrics = self._core.TextNode("starting-grid-metrics")
        if self._font is not None:
            metrics.setFont(self._font)
        name_width = min(1.35, max(
            0.65,
            max((float(metrics.calcWidth(participant_names.get(row.car_id, row.name))) * 0.047 for row in rows),
                default=0.0),
            max((float(metrics.calcWidth(detail)) * 0.032 for detail in (details or {}).values()), default=0.0),
        ))
        self._grid_content_width = 0.63 + name_width + 0.05
        row_height = min(0.22, 1.45 / max(1, len(rows))) if details is not None else min(
            0.19, 1.12 / max(1, len(rows)),
        )
        self._grid_content_height = 0.35 + len(rows) * row_height
        title = text(race_title, 0.04, -0.06, 0.08)
        title.setScale(min(
            0.08, (self._grid_content_width - 0.08) / max(1.0, float(title.node().calcWidth(race_title))),
        ))
        text(heading, 0.04, -0.18, 0.043, _BLUE)
        text("POS", 0.04, -0.30, 0.027, _MUTED)
        text("CAR", 0.43, -0.30, 0.027, _MUTED)
        text("PARTICIPANTS", 0.63, -0.30, 0.027, _MUTED)
        for index, row in enumerate(rows):
            top = -0.35 - index * row_height
            self._card(
                self._grid_content, "grid-row", 0.0, top, self._grid_content_width, row_height - 0.008,
                _DARK if index % 2 == 0 else (0.055, 0.066, 0.082, 1.0), bin_order=710,
            )
            baseline = top - row_height / 2.0 - 0.016
            text("—" if row.dnf else f"{row.rank:02}", 0.05, baseline, 0.048, _MUTED)
            self._card(
                self._grid_content, "grid-car-color", 0.25, baseline + 0.041, 0.105, 0.046,
                row.color, bin_order=715,
            )
            car = text(row.name, 0.43, baseline, 0.035, _MUTED)
            car.setScale(min(0.035, 0.15 / max(1.0, float(car.node().calcWidth(row.name)))))
            name = text(
                participant_names.get(row.car_id, row.name), 0.63,
                baseline + (0.026 if details is not None else 0.0), 0.047,
            )
            name.setScale(min(0.047, name_width / max(1.0, float(name.node().calcWidth(name.node().getText())))))
            if details is not None:
                detail = text(details.get(row.car_id, "—"), 0.63, baseline - 0.028, 0.032, _BLUE)
                detail.setScale(min(
                    0.032, name_width / max(1.0, float(detail.node().calcWidth(detail.node().getText()))),
                ))
        self._grid_root.show()
        self._layout_key = None
        self._update_layout()

    def hide_starting_grid(self) -> None:
        self._results_elapsed = None
        self._grid_root.hide()
        self._root.show()
        for button in (self._toggle, self._auto, *self.row_buttons.values()):
            button["state"] = "normal"

    def update(
        self,
        rows: tuple[TimingTowerRow, ...],
        remaining_seconds: float,
        selected_car_id: str | None,
        clock_text: str | None = None,
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
            self._set_text(widgets.rank, "—" if row.dnf else str(row.rank))
            self._set_text(widgets.name, self._fit_name(row.name, widgets.name.node(), TOWER_NAME_WIDTH, 0.034))
            gap_text = format_timing_gap(row)
            self._set_text(widgets.gap, gap_text)
            gap_width = TOWER_WIDTH - 0.026 - TOWER_NAME_X - TOWER_NAME_WIDTH - 0.018
            widgets.gap.setScale(min(0.032, gap_width / max(1.0, float(widgets.gap.node().calcWidth(gap_text)))))
            color = _MUTED if row.eliminated else _WHITE
            widgets.rank.node().setTextColor(*color)
            widgets.name.node().setTextColor(*color)
            widgets.gap.node().setTextColor(*(_BLUE if selected and not row.eliminated else color))
        self._row_count = len(rows)
        self._set_text(self._clock, clock_text if clock_text is not None else format_race_countdown(remaining_seconds))
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
        self._grid_root.removeNode()
        self._destroyed = True

    def _create_row(self, car_id: str) -> _RowWidgets:
        button = self._button(self._panel, TOWER_WIDTH, TOWER_ROW_HEIGHT, lambda: self._on_select(car_id))
        background = self._card(button, "row-background", 0.0, 0.0, TOWER_WIDTH, TOWER_ROW_HEIGHT, _DARK)
        stripe = self._card(button, "team-stripe", 0.075, -0.019, 0.008, 0.046, _BLUE, bin_order=510)
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
        rank = self._text(button, "", 0.035, -0.055, 0.036, _WHITE, align="center")
        name = self._text(button, "", TOWER_NAME_X, -0.053, 0.034, _WHITE)
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
        self._logo_texture = texture
        if texture is None:
            self._text(self._toggle, "F110", TOWER_WIDTH / 2, -0.102, 0.12, _BLUE, align="center")
            return
        width = min(
            TOWER_WIDTH - 0.024,
            (TOWER_HEADER_HEIGHT - 0.024) * int(texture.getXSize()) / max(1, int(texture.getYSize())),
        )
        height = width * int(texture.getYSize()) / max(1, int(texture.getXSize()))
        logo = self._card(
            self._toggle,
            "f110-logo",
            (TOWER_WIDTH - width) / 2,
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
        bin_order: int = 520,
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
        self._set_hud_render_state(path, bin_order)
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
        self._grid_background.setSx(aspect)
        logo_width = min(GRID_LOGO_WIDTH, aspect * 0.8)
        table_width = 0.95 * aspect
        grid_scale = min(1.8 / self._grid_content_height, table_width / self._grid_content_width)
        self._grid_content.setScale(grid_scale)
        self._grid_content.setPos(*self._position(0.0, self._grid_content_height * grid_scale / 2))
        self._grid_branding.setScale(logo_width / GRID_LOGO_WIDTH)
        self._grid_branding.setPos(*self._position(-aspect / 2, 0.0))
        content_height = TOWER_HEADER_HEIGHT + TOWER_CLOCK_HEIGHT + self._row_count * TOWER_ROW_HEIGHT
        scale = min(1.0, (2 * aspect - 0.11) / TOWER_WIDTH, 1.86 / content_height if self._visible else 1.0)
        self._root.setPos(*self._position(-aspect + 0.055, 0.95))
        self._root.setScale(max(0.1, scale))
        self._toggle.setPos(*self._position(0.0, 0.0 if self._visible else -0.22))

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
