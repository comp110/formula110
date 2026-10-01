"""Shared formula car appearance values."""

from __future__ import annotations

ColorRGBA = tuple[float, float, float, float]

UNC_CAROLINA_BLUE: ColorRGBA = (75 / 255, 156 / 255, 211 / 255, 1.0)
UNC_FORDHAM_FOUNTAIN: ColorRGBA = (183 / 255, 215 / 255, 237 / 255, 1.0)
UNC_BOLIN_CREEK: ColorRGBA = (44 / 255, 80 / 255, 128 / 255, 1.0)
UNC_CORNERSTONE: ColorRGBA = (207 / 255, 211 / 255, 213 / 255, 1.0)

DEFAULT_FORMULA_TEAM_COLOR: ColorRGBA = UNC_FORDHAM_FOUNTAIN
DEFAULT_CHALLENGER_TEAM_COLOR: ColorRGBA = UNC_FORDHAM_FOUNTAIN
DEFAULT_INCUMBENT_TEAM_COLOR: ColorRGBA = UNC_CAROLINA_BLUE

# Historical public name kept for callers that imported the original default.
FORMULA_TEAM_RED: ColorRGBA = DEFAULT_FORMULA_TEAM_COLOR


def starting_grid_colors(grid_cars: tuple[tuple[str, ColorRGBA], ...]) -> dict[str, ColorRGBA]:
    """Assign duplicate-color shades once, ordered by starting grid position."""
    leading_ranks: dict[tuple[float, float, float], int] = {}
    shaded: dict[str, ColorRGBA] = {}
    for rank, (car_id, color) in enumerate(grid_cars, start=1):
        rgb = (color[0], color[1], color[2])
        lead_rank = leading_ranks.setdefault(rgb, rank)
        brightness = max(0.1, 1.0 - 0.1 * (rank - lead_rank))
        shaded[car_id] = (color[0] * brightness, color[1] * brightness, color[2] * brightness, color[3])
    return shaded
