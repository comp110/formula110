from __future__ import annotations

import pytest

from racing.graphics.colors import ColorRGBA, starting_grid_colors


def test_duplicate_colors_darken_by_starting_grid_places_behind_each_color_leader() -> None:
    yellow: ColorRGBA = (1.0, 0.8, 0.1, 1.0)
    blue: ColorRGBA = (0.2, 0.4, 0.8, 0.6)
    colors = starting_grid_colors((("y1", yellow), ("b1", blue), ("y2", yellow), ("y3", yellow), ("b2", blue)))
    assert colors["y1"] == yellow
    assert colors["b1"] == blue
    assert colors["y2"] == pytest.approx((0.8, 0.64, 0.08, 1.0))
    assert colors["y3"] == pytest.approx((0.7, 0.56, 0.07, 1.0))
    assert colors["b2"] == pytest.approx((0.14, 0.28, 0.56, 0.6))


def test_new_race_assigns_shades_from_its_grid_without_compounding_darkness() -> None:
    color: ColorRGBA = (0.4, 0.6, 0.8, 1.0)
    first = starting_grid_colors((("a", color), ("b", color), ("c", color)))
    next_race = starting_grid_colors((("b", color), ("c", color), ("a", color)))
    assert first["a"] == next_race["b"] == color
    assert first["b"] == next_race["c"] == pytest.approx((0.36, 0.54, 0.72, 1.0))
    assert first["c"] == next_race["a"] == pytest.approx((0.32, 0.48, 0.64, 1.0))
    assert starting_grid_colors((("a", color), ("b", color), ("c", color))) == first


def test_large_grids_do_not_produce_negative_or_completely_black_colors() -> None:
    color: ColorRGBA = (1.0, 0.5, 0.3, 1.0)
    colors = starting_grid_colors(tuple((str(index), color) for index in range(15)))
    assert colors["14"] == pytest.approx((0.1, 0.05, 0.03, 1.0))
