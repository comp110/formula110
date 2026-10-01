from __future__ import annotations

from dataclasses import replace

import pytest

from racing.graphics.position_markers import (
    CARD_HEIGHT,
    CARD_TOP,
    CARD_WIDTH,
    POSITION_WIDTH,
    PositionTarget,
    leader_card_layout,
    marker_text_color,
)


def _target(rank: int, *, position: tuple[float, float] = (0.2, -0.2)) -> PositionTarget:
    return PositionTarget(
        f"car-{rank}", rank, (1, 0.78, 0.12, 1), f"Driver {rank}", position, (position[0], position[1] + 0.04)
    )


@pytest.mark.parametrize("aspect", [16 / 9, 4 / 3])
@pytest.mark.parametrize("tower_visible", [True, False])
def test_top_row_clears_hud_and_keeps_all_three_cards_inside_viewport(aspect: float, tower_visible: bool) -> None:
    hud = (-aspect + 0.055, 0.1, -aspect + (0.595 if tower_visible else 0.98), 0.97)
    targets = tuple(_target(rank, position=(0.5, -0.3)) for rank in range(1, 4))
    cards = leader_card_layout(targets, aspect, (hud,))
    assert len(cards) == 3
    previous_right = hud[2]
    for target in targets:
        card = cards[target.car_id]
        left, bottom, right, top = card.bounds
        assert -aspect < left < right < aspect
        assert left > previous_right
        assert top == pytest.approx(CARD_TOP)
        assert bottom >= CARD_TOP - CARD_HEIGHT - 1e-9
        assert right - left <= CARD_WIDTH + 1e-9
        assert card.pointer[0] == pytest.approx((left, bottom))
        assert card.pointer[1] == pytest.approx((left + POSITION_WIDTH * card.scale, bottom))
        assert card.pointer[2] == target.pointer_position
        previous_right = right


def test_cards_hold_rank_slots_while_pointers_track_cars_and_other_leaders_leave_view() -> None:
    targets = tuple(_target(rank) for rank in range(1, 4))
    original = leader_card_layout(targets, 16 / 9, ())
    moved = replace(targets[1], position=(-0.5, -0.6), pointer_position=(-0.5, -0.56))
    only_second = leader_card_layout((moved,), 16 / 9, ())
    assert only_second[moved.car_id].center == original[moved.car_id].center
    assert only_second[moved.car_id].pointer[:2] == original[moved.car_id].pointer[:2]
    assert only_second[moved.car_id].pointer[2] == moved.pointer_position
    assert only_second[moved.car_id].pointer[2] != original[moved.car_id].pointer[2]


def test_overtake_reassigns_rank_slots_and_drops_fourth_place_immediately() -> None:
    targets = tuple(_target(rank) for rank in range(1, 5))
    original = leader_card_layout(targets, 16 / 9, ())
    ranks = (2, 3, 4, 1)
    updated = leader_card_layout(
        tuple(replace(target, rank=rank) for target, rank in zip(targets, ranks, strict=True)), 16 / 9, ()
    )
    assert set(updated) == {"car-1", "car-2", "car-4"}
    assert updated["car-4"].center == original["car-1"].center
    assert updated["car-1"].center == original["car-2"].center
    assert leader_card_layout((), 16 / 9, ()) == {}


@pytest.mark.parametrize("position", [(-2, 0), (2, 0), (0, -1.1), (0, 1.1), (0, 0.9)])
def test_offscreen_cars_and_cars_behind_top_cards_do_not_get_pointers(position: tuple[float, float]) -> None:
    assert leader_card_layout((_target(1, position=position),), 16 / 9, ()) == {}


def test_car_hidden_by_timing_tower_does_not_get_a_card() -> None:
    assert leader_card_layout((_target(1, position=(-1.5, 0.5)),), 16 / 9, ((-1.7, 0.0, -1.1, 0.95),)) == {}


def test_obstructed_top_row_hides_cards_instead_of_overlapping_hud() -> None:
    assert leader_card_layout((_target(1),), 16 / 9, ((-2, 0.7, 2, 1),)) == {}


@pytest.mark.parametrize("aspect", [16 / 9, 4 / 3])
@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("rank", [1, 3, 4, 10])
def test_split_card_shows_only_its_focus_at_any_rank_inside_its_pane(aspect: float, side: str, rank: int) -> None:
    viewport = (-aspect, -1.0, 0.0, 1.0) if side == "left" else (0.0, -1.0, aspect, 1.0)
    x = (viewport[0] + viewport[2]) / 2
    targets = tuple(_target(place, position=(x, -0.6)) for place in range(1, 11))
    focus = targets[rank - 1]
    tower = (-aspect + 0.055, 0.1, -aspect + 0.595, 0.97)
    cards = leader_card_layout(targets, aspect, (tower,), viewport=viewport, focused_car_id=focus.car_id)
    assert set(cards) == {focus.car_id}
    card = cards[focus.car_id]
    left, bottom, right, top = card.bounds
    assert viewport[0] < left < right < viewport[2]
    assert left > tower[2]
    assert top == pytest.approx(CARD_TOP)
    assert card.scale == 1.0
    assert card.pointer[0] == pytest.approx((left, bottom))
    assert card.pointer[1] == pytest.approx((left + POSITION_WIDTH, bottom))
    assert card.pointer[2] == focus.pointer_position
    assert all(viewport[0] <= px <= viewport[2] and viewport[1] <= py <= viewport[3] for px, py in card.pointer)


def test_split_focus_keeps_same_card_slot_after_position_changes() -> None:
    target = _target(10, position=(0.8, -0.5))
    viewport = (0.0, -1.0, 16 / 9, 1.0)
    original = leader_card_layout((target,), 16 / 9, (), viewport=viewport, focused_car_id=target.car_id)
    updated = leader_card_layout(
        (replace(target, rank=2),), 16 / 9, (), viewport=viewport, focused_car_id=target.car_id,
    )
    assert updated == original


@pytest.mark.parametrize("position", [(-0.1, -0.3), (1.8, -0.3), (0.5, -1.2), (0.5, 0.95)])
def test_split_focus_outside_its_pane_or_under_card_is_hidden(position: tuple[float, float]) -> None:
    target = _target(10, position=position)
    assert leader_card_layout(
        (target,), 16 / 9, (), viewport=(0.0, -1.0, 16 / 9, 1.0), focused_car_id=target.car_id,
    ) == {}


@pytest.mark.parametrize("color", [(1.0, 0.78, 0.12, 1.0), (0.93, 0.93, 0.96, 1.0), (0.22, 0.78, 0.40, 1.0)])
def test_light_car_colors_use_dark_rank_text(color: tuple[float, float, float, float]) -> None:
    assert marker_text_color(color) == (0, 0, 0, 1)


def test_dark_car_colors_use_white_rank_text() -> None:
    assert marker_text_color((0.02, 0.04, 0.08, 1)) == (1, 1, 1, 1)
