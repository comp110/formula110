from __future__ import annotations

from itertools import combinations
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from racing.game import app
from racing.game.app import (
    HeadToHeadCarLabel,
    _head_to_head_car_label,  # pyright: ignore[reportPrivateUsage]
    _head_to_head_car_paint_color,  # pyright: ignore[reportPrivateUsage]
    _head_to_head_viewer_controller,  # pyright: ignore[reportPrivateUsage]
    _head_to_head_viewer_controllers,  # pyright: ignore[reportPrivateUsage]
    _head_to_head_viewer_keyboard_controlled,  # pyright: ignore[reportPrivateUsage]
    _race_viewer_entries,  # pyright: ignore[reportPrivateUsage]
    _update_head_to_head_car_labels,  # pyright: ignore[reportPrivateUsage]
    create_heat_viewer_app,
    damage_hud_layout,
)
from racing.game.config import CameraView, HeatViewerConfig
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatEntrant, HeatRaceEntry
from racing.race.runtime import RaceCarRuntime
from racing.student.api import RobotCommand, RobotController, RobotSensors, load_student_controller


def _heat_entrants(entrant_count: int = 4) -> tuple[HeatEntrant, ...]:
    return tuple(
        HeatEntrant(
            name=f"Team {index + 1}",
            controller=cast(RobotController, Mock(return_value=RobotCommand(throttle=index / entrant_count))),
            team_color=DEFAULT_HEAT_COLORS[index],
        )
        for index in range(entrant_count)
    )


@pytest.mark.parametrize("entrant_count", [2, 3, 4, 7, 8, 9, 10, 20])
def test_heat_viewer_preserves_grid_order_and_entrant_metadata(entrant_count: int) -> None:
    config = HeatViewerConfig(entrants=_heat_entrants(entrant_count), random_seed=110)
    orders: set[tuple[int, ...]] = set()

    for race_index in range(1, 5):
        entries = _race_viewer_entries(config=config, race_index=race_index)
        assert len(entries) == entrant_count
        entrant_indices: list[int] = []
        for entry in entries:
            assert isinstance(entry, HeatRaceEntry)
            entrant_indices.append(entry.entrant_index)
            entrant = config.entrants[entry.entrant_index]
            assert _head_to_head_viewer_controller(config=config, entry=entry) is entrant.controller
            assert _head_to_head_car_label(config=config, entry=entry) == entrant.name
            assert _head_to_head_car_paint_color(config=config, entry=entry) == entrant.team_color
            assert not _head_to_head_viewer_keyboard_controlled(config=config, entry=entry)
        assert entrant_indices == list(range(entrant_count))
        assert _race_viewer_entries(config=config, race_index=race_index) == entries
        orders.add(tuple(entrant_indices))

    assert orders == {tuple(range(entrant_count))}


def test_retired_car_badge_hides_without_projection_and_returns_after_reset(monkeypatch: pytest.MonkeyPatch) -> None:
    background, text = Mock(), Mock()
    text.node.return_value.calcWidth.return_value = 5.0
    robot = SimpleNamespace(eliminated=True)
    runtime = cast(RaceCarRuntime, SimpleNamespace(robot=robot, label=HeadToHeadCarLabel(background, text)))
    project = Mock(return_value=(0.0, 0.0))
    monkeypatch.setattr(app, "_head_to_head_car_label_screen_position", project)

    _update_head_to_head_car_labels(ursina=Mock(), view=CameraView.THREE_QUARTER, runtimes=(runtime,))
    project.assert_not_called()
    background.hide.assert_called_once()
    text.hide.assert_called_once()
    background.show.assert_not_called()
    text.show.assert_not_called()

    robot.eliminated = False
    _update_head_to_head_car_labels(ursina=Mock(), view=CameraView.THREE_QUARTER, runtimes=(runtime,))
    project.assert_called_once()
    background.show.assert_called_once()
    text.show.assert_called_once()


@pytest.mark.parametrize("entrant_count", [2, 3, 4, 7, 8, 9, 10, 20])
def test_heat_viewer_recreates_factory_controller_state_for_each_car_and_race(
    tmp_path: Path, entrant_count: int
) -> None:
    module = tmp_path / "factory_driver.py"
    module.write_text(
        "from racing import RobotCommand\n"
        "def create_controller():\n"
        "    calls = 0\n"
        "    def control(sensors):\n"
        "        nonlocal calls\n"
        "        calls += 1\n"
        "        return RobotCommand(throttle=calls / 10.0)\n"
        "    return control\n",
        encoding="utf-8",
    )
    prototype = load_student_controller(module)
    config = HeatViewerConfig(
        entrants=tuple(
            HeatEntrant(name=f"Team {index}", controller=prototype, team_color=color)
            for index, color in enumerate(DEFAULT_HEAT_COLORS[:entrant_count])
        )
    )
    first = _head_to_head_viewer_controllers(config=config, entries=_race_viewer_entries(config=config, race_index=1))
    for controller in first:
        assert controller is not None
        assert controller(RobotSensors()).throttle == pytest.approx(0.1)
        assert controller(RobotSensors()).throttle == pytest.approx(0.2)
    second = _head_to_head_viewer_controllers(config=config, entries=_race_viewer_entries(config=config, race_index=2))
    for controller in second:
        assert controller is not None
        assert controller(RobotSensors()).throttle == pytest.approx(0.1)
    assert len(first) == len(second) == entrant_count
    assert len({id(controller) for controller in (*first, *second, prototype)}) == 2 * entrant_count + 1


@pytest.mark.parametrize("entrant_count", [0, 1, 21])
def test_heat_viewer_rejects_wrong_entrant_count_before_graphics_startup(
    entrant_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    create_scene = Mock()
    monkeypatch.setattr(app, "_build_race_viewer_scene", create_scene)
    entrants = (_heat_entrants() * 6)[:entrant_count]

    with pytest.raises(ValueError, match="two to twenty"):
        create_heat_viewer_app(HeatViewerConfig(entrants=entrants))

    create_scene.assert_not_called()


def test_heat_viewer_accepts_split_view(monkeypatch: pytest.MonkeyPatch) -> None:
    create_scene = Mock()
    monkeypatch.setattr(app, "_build_race_viewer_scene", create_scene)

    config = HeatViewerConfig(entrants=_heat_entrants(), camera_view=CameraView.SPLIT_FOLLOW)
    assert create_heat_viewer_app(config) is create_scene.return_value
    create_scene.assert_called_once_with(config)


@pytest.mark.parametrize("entrant_count", [2, 3, 4, 7, 8, 9, 10, 20])
def test_heat_viewer_accepts_supported_counts_before_starting_shared_scene(
    entrant_count: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    viewer = Mock()
    create_scene = Mock(return_value=viewer)
    monkeypatch.setattr(app, "_build_race_viewer_scene", create_scene)
    config = HeatViewerConfig(entrants=_heat_entrants(entrant_count))

    assert create_heat_viewer_app(config) is viewer
    create_scene.assert_called_once_with(config)


@pytest.mark.parametrize(("entrant_count", "row_counts"), [(8, [4, 4]), (9, [4, 4, 1])])
def test_larger_heat_damage_hud_wraps_without_overlap(entrant_count: int, row_counts: list[int]) -> None:
    slots = damage_hud_layout(entrant_count)

    assert len(slots) == entrant_count
    assert len({(slot.center_x, slot.center_y) for slot in slots}) == entrant_count
    rows = sorted({slot.center_y for slot in slots})
    assert [sum(slot.center_y == row for slot in slots) for row in rows] == row_counts
    for first, second in combinations(slots, 2):
        separated_horizontally = abs(first.center_x - second.center_x) > (first.width + second.width) / 2.0
        separated_vertically = abs(first.center_y - second.center_y) > (first.height + second.height) / 2.0
        assert separated_horizontally or separated_vertically
