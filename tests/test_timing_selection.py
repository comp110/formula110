from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from racing.game.app import (
    _head_to_head_camera_target_runtime,  # pyright: ignore[reportPrivateUsage]
    _race_viewer_car_id,  # pyright: ignore[reportPrivateUsage]
)
from racing.game.config import CameraView, HeadToHeadViewerConfig, HeatViewerConfig
from racing.graphics.camera import CameraRig, update_camera_cycle
from racing.race.head_to_head import HeadToHeadRaceEntry
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatEntrant, HeatRaceEntry, heat_race_entries
from racing.race.progress import LapProgressTracker
from racing.race.runtime import RaceCarRuntime
from racing.student.api import default_student_controller


def _runtime(distance_m: float, *, eliminated: bool = False) -> RaceCarRuntime:
    return RaceCarRuntime(
        robot=cast(Any, SimpleNamespace(eliminated=eliminated)),
        tracker=LapProgressTracker(total_length_m=100.0, best_distance_m=distance_m),
    )


def _heat_config(count: int = 4) -> HeatViewerConfig:
    return HeatViewerConfig(
        entrants=tuple(
            HeatEntrant(
                name=f"Car {index}", controller=default_student_controller, team_color=DEFAULT_HEAT_COLORS[index]
            )
            for index in range(count)
        )
    )


@pytest.mark.parametrize("car_id", ["heat-3:0", "challenger:1", None])
def test_row_selection_enters_close_follow_and_resets_previous_camera_history(car_id: str | None) -> None:
    rig = CameraRig(
        view=CameraView.SPLIT_FOLLOW,
        selected_car_id="previous-car",
        follow_target_id=123,
        follow_direction_initialized=True,
    )
    rig.follow_forward_samples.append((0.1, 1.0, 0.0))

    rig.select_follow_car(car_id)

    assert rig.view is CameraView.FOLLOW
    assert rig.selected_car_id == car_id
    assert rig.follow_target_id is None
    assert not rig.follow_direction_initialized
    assert rig.follow_forward_samples == []


@pytest.mark.parametrize("include_split", [False, True])
def test_selected_car_survives_camera_cycle_back_to_follow(include_split: bool) -> None:
    rig = CameraRig()
    rig.select_follow_car("heat-2:0")
    visited_views: list[CameraView] = []
    for _ in range(5 if include_split else 4):
        update_camera_cycle(rig, cycle_key_down=True, include_split=include_split)
        visited_views.append(rig.view)
        assert rig.selected_car_id == "heat-2:0"
        update_camera_cycle(rig, cycle_key_down=False, include_split=include_split)
    assert CameraView.TOP_DOWN in visited_views
    assert rig.view is CameraView.FOLLOW


def test_selected_heat_car_stays_followed_when_rank_changes_and_auto_restores_leader() -> None:
    config = _heat_config()
    entries = tuple(HeatRaceEntry(index) for index in range(4))
    runtimes = tuple(_runtime(index * 10.0) for index in range(4))
    rig = CameraRig()
    rig.select_follow_car(_race_viewer_car_id(entries[1]))

    for distances in ((0.0, 10.0, 20.0, 30.0), (40.0, 100.0, 60.0, 80.0), (90.0, 10.0, 30.0, 20.0)):
        for runtime, distance_m in zip(runtimes, distances, strict=True):
            runtime.tracker.best_distance_m = distance_m
        assert (
            _head_to_head_camera_target_runtime(
                config=config, entries=entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
            )
            is runtimes[1]
        )

    rig.select_follow_car(None)
    assert (
        _head_to_head_camera_target_runtime(
            config=config, entries=entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
        )
        is runtimes[0]
    )


@pytest.mark.parametrize("count", [4, 8])
def test_selected_identity_resolves_new_runtime_slot_after_race_shuffle(count: int) -> None:
    config = _heat_config(count)
    first_entries = heat_race_entries(entrant_count=count, race_index=1, random_seed=110)
    # Identity-based camera selection also survives a caller-provided grid reorder.
    next_entries = tuple(reversed(first_entries))
    runtimes = tuple(_runtime(index * 10.0) for index in range(count))
    changed_slot = next(index for index in range(count) if first_entries[index] != next_entries[index])
    selected_entry = first_entries[changed_slot]
    rig = CameraRig()
    rig.select_follow_car(_race_viewer_car_id(selected_entry))
    assert (
        _head_to_head_camera_target_runtime(
            config=config, entries=first_entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
        )
        is runtimes[changed_slot]
    )

    rig.reset_follow_history()
    next_slot = next_entries.index(selected_entry)
    assert next_slot != changed_slot
    assert (
        _head_to_head_camera_target_runtime(
            config=config, entries=next_entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
        )
        is runtimes[next_slot]
    )


def test_head_to_head_selection_distinguishes_team_copies_and_overrides_keyboard_target() -> None:
    config = HeadToHeadViewerConfig(challenger_keyboard=True, incumbent_copies=2)
    entries = (
        HeadToHeadRaceEntry("incumbent", 1),
        HeadToHeadRaceEntry("challenger", 0),
        HeadToHeadRaceEntry("incumbent", 0),
    )
    runtimes = (_runtime(10.0), _runtime(20.0), _runtime(30.0))
    ids = tuple(_race_viewer_car_id(entry) for entry in entries)
    assert ids == ("incumbent:1", "challenger:0", "incumbent:0")
    assert _head_to_head_camera_target_runtime(config=config, entries=entries, runtimes=runtimes) is runtimes[1]

    for index in (0, 2):
        assert (
            _head_to_head_camera_target_runtime(
                config=config, entries=entries, runtimes=runtimes, selected_car_id=ids[index]
            )
            is runtimes[index]
        )


def test_selected_eliminated_car_stays_followed_until_auto_is_selected() -> None:
    config = _heat_config()
    entries = tuple(HeatRaceEntry(index) for index in range(4))
    runtimes = (_runtime(100.0, eliminated=True), _runtime(30.0), _runtime(20.0), _runtime(10.0))
    rig = CameraRig()
    rig.select_follow_car(_race_viewer_car_id(entries[0]))

    assert (
        _head_to_head_camera_target_runtime(
            config=config, entries=entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
        )
        is runtimes[0]
    )
    rig.select_follow_car(None)
    assert (
        _head_to_head_camera_target_runtime(
            config=config, entries=entries, runtimes=runtimes, selected_car_id=rig.selected_car_id
        )
        is runtimes[1]
    )
