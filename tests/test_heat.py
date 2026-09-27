from __future__ import annotations

import json
from types import SimpleNamespace
from typing import cast

import pytest

from racing import RobotCommand, RobotSensors
from racing.race.heat import (
    DEFAULT_HEAT_COLORS,
    HeatEntrant,
    HeatRaceEntry,
    HeatRaceResult,
    HeatResult,
    HeatStanding,
    format_heat_result,
    format_heat_result_banner,
    heat_race_entries,
    heat_race_result_from_runtimes,
    run_headless_heat,
    validate_heat_entrants,
)
from racing.race.rules import HeadToHeadRaceRules
from racing.race.runtime import RaceCarRuntime
from racing.student.api import RobotController


def _control(sensors: RobotSensors) -> RobotCommand:
    return RobotCommand()


def _entrants(count: int = 4) -> tuple[HeatEntrant, ...]:
    return tuple(
        HeatEntrant(name=f"Car {index}", controller=_control, team_color=color)
        for index, color in enumerate(DEFAULT_HEAT_COLORS[:count])
    )


@pytest.mark.parametrize("count", [0, 1, 2, 3, 5, 6, 7, 9])
def test_heat_requires_four_or_eight_entrants_before_starting_physics(count: int) -> None:
    entrants = tuple(_entrants()[0] for _ in range(count))

    with pytest.raises(ValueError, match="exactly four or eight"):
        validate_heat_entrants(entrants)
    with pytest.raises(ValueError, match="exactly four or eight"):
        run_headless_heat(entrants=entrants)
    with pytest.raises(ValueError, match="exactly four or eight"):
        heat_race_entries(entrant_count=count, race_index=1)
    with pytest.raises(ValueError, match="exactly four or eight"):
        HeatResult(entrant_names=tuple(entrant.name for entrant in entrants), round_seconds=30.0, races=())


@pytest.mark.parametrize("race_index", [1, 2, 3])
@pytest.mark.parametrize("entrant_count", [4, 8])
def test_heat_grid_preserves_input_order_and_individual_identity(race_index: int, entrant_count: int) -> None:
    arguments = {"entrant_count": entrant_count, "race_index": race_index, "random_seed": 781}
    entries = heat_race_entries(**arguments)
    assert entries == heat_race_entries(**arguments)
    assert entries == heat_race_entries(entrant_count=entrant_count, race_index=race_index, random_seed=42)
    assert [entry.entrant_index for entry in entries] == list(range(entrant_count))
    assert {entry.role for entry in entries} == {f"heat-{index}" for index in range(entrant_count)}
    assert all(entry.copy_index == 0 for entry in entries)


def _runtime(distance: float, *, penalty: float = 0.0, damage: float = 0.0) -> RaceCarRuntime:
    return cast(
        RaceCarRuntime,
        SimpleNamespace(
            tracker=SimpleNamespace(best_distance_m=distance, lap_count=int(distance // 100)),
            robot=SimpleNamespace(damage=damage, eliminated=damage >= 1.0),
            marshal_count=int(penalty > 0.0),
            marshal_penalty_m=penalty,
        ),
    )


def test_heat_scores_each_controller_separately_after_grid_shuffle_and_marshal_penalties() -> None:
    entries = (HeatRaceEntry(2), HeatRaceEntry(0), HeatRaceEntry(3), HeatRaceEntry(1))
    runtimes = (
        _runtime(160.0, penalty=70.0, damage=0.9),
        _runtime(100.0, damage=1.0),
        _runtime(5.0, penalty=10.0),
        _runtime(120.0, damage=0.2),
    )

    result = heat_race_result_from_runtimes(entrants=_entrants(), entries=entries, runtimes=runtimes, race_index=2)

    assert result.race_index == 2
    assert [(standing.entrant_index, standing.distance_m) for standing in result.standings] == [
        (1, 120.0),
        (0, 100.0),
        (2, 90.0),
        (3, 0.0),
    ]
    assert [standing.name for standing in result.standings] == ["Car 1", "Car 0", "Car 2", "Car 3"]
    assert result.standings[1].damage == 1.0
    assert result.standings[2].marshal_count == 1


def _race(index: int, distances: tuple[float, ...], *, names: tuple[str, ...] | None = None) -> HeatRaceResult:
    entrant_names = names if names is not None else tuple("Same name" for _ in distances)
    return HeatRaceResult(
        race_index=index,
        standings=tuple(
            HeatStanding(entrant, entrant_names[entrant], distance, entrant, index * 0.1, index)
            for entrant, distance in reversed(tuple(enumerate(distances)))
        ),
    )


@pytest.mark.parametrize("entrant_count", [4, 8])
def test_heat_aggregates_total_individual_distance_across_races_with_duplicate_names(entrant_count: int) -> None:
    first_distances = (10.0, 100.0, 80.0, 50.0, 5.0, 200.0, 210.0, 300.0)[:entrant_count]
    second_distances = (100.0, 0.0, 60.0, 20.0, 5.0, 0.0, 20.0, 0.0)[:entrant_count]
    result = HeatResult(
        entrant_names=("Same name",) * entrant_count,
        round_seconds=30.0,
        races=(_race(1, first_distances), _race(2, second_distances)),
        random_seed=42,
        track_id="procedural",
        track_seed=987,
    )

    assert result.race_count == 2
    expected_standings = [
        (2, 140.0),
        (0, 110.0),
        (1, 100.0),
        (3, 70.0),
    ]
    if entrant_count == 8:
        expected_standings = [(7, 300.0), (6, 230.0), (5, 200.0), *expected_standings, (4, 10.0)]
    assert result.winner_index == expected_standings[0][0]
    assert [(standing.entrant_index, standing.distance_m) for standing in result.standings] == expected_standings
    assert result.standings[0].lap_count == expected_standings[0][0] * 2
    assert result.standings[0].damage == pytest.approx(0.15)
    assert result.standings[0].marshal_count == 3
    payload = json.loads(json.dumps(result.to_dict(), allow_nan=False))
    assert payload["track_seed"] == 987
    assert payload["summary"]["winner_index"] == expected_standings[0][0]
    assert payload["summary"]["standings"][0]["entrant_index"] == expected_standings[0][0]
    assert len(payload["summary"]["standings"]) == entrant_count
    assert payload["scoring"] == "total-distance"
    assert "scoring" not in payload["rules"]


def test_heat_tied_totals_share_first_place_without_arbitrary_winner() -> None:
    names = ("Alpha", "Beta", "Gamma", "Delta")
    result = HeatResult(
        entrant_names=names, round_seconds=30.0, races=(_race(1, (20.0, 20.0, 15.0, 10.0), names=names),)
    )

    assert result.winner_index is None
    assert format_heat_result_banner(result).splitlines() == [
        "HEAT RESULTS",
        "1. Alpha  20.0 m  |  10.0% total damage",
        "1. Beta  20.0 m  |  10.0% total damage",
        "3. Gamma  15.0 m  |  10.0% total damage",
        "4. Delta  10.0 m  |  10.0% total damage",
    ]


def test_heat_banner_shortens_names_without_changing_terminal_or_json_names() -> None:
    long_name = "  An exceptionally long\n\tstudent racing team name  "
    names = (long_name, "Beta", "Gamma", "Delta")
    result = HeatResult(
        entrant_names=names, round_seconds=30.0, races=(_race(1, (40.0, 30.0, 20.0, 10.0), names=names),)
    )

    banner = format_heat_result_banner(result)
    assert len(banner.splitlines()) == 5
    assert "…" in banner
    assert long_name in format_heat_result(result)
    assert result.to_dict()["entrant_names"] == list(names)


def test_heat_banner_sums_damage_across_races_instead_of_showing_the_average() -> None:
    names = ("KJ", "MJ", "AB", "CD")
    result = HeatResult(
        entrant_names=names,
        round_seconds=30.0,
        races=(_race(1, (40.0, 30.0, 20.0, 10.0), names=names), _race(2, (40.0, 30.0, 20.0, 10.0), names=names)),
    )

    assert result.standings[0].damage == pytest.approx(0.15)
    assert "KJ  80.0 m  |  30.0% total damage" in format_heat_result_banner(result)


@pytest.mark.parametrize("entrant_count", [4, 8])
def test_heat_rejects_results_that_drop_or_repeat_entrants(entrant_count: int) -> None:
    duplicate_entries = (HeatRaceEntry(0),) * entrant_count
    with pytest.raises(ValueError, match="every entrant exactly once"):
        heat_race_result_from_runtimes(
            entrants=_entrants(entrant_count),
            entries=duplicate_entries,
            runtimes=tuple(_runtime(1.0) for _ in range(entrant_count)),
            race_index=1,
        )
    with pytest.raises(ValueError, match="every entrant exactly once"):
        HeatResult(
            entrant_names=tuple(entrant.name for entrant in _entrants(entrant_count)),
            round_seconds=30.0,
            races=(_race(1, (1.0,) * (entrant_count - 1)),),
        )


class _RecordingController:
    def __init__(self, throttle: float) -> None:
        self.throttle = throttle
        self.race_ticks: list[list[int]] = []
        self.competitor_distances: list[tuple[float, ...]] = []

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        raise AssertionError("each race must use a fresh controller copy")

    def copy_for_car(self) -> RobotController:
        ticks: list[int] = []
        self.race_ticks.append(ticks)

        def control(sensors: RobotSensors) -> RobotCommand:
            ticks.append(sensors.tick)
            self.competitor_distances.append(tuple(competitor.distance_m for competitor in sensors.camera.competitors))
            return RobotCommand(throttle=self.throttle)

        return control


@pytest.mark.parametrize("entrant_count", [4, 8])
def test_headless_heat_runs_all_controllers_with_fresh_state_each_race(entrant_count: int) -> None:
    controllers = tuple(_RecordingController(index * 0.1) for index in range(entrant_count))
    entrants = tuple(
        HeatEntrant(name=f"Controller {index}", controller=controller, team_color=DEFAULT_HEAT_COLORS[index])
        for index, controller in enumerate(controllers)
    )

    result = run_headless_heat(
        entrants=entrants,
        race_count=2,
        round_seconds=0.05,
        random_seed=172,
        rules=HeadToHeadRaceRules(marshal_enabled=False),
    )

    assert result.race_count == 2
    assert all(controller.race_ticks == [[0, 1, 2], [0, 1, 2]] for controller in controllers)
    for race in result.races:
        assert sorted(standing.entrant_index for standing in race.standings) == list(range(entrant_count))
        assert {standing.name for standing in race.standings} == {entrant.name for entrant in entrants}
    assert len(result.standings) == entrant_count
    assert len(format_heat_result_banner(result).splitlines()) == entrant_count + 1
    for controller in controllers:
        assert len(controller.competitor_distances) == 6
        assert all(len(distances) <= 3 for distances in controller.competitor_distances)
        assert all(distances == tuple(sorted(distances)) for distances in controller.competitor_distances)
