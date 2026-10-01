from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from racing.game import cli
from racing.game.config import HeadToHeadViewerConfig, HeatViewerConfig
from racing.race.head_to_head import run_headless_head_to_head
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatEntrant, run_headless_heat
from racing.race.rules import HeadToHeadRaceRules
from racing.student.api import RobotCommand, RobotSensors


@pytest.mark.parametrize("entrant_count", [2, 4, 8])
@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("no_damage", [False, True])
def test_cli_forwards_damage_rule_to_each_race_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, entrant_count: int, watch: bool, no_damage: bool
) -> None:
    controller = tmp_path / "controller.py"
    controller.write_text("from racing import RobotCommand\ndef control(sensors):\n    return RobotCommand()\n")
    runner = Mock()
    if entrant_count == 2:
        args = ["h2h", "--challenger-module", str(controller), "--incumbent-module", str(controller)]
        runner_name = "create_head_to_head_viewer_app" if watch else "run_headless_head_to_head"
        formatter_name = "format_head_to_head_result"
    else:
        args = ["heat", *(["--module", str(controller)] * entrant_count)]
        runner_name = "create_heat_viewer_app" if watch else "run_headless_heat"
        formatter_name = "format_heat_result"
    monkeypatch.setattr(cli, runner_name, runner)
    monkeypatch.setattr(cli, formatter_name, Mock(return_value="results"))

    cli.main([*args, *(["--watch"] if watch else []), *(["--no-damage"] if no_damage else [])])

    runner.assert_called_once()
    if watch:
        config = cast(HeadToHeadViewerConfig | HeatViewerConfig, runner.call_args.args[0])
        rules = config.rules
        runner.return_value.run.assert_called_once()
    else:
        rules = cast(HeadToHeadRaceRules, runner.call_args.kwargs["rules"])
    assert rules.damage_enabled is (not no_damage)
    assert rules.marshal_enabled is True


@pytest.mark.parametrize("entrant_count", [2, 4, 8])
@pytest.mark.parametrize("no_damage", [False, True])
def test_races_preserve_wall_contacts_but_disable_damage_when_requested(entrant_count: int, no_damage: bool) -> None:
    samples: list[RobotSensors] = []

    def crash_into_wall(sensors: RobotSensors) -> RobotCommand:
        samples.append(sensors)
        return RobotCommand(throttle=1.0)

    rules = (
        HeadToHeadRaceRules(marshal_enabled=False, damage_enabled=False)
        if no_damage
        else HeadToHeadRaceRules(marshal_enabled=False)
    )
    if entrant_count == 2:
        result = run_headless_head_to_head(
            challenger_controller=crash_into_wall,
            incumbent_controller=crash_into_wall,
            copies_per_side=1,
            race_count=2,
            round_seconds=5.0,
            random_seed=42,
            rules=rules,
        )
        damages_by_race = [list(race.challenger.damages + race.incumbent.damages) for race in result.races]
        payload = result.to_dict()
    else:
        heat_result = run_headless_heat(
            entrants=tuple(
                HeatEntrant(name=f"Car {index}", controller=crash_into_wall, team_color=color)
                for index, color in enumerate(DEFAULT_HEAT_COLORS[:entrant_count])
            ),
            race_count=2,
            round_seconds=5.0,
            random_seed=42,
            rules=rules,
        )
        damages_by_race = [[standing.damage for standing in race.standings] for race in heat_result.races]
        payload = heat_result.to_dict()

    assert any(sensors.contact.wall > 0.0 for sensors in samples)
    assert cast(dict[str, object], payload["rules"])["damage_enabled"] is (not no_damage)
    for damages in damages_by_race:
        if no_damage:
            assert damages == [0.0] * entrant_count
        else:
            assert max(damages) > 0.0
    if no_damage:
        assert all(sensors.contact.damage == 0.0 for sensors in samples)
