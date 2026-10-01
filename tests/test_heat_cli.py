from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest

from racing.game import cli
from racing.game.config import CameraView, HeatViewerConfig
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatEntrant
from racing.race.rules import HeadToHeadRaceRules
from racing.student.api import RobotSensors


@pytest.fixture
def heat_modules(tmp_path: Path, request: pytest.FixtureRequest) -> list[str]:
    entrant_count = cast(int, getattr(request, "param", 4))
    args: list[str] = []
    for index in range(entrant_count):
        path = tmp_path / f"entrant_{index}" / "controller.py"
        path.parent.mkdir()
        metadata = "RACING_NAME = 'Student Team'\nRACING_COLOR = '#ff8000'\n" if index == 0 else ""
        path.write_text(
            "from racing import RobotCommand\n"
            + metadata
            + "def control(sensors):\n"
            + f"    return RobotCommand(throttle={(index + 1) / 10})\n",
            encoding="utf-8",
        )
        args.extend(["--module", str(path)])
    return args


@pytest.mark.parametrize("heat_modules", [2, 3, 4, 7, 8, 9, 10, 20], indirect=True)
def test_headless_heat_loads_independent_controllers_and_forwards_race_settings(
    heat_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    entrant_count = len(heat_modules) // 2
    fallback_names = [f"#{index * 101}" for index in range(1, entrant_count + 1)]
    expected_names = ["Student Team", *fallback_names[1:]]

    class FakeResult:
        def to_dict(self) -> dict[str, object]:
            return {"schema_version": 1, "entrant_names": expected_names}

    def fake_heat(**kwargs: object) -> FakeResult:
        captured.update(kwargs)
        return FakeResult()

    monkeypatch.setattr(cli, "run_headless_heat", fake_heat)
    cli.main(
        [
            "heat",
            *heat_modules,
            *[value for name in fallback_names for value in ("--fallback-name", name)],
            "--seed",
            "7656",
            "--track-seed",
            "2026",
            "--races",
            "3",
            "--round-seconds",
            "4",
            "--fixed-delta-seconds",
            "0.02",
            "--no-marshal",
            "--json",
        ]
    )

    entrants = cast(tuple[HeatEntrant, ...], captured["entrants"])
    assert [entrant.name for entrant in entrants] == expected_names
    assert [entrant.controller(RobotSensors()).throttle for entrant in entrants] == [
        index / 10 for index in range(1, entrant_count + 1)
    ]
    assert entrants[0].team_color == (1.0, 128 / 255, 0.0, 1.0)
    assert tuple(entrant.team_color for entrant in entrants[1:]) == DEFAULT_HEAT_COLORS[1:entrant_count]
    assert captured["random_seed"] == 7656
    assert captured["track_id"] == "procedural"
    assert captured["track_seed"] == 2026
    assert captured["race_count"] == 3
    assert captured["round_seconds"] == 4.0
    assert captured["fixed_delta_seconds"] == 0.02
    rules = cast(HeadToHeadRaceRules, captured["rules"])
    assert rules.marshal_enabled is False
    assert json.loads(capsys.readouterr().out) == FakeResult().to_dict()


@pytest.mark.parametrize("heat_modules", [2, 3, 4, 7, 8, 9, 10, 20], indirect=True)
def test_watched_heat_applies_names_camera_window_and_audio_settings(
    heat_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configs: list[HeatViewerConfig] = []
    runs: list[bool] = []
    names = [f"Override {index}" for index in range(1, len(heat_modules) // 2 + 1)]

    class FakeApp:
        def run(self) -> None:
            runs.append(True)

    def fake_viewer(config: HeatViewerConfig) -> FakeApp:
        configs.append(config)
        return FakeApp()

    monkeypatch.setattr(cli, "create_heat_viewer_app", fake_viewer)
    cli.main(
        [
            "heat",
            *heat_modules,
            "--watch",
            "--fullscreen",
            "--size",
            "960x600",
            "--window-type",
            "offscreen",
            "--camera",
            "follow",
            "--no-music",
            "--muted",
            "--seed",
            "42",
            "--track-seed",
            "110",
            "--races",
            "2",
            "--round-seconds",
            "10",
            "--marshal-penalty-m",
            "7",
            *[value for name in names for value in ("--name", name)],
        ]
    )

    assert runs == [True]
    assert len(configs) == 1
    config = configs[0]
    assert [entrant.name for entrant in config.entrants] == names
    assert config.fullscreen is True
    assert config.size == (960, 600)
    assert config.window_type == "offscreen"
    assert config.camera_view is CameraView.FOLLOW
    assert config.audio.music_enabled is False
    assert config.audio.muted is True
    assert config.random_seed == 42
    assert config.track_id == "procedural"
    assert config.track_seed == 110
    assert config.race_count == 2
    assert config.round_seconds == 10.0
    assert config.rules.marshal_penalty_m == 7.0


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--module", "missing.py"], "heat requires two to twenty --module arguments"),
        (["--name", "First"], "--name must be repeated 4 times"),
        (["--fallback-name", "First"], "--fallback-name must be repeated 4 times"),
        (["--watch", "--json"], "--json is only available for headless heats"),
    ],
)
def test_invalid_heat_arguments_fail_before_loading_modules(
    args: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    modules = [] if "--module" in args else ["--module", "missing.py"] * 4
    with pytest.raises(SystemExit) as error:
        cli.main(["heat", *modules, *args])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("entrant_count", [1, 21])
def test_heat_rejects_unsupported_entrant_counts(entrant_count: int, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["heat", *(["--module", "missing.py"] * entrant_count)])

    assert error.value.code == 2
    assert "heat requires two to twenty --module arguments" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--name", "--fallback-name"])
@pytest.mark.parametrize("entrant_count", [8, 9])
def test_larger_heats_require_one_name_per_car_when_provided(
    flag: str, entrant_count: int, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["heat", *(["--module", "missing.py"] * entrant_count), *([flag, "Named"] * 4)])

    assert error.value.code == 2
    assert f"{flag} must be repeated {entrant_count} times" in capsys.readouterr().err


def test_heat_accepts_split_follow_camera() -> None:
    args = cli.build_argument_parser().parse_args(
        ["heat", *(["--module", "missing.py"] * 4), "--camera", "split_follow"],
    )
    assert args.camera == CameraView.SPLIT_FOLLOW.value


@pytest.mark.parametrize("option", [["--scoring", "best-copy"], ["--win-margin-m", "2"]])
def test_heat_rejects_head_to_head_scoring_options(option: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.build_argument_parser().parse_args(["heat", *(["--module", "missing.py"] * 4), *option])

    assert error.value.code == 2
    assert f"unrecognized arguments: {' '.join(option)}" in capsys.readouterr().err


def test_heat_defaults_to_shared_three_quarter_camera_and_accepts_global_seed() -> None:
    args = cli.build_argument_parser().parse_args(["--seed", "110", "heat", *(["--module", "missing.py"] * 4)])

    assert args.camera == "three_quarter"
    assert args.seed == 110
    assert args.watch is False


@pytest.mark.parametrize("watch", [False, True])
@pytest.mark.parametrize("timeout", [None, 0.0, 7.5])
def test_lap_limit_is_forwarded_to_both_heat_runners(
    heat_modules: list[str], monkeypatch: pytest.MonkeyPatch, watch: bool, timeout: float | None
) -> None:
    from unittest.mock import Mock

    headless, viewer = Mock(), Mock()
    monkeypatch.setattr(cli, "run_headless_heat", headless)
    monkeypatch.setattr(cli, "create_heat_viewer_app", viewer)
    monkeypatch.setattr(cli, "format_heat_result", Mock(return_value="results"))
    timeout_args = [] if timeout is None else ["--finish-timeout-seconds", str(timeout)]
    cli.main(["heat", *heat_modules, "--round-laps", "3", *timeout_args, *(["--watch"] if watch else [])])

    if watch:
        assert viewer.call_args.args[0].round_laps == 3
        assert viewer.call_args.args[0].finish_timeout_seconds == (20.0 if timeout is None else timeout)
    else:
        assert headless.call_args.kwargs["round_laps"] == 3
        assert headless.call_args.kwargs["finish_timeout_seconds"] == (20.0 if timeout is None else timeout)


@pytest.mark.parametrize("timeout", ["-1", "nan", "inf", "-inf"])
def test_invalid_finish_timeouts_fail_before_loading_controllers(
    timeout: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(
            ["heat", *(["--module", "missing.py"] * 4), "--round-laps", "3", f"--finish-timeout-seconds={timeout}"]
        )
    assert error.value.code == 2
    assert "--finish-timeout-seconds must be finite and nonnegative" in capsys.readouterr().err


@pytest.mark.parametrize(
    "length_args",
    [
        ["--round-laps", "0"],
        ["--round-laps", "-1"],
        ["--round-laps", "1.5"],
        ["--round-laps", "3", "--round-seconds", "60"],
    ],
)
def test_invalid_lap_limits_fail_before_loading_controllers(length_args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["heat", *(["--module", "missing.py"] * 4), *length_args])
    assert error.value.code == 2


def test_headless_heat_formats_terminal_results(
    heat_modules: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = object()

    def fake_heat(**_kwargs: object) -> object:
        return sentinel

    def fake_format(result: object) -> str:
        assert result is sentinel
        return "Heat standings"

    monkeypatch.setattr(cli, "run_headless_heat", fake_heat)
    monkeypatch.setattr(cli, "format_heat_result", fake_format)
    cli.main(["heat", *heat_modules])

    assert capsys.readouterr().out == "Heat standings\n"


def test_heat_json_is_parseable_when_submission_prints_during_import_and_ticks(
    heat_modules: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    first_controller = Path(heat_modules[1])
    first_controller.write_text(
        "from racing import RobotCommand\n"
        "print('controller imported')\n"
        "def control(sensors):\n"
        "    print('controller tick')\n"
        "    return RobotCommand(throttle=0.2)\n",
        encoding="utf-8",
    )

    cli.main(["heat", *heat_modules, "--round-seconds", "0.05", "--json"])

    output = capsys.readouterr()
    payload = json.loads(output.out)
    assert payload["mode"] == "heat"
    assert len(payload["races"]) == 1
    assert "controller imported" in output.err
    assert "controller tick" in output.err
