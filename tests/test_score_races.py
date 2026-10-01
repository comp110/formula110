from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from racing.race.interest import InterestWeights


@pytest.fixture
def scorer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    spec = importlib.util.spec_from_file_location("score_races", scripts / "score_races.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_leaderboard_command_preserves_names_rules_track_and_paths(scorer: ModuleType, tmp_path: Path) -> None:
    export = tmp_path / "export with spaces"
    name = "$(touch should-not-exist)"
    command = shlex.join(
        [
            "uv",
            "run",
            "python",
            "scripts/run_submissions.py",
            "101",
            "102",
            "103",
            "104",
            "--name",
            name,
            "--name",
            "B",
            "--name",
            "C",
            "--name",
            "D",
            "--export-dir",
            str(export),
            "--track-seed",
            "42",
            "--seed",
            "11",
            "--round-laps",
            "3",
            "--finish-timeout-seconds",
            "12",
            "--no-damage",
            "--no-marshal",
            "--fullscreen",
            "--camera",
            "cinematic",
        ]
    )
    spec = scorer.specs_from_commands("# Clock It (s) — lower is better\n# 1. A\n" + command)[0]
    assert spec.submissions == ("101", "102", "103", "104")
    assert spec.names == (name, "B", "C", "D")
    assert spec.label == "Clock It (s)"
    assert spec.track == "procedural" and spec.track_seed == 42 and spec.seed == 11
    assert spec.round_laps == 3 and spec.finish_timeout_seconds == 12
    assert not spec.rules.damage_enabled and not spec.rules.marshal_enabled
    replay = scorer.specs_from_commands(scorer.replay_command(spec))[0]
    assert asdict(replay) == {**asdict(spec), "label": "field"}
    assert not Path("should-not-exist").exists()


def test_cli_overrides_inherited_settings_without_confusing_track_and_spawn_seeds(scorer: ModuleType) -> None:
    spec = scorer.specs_from_commands(
        "uv run python scripts/run_submissions.py 1 2 3 4 --track-seed 22 --seed 9 --round-laps 2 --no-damage"
    )[0]
    args = scorer.build_parser().parse_args(["--track", "bahrain", "--seed", "100", "--laps", "6", "--damage"])
    result = scorer.apply_overrides(spec, args)
    assert result.track == "bahrain" and result.track_seed is None and result.seed == 100
    assert result.round_laps == 6 and result.rules.damage_enabled
    args = scorer.build_parser().parse_args(["--track-seed", "0", "--no-damage"])
    result = scorer.apply_overrides(result, args)
    assert result.track == "procedural" and result.track_seed == 0
    assert not result.rules.damage_enabled


@pytest.mark.parametrize("extra", ["--races 2", "--round-seconds 30", "--round-seconds=30"])
def test_incompatible_source_race_options_fail_instead_of_being_ignored(scorer: ModuleType, extra: str) -> None:
    with pytest.raises(ValueError):
        scorer.specs_from_commands(f"uv run python scripts/run_submissions.py 1 2 3 4 {extra}")


def test_worker_is_a_fresh_interpreter_with_no_shell(scorer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '{"interest": {}, "result": {}}', "")

    monkeypatch.setattr(scorer.subprocess, "run", fake_run)
    spec = scorer.RaceSpec(submissions=("1", "2", "3", "4"))
    scorer.run_worker(spec, InterestWeights(), 30)
    command, kwargs = calls[0]
    assert command[0] == sys.executable and command[-1] == "--worker"
    assert not kwargs.get("shell", False)
    assert kwargs["timeout"] == 30
    restored = scorer.RaceSpec.from_dict(json.loads(kwargs["input"])["spec"])
    assert restored == spec


def test_batch_logs_timeouts_and_ranks_successful_trials(
    scorer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tried: list[int] = []

    def validate(spec: Any) -> None:
        assert spec.track == "mugello-short" and spec.round_laps == 5
        assert not spec.rules.damage_enabled

    def worker(spec: Any, weights: InterestWeights, timeout: float) -> dict[str, Any]:
        tried.append(spec.seed)
        assert weights == InterestWeights(30, 30, 40)
        if spec.seed == 111:
            raise subprocess.TimeoutExpired("race", timeout)
        return {
            "interest": {
                "score": spec.seed - 100,
                "lead_points": 0,
                "overtake_points": 0,
                "finish_points": spec.seed - 100,
                "lead_changes": 0,
                "overtakes": 0,
                "finishers": 4,
                "gap_p3_seconds": 1,
            },
            "result": {},
            "events": [],
        }

    monkeypatch.setattr(scorer.RaceSpec, "validate", validate)
    monkeypatch.setattr(scorer, "run_worker", worker)
    output = tmp_path / "scores.jsonl"
    assert scorer.main(["1", "2", "3", "4", "--count", "3", "--no-damage", "--output", str(output)]) == 2
    assert tried == [110, 111, 112]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [record["status"] for record in records] == ["completed", "timeout", "completed"]
    assert "interest" not in records[1]
    assert "1. field, seed 112" in capsys.readouterr().out


def test_explanation_runs_without_submission_exports(scorer: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    assert scorer.main(["--explain-score"]) == 0
    assert "finish 40%" in capsys.readouterr().out


def test_track_search_runs_each_layout_with_each_start_seed_and_saves_replays(
    scorer: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tried: list[tuple[int, int]] = []

    def validate(spec: Any) -> None:
        assert spec.track == "procedural" and spec.track_seed == 0

    def worker(spec: Any, weights: InterestWeights, timeout: float) -> dict[str, Any]:
        tried.append((spec.track_seed, spec.seed))
        points = spec.track_seed * 10 + spec.seed - 100
        return {
            "interest": {
                "score": points,
                "lead_points": 0,
                "overtake_points": 0,
                "finish_points": points,
                "lead_changes": 0,
                "overtakes": 0,
                "finishers": 4,
                "gap_p3_seconds": 1,
            },
            "result": {},
            "events": [],
        }

    monkeypatch.setattr(scorer.RaceSpec, "validate", validate)
    monkeypatch.setattr(scorer, "run_worker", worker)
    output = tmp_path / "track-search.jsonl"
    assert (
        scorer.main(
            [
                "1",
                "2",
                "3",
                "4",
                "--start-track-seed",
                "0",
                "--track-count",
                "2",
                "--start-seed",
                "110",
                "--count",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert tried == [(0, 110), (0, 111), (1, 110), (1, 111)]
    records = [json.loads(line) for line in output.read_text().splitlines()]
    assert [(record["spec"]["track_seed"], record["spec"]["seed"]) for record in records] == tried
    for record in records:
        replay = scorer.specs_from_commands(record["replay_command"])[0]
        assert replay.track == "procedural"
        assert replay.track_seed == record["spec"]["track_seed"]
        assert replay.seed == record["spec"]["seed"]
    console = capsys.readouterr().out
    assert "1 field(s) x 2 track(s) x 2 starting seed(s) = 4 races" in console
    assert "1. field, seed 111, procedural (track seed 1)" in console


def test_track_range_can_start_from_an_inherited_command_seed(scorer: ModuleType) -> None:
    spec = scorer.specs_from_commands("uv run python scripts/run_submissions.py 1 2 3 4 --track-seed 42 --seed 7")[0]
    args = scorer.build_parser().parse_args(["--track-count", "3"])
    spec = scorer.apply_overrides(spec, args)
    trials = list(scorer.trial_specs(spec, seed_count=args.count, track_count=args.track_count))
    assert [(trial.track_seed, trial.seed) for trial in trials] == [(42, 7), (43, 7), (44, 7)]


@pytest.mark.parametrize("track_args", [[], ["--track", "bahrain"]])
def test_track_range_requires_a_procedural_seed(
    scorer: ModuleType,
    track_args: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        scorer.main(["1", "2", "3", "4", "--track-count", "2", *track_args])
    assert error.value.code == 2
    assert "requires --track-seed" in capsys.readouterr().err


@pytest.mark.parametrize("count", ["0", "-1"])
def test_track_count_must_be_positive(scorer: ModuleType, count: str) -> None:
    with pytest.raises(SystemExit) as error:
        scorer.build_parser().parse_args(["--track-count", count])
    assert error.value.code == 2
