from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest


@pytest.fixture
def search(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    spec = importlib.util.spec_from_file_location("search_bahrain_seeds", scripts / "search_bahrain_seeds.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def standings(first: int = 0, second: int = 3, *, second_dnf: bool = False) -> list[dict[str, Any]]:
    names = ("LB+NN", "AR", "YZ", "JB", "YH", "KC", "ZY+JL", "JS+MH")
    order = [first, second, *(index for index in range(8) if index not in (first, second))]
    return [
        {
            "entrant_index": index,
            "name": names[index],
            "finish_position": None if second_dnf and position == 2 else position,
            "finish_time_seconds": None if second_dnf and position == 2 else 200.0 + position,
            "dnf": second_dnf and position == 2,
        }
        for position, index in enumerate(order, start=1)
    ]


def race_result(seed: int, rows: list[dict[str, Any]], round_laps: int = 5) -> dict[str, Any]:
    return {
        "mode": "heat",
        "random_seed": seed,
        "track_id": "bahrain",
        "round_laps": round_laps,
        "races": [{"race_index": 1, "standings": rows}],
    }


def fake_resolve_controller(export_dir: Path, identifier: str) -> Path:
    return export_dir / identifier


@pytest.mark.parametrize(
    ("first", "second", "either_order", "expected"),
    [(0, 3, False, True), (3, 0, False, False), (3, 0, True, True), (0, 1, True, False), (1, 3, False, False)],
)
def test_matches_actual_finish_positions(
    search: ModuleType,
    first: int,
    second: int,
    either_order: bool,
    expected: bool,
) -> None:
    # The result array need not be sorted; use finish positions and stable IDs.
    rows = list(reversed(standings(first, second)))
    assert search.is_match(rows, either_order=either_order) is expected


def test_dnf_in_second_row_does_not_match(search: ModuleType) -> None:
    rows = standings(second_dnf=True)
    assert not search.is_match(rows, either_order=True)
    rows[1]["finish_position"] = 2
    assert not search.is_match(rows, either_order=True)


@pytest.mark.parametrize("round_laps", [1, 3, 5])
def test_headless_trial_and_replay_preserve_all_race_settings(
    search: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    round_laps: int,
) -> None:
    export_dir = tmp_path / "export with spaces"
    calls: list[tuple[list[str], dict[str, Any]]] = []
    result = race_result(42, standings(), round_laps)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, json.dumps(result), "diagnostics")

    monkeypatch.setattr(search.subprocess, "run", fake_run)
    assert search.run_seed(42, export_dir, 123.0, round_laps=round_laps) == result
    command, kwargs = calls[0]
    replay = shlex.split(search.replay_command(42, export_dir, round_laps=round_laps))
    assert command[0] == sys.executable
    assert command[1:-2] == replay[3:-3]
    assert command[-2:] == ["--headless", "--json"]
    assert replay[-3:] == ["--fullscreen", "--camera", "cinematic"]
    assert command[2:10] == [identifier for identifier, _ in search.ENTRANTS]
    assert [command[i + 1] for i, value in enumerate(command) if value == "--name"] == [
        name for _, name in search.ENTRANTS
    ]
    for flag, expected in (
        ("--track", "bahrain"),
        ("--seed", "42"),
        ("--races", "1"),
        ("--round-laps", str(round_laps)),
        ("--finish-timeout-seconds", "10"),
        ("--export-dir", str(export_dir)),
    ):
        assert command[command.index(flag) + 1] == expected
    assert "--no-damage" in command
    assert kwargs["timeout"] == 123.0
    assert kwargs["cwd"] == search.PROJECT_ROOT


@pytest.mark.parametrize("lap_flag", ["--round-laps", "--laps"])
def test_search_skips_timeouts_and_stops_after_requested_matches(
    search: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    lap_flag: str,
) -> None:
    tried: list[int] = []
    monkeypatch.setattr(search, "resolve_controller", fake_resolve_controller)

    def fake_seed(seed: int, *_: Any, round_laps: int) -> dict[str, Any]:
        tried.append(seed)
        assert round_laps == 3
        if seed == 10:
            raise subprocess.TimeoutExpired("race", 5)
        return race_result(seed, standings(3, 0) if seed == 11 else standings(), round_laps)

    monkeypatch.setattr(search, "run_seed", fake_seed)
    output_file = tmp_path / "nested" / "results.jsonl"
    assert (
        search.main(
            [
                "--start-seed",
                "10",
                "--count",
                "20",
                "--matches",
                "2",
                "--either-order",
                lap_flag,
                "3",
                "--output",
                str(output_file),
            ]
        )
        == 0
    )
    assert tried == [10, 11, 12]
    records = [json.loads(line) for line in output_file.read_text().splitlines()]
    assert records[0]["status"] == "timeout"
    assert all(record["matched"] for record in records[1:])
    assert "--seed 11" in records[1]["replay_command"]
    assert "--round-laps 3" in records[1]["replay_command"]
    assert records[1]["result"]["round_laps"] == 3
    assert records[1]["result"]["races"][0]["standings"][0]["name"] == "JB"
    assert "Matched seeds: 11, 12; skipped timeouts: 1." in capsys.readouterr().out


def test_search_exhausts_range_without_accepting_reversed_finish(
    search: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tried: list[int] = []
    monkeypatch.setattr(search, "resolve_controller", fake_resolve_controller)

    def fake_seed(seed: int, *_: Any, round_laps: int) -> dict[str, Any]:
        tried.append(seed)
        assert round_laps == 5
        return race_result(seed, standings(3, 0))

    monkeypatch.setattr(search, "run_seed", fake_seed)
    assert search.main(["--start-seed", "7", "--count", "2"]) == 1
    assert tried == [7, 8]


@pytest.mark.parametrize(("returncode", "stdout"), [(1, ""), (0, "not JSON")])
def test_failed_races_are_reported_as_errors(
    search: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
) -> None:
    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, returncode, stdout, "controller error")

    monkeypatch.setattr(search.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        search.run_seed(42, Path("export"), 30.0)


@pytest.mark.parametrize(
    "args",
    [
        ["--count", "0"],
        ["--matches", "-1"],
        ["--timeout-seconds", "nan"],
        ["--timeout-seconds", "0"],
        ["--round-laps", "0"],
        ["--laps", "-1"],
        ["--round-laps", "1.5"],
    ],
)
def test_invalid_search_limits_are_rejected(search: ModuleType, args: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        search.build_parser().parse_args(args)
    assert error.value.code == 2
