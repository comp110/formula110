from __future__ import annotations

import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


@pytest.fixture
def generator(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    spec = importlib.util.spec_from_file_location("leaderboard_heats", scripts / "leaderboard_heats.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def make_submission(export_dir: Path, identifier: str, name: str = "Level 3") -> None:
    root = export_dir / f"submission_{identifier}"
    (root / "controllers").mkdir(parents=True)
    (root / "formula110-submission.json").write_text(
        json.dumps({"schema_version": 1, "controller_module": "controllers.driver"}), encoding="utf-8"
    )
    (root / "controllers" / "driver.py").write_text(
        f"RACING_NAME: str = {name!r}\nraise RuntimeError('must not run student code')\n", encoding="utf-8"
    )


def write_metadata(export_dir: Path, payload: object) -> None:
    (export_dir / "submission_metadata.yml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def record(entries: list[dict[str, object]]) -> dict[str, object]:
    return {":results": {"leaderboard": entries}}


@pytest.fixture
def exported_leaders(tmp_path: Path) -> Path:
    export_dir = tmp_path / "export with spaces"
    payload: dict[str, object] = {}
    for index in range(1, 11):
        identifier = str(index)
        make_submission(export_dir, identifier, "Driver's $(touch should-not-exist)")
        payload[f"submission_{identifier}"] = {
            ":submitters": [{":name": "Kris Jordan"}, {":name": "Morgan Jordan"}]
            if index % 2 == 0
            else [{":name": "Kris Jordan"}],
            **record(
                [
                    {"name": "Clock It (s)", "order": "asc", "value": index},
                    {"name": "Hits Different (damage %)", "value": index},
                ]
            ),
            ":history": [{":id": 999, **record([{"name": "Clock It (s)", "order": "asc", "value": -999}])}],
        }
    write_metadata(export_dir, payload)
    return export_dir


def commands(output: str) -> list[list[str]]:
    return [shlex.split(line) for line in output.splitlines() if line and not line.startswith("#")]


@pytest.mark.parametrize("entrant_count", [8, 9, 10])
def test_current_scores_rank_in_both_directions_and_commands_resolve_without_loading_code(
    generator: ModuleType, exported_leaders: Path, capsys: pytest.CaptureFixture[str], entrant_count: int
) -> None:
    generator.main(
        [
            "--export-dir",
            str(exported_leaders),
            "--cars", str(entrant_count),
            "--",
            "--seed",
            "110",
            "--fullscreen",
            "--races",
            "3",
        ]
    )
    output = capsys.readouterr()
    assert not output.err
    printed = commands(output.out)
    assert len(printed) == 2
    expected_orders = [list(range(1, entrant_count + 1)), list(range(10, 10 - entrant_count, -1))]
    for command, expected in zip(printed, expected_orders, strict=True):
        script_index = command.index("python") + 1
        assert command[script_index + 1 : script_index + 1 + entrant_count] == [str(i) for i in expected]
        assert command.count("--name") == entrant_count
        names = [command[i + 1] for i, arg in enumerate(command) if arg == "--name"]
        assert names == ["KJ+MJ" if i % 2 == 0 else "KJ" for i in expected]
        assert "Driver's" not in output.out
        assert command[-5:] == ["--seed", "110", "--fullscreen", "--races", "3"]
        result = subprocess.run(
            [sys.executable, *command[script_index:], "--dry-run"],
            cwd=Path(__file__).parents[1],
            capture_output=True,
            text=True,
            check=True,
        )
        resolved = shlex.split(result.stdout)
        assert resolved.count("--module") == entrant_count
        assert "--fullscreen" in resolved
        assert resolved[resolved.index("--races") + 1] == "3"
        assert resolved[resolved.index("--seed") + 1] == "110"
    assert "# 1." in output.out
    assert "999" not in output.out


@pytest.mark.parametrize("available", [8, 9, 10])
def test_default_heat_uses_up_to_ten_eligible_cars(
    generator: ModuleType, exported_leaders: Path, capsys: pytest.CaptureFixture[str], available: int,
) -> None:
    metadata = exported_leaders / "submission_metadata.yml"
    payload = yaml.safe_load(metadata.read_text())
    for identifier in range(available + 1, 11):
        del payload[f"submission_{identifier}"]
    write_metadata(exported_leaders, payload)
    generator.main(["--export-dir", str(exported_leaders)])
    printed = commands(capsys.readouterr().out)
    assert len(printed) == 2
    assert all(command.count("--name") == available for command in printed)


def test_missing_scores_and_controllers_are_excluded_and_numeric_ties_are_stable(
    generator: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload: dict[str, object] = {}
    values = {"10": 0, "2": "0", "3": None, "4": "N/A", "5": float("nan"), "6": float("inf"), "7": True}
    for identifier, score in values.items():
        make_submission(tmp_path, identifier)
        payload[f"submission_{identifier}"] = record([{"name": "Future Trophy", "value": score}])
    payload["submission_99"] = record([{"name": "Future Trophy", "value": 999}])
    write_metadata(tmp_path, payload)

    generator.main(["--export-dir", str(tmp_path)])

    output = capsys.readouterr()
    assert "# 1. #2 (submission 2): 0" in output.out
    assert "# 2. #10 (submission 10): 0" in output.out
    assert "need 8 distinct eligible cars; found 2" in output.out
    assert not commands(output.out)
    assert "submission 99 not found" in output.err


def test_filter_accepts_punctuation_and_repeated_metrics(
    generator: ModuleType, exported_leaders: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    generator.main(
        ["--export-dir", str(exported_leaders), "--metric", "clock-it", "--metric", "hits different", "--list"]
    )
    output = capsys.readouterr().out
    assert "Clock It (s) — lower is better; 10 eligible cars" in output
    assert "Hits Different (damage %) — higher is better; 10 eligible cars" in output
    assert not commands(output)
    board = generator.Leaderboard("Gs Going Crazy (g-s)", "desc")
    assert generator.select_boards([board], ["g's going crazy"]) == [board]


@pytest.mark.parametrize(("query", "message"), [("missing", "unknown metric"), ("i", "ambiguous metric")])
def test_bad_metric_reports_choices(
    generator: ModuleType, exported_leaders: Path, capsys: pytest.CaptureFixture[str], query: str, message: str
) -> None:
    with pytest.raises(SystemExit) as error:
        generator.main(["--export-dir", str(exported_leaders), "--metric", query])
    assert error.value.code == 2
    assert message in capsys.readouterr().err


def test_four_car_option_and_commands_work_outside_project(
    generator: ModuleType,
    exported_leaders: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    generator.main(["--export-dir", str(exported_leaders), "--metric", "clock", "--cars", "4"])
    command = commands(capsys.readouterr().out)[0]
    assert "--project" in command
    script_index = command.index("python") + 1
    result = subprocess.run(
        [sys.executable, *command[script_index:], "--dry-run"], capture_output=True, text=True, check=True
    )
    assert shlex.split(result.stdout).count("--module") == 4


@pytest.mark.parametrize("orders", [["asc", "desc"], ["ascending"]])
def test_inconsistent_or_unknown_directions_fail_clearly(
    generator: ModuleType, tmp_path: Path, orders: list[str]
) -> None:
    payload: dict[str, object] = {}
    for i, order in enumerate(orders, start=1):
        make_submission(tmp_path, str(i))
        payload[f"submission_{i}"] = record([{"name": "Clock It (s)", "order": order, "value": i}])
    write_metadata(tmp_path, payload)
    with pytest.raises(ValueError, match=r"sort orders|invalid order"):
        generator.read_leaderboards(tmp_path)


def test_duplicate_metric_does_not_put_same_car_on_grid_twice(generator: ModuleType, tmp_path: Path) -> None:
    make_submission(tmp_path, "1")
    entry: dict[str, object] = {"name": "Clock It", "order": "asc", "value": 1}
    write_metadata(tmp_path, {"submission_1": record([entry, entry])})
    with pytest.raises(ValueError, match="duplicate leaderboard entry"):
        generator.read_leaderboards(tmp_path)


@pytest.mark.parametrize("text", ["[", "[]", "!!python/object:builtins.object {}"])
def test_malformed_metadata_reports_error(generator: ModuleType, tmp_path: Path, text: str) -> None:
    (tmp_path / "submission_metadata.yml").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=r"could not read|expected a mapping"):
        generator.read_leaderboards(tmp_path)


def test_absent_metadata_reports_path(generator: ModuleType, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"submission_metadata\.yml"):
        generator.read_leaderboards(tmp_path)
