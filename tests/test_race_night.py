from __future__ import annotations

import csv
import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from racing.game import cli


@pytest.fixture
def planner(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    scripts = Path(__file__).parents[1] / "scripts"
    monkeypatch.setattr(sys, "path", [str(scripts), *sys.path])
    spec = importlib.util.spec_from_file_location("plan_race_night", scripts / "plan_race_night.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def export(tmp_path: Path) -> Path:
    root = tmp_path / "export with spaces"
    payload: dict[str, Any] = {}
    boards = (
        ("Sips Tea (g-s)", "asc"),
        ("Gs Going Crazy (g-s)", "desc"),
        ("Hits Different (damage %)", "desc"),
        ("Gas Locked In (s)", "asc"),
        ("Clock It (s)", "asc"),
        ("All Spawns, No Crumbs (Laps)", "desc"),
    )
    for index in range(1, 61):
        directory = root / f"submission_{index}"
        (directory / "controllers").mkdir(parents=True)
        (directory / "formula110-submission.json").write_text(
            json.dumps({"schema_version": 1, "controller_module": "controllers.driver"})
        )
        (directory / "controllers" / "driver.py").write_text(
            "from racing import RobotCommand\n"
            "RACING_NAME = 'Demo'\nRACING_COLOR = '#123456'\n"
            "def control(sensors):\n    return RobotCommand()\n"
        )
        payload[f"submission_{index}"] = {
            ":submitters": [{":name": f"Racer {index}", ":email": "do-not-export@example.com", ":sid": "private"}],
            ":results": {
                "leaderboard": [
                    {"name": name, "value": 61 - index if name.startswith("All Spawns") else index, "order": order}
                    for name, order in boards
                ]
            },
        }
    (root / "submission_metadata.yml").write_text(yaml.safe_dump(payload))
    return root


def test_priority_allocation_backfills_in_metric_order(planner: ModuleType, export: Path) -> None:
    groups, warnings = planner.select_groups(export)
    assert not warnings
    ids = {g["id"]: [int(c["submission_id"]) for c in g["competitors"]] for g in groups}
    assert ids == {
        "clock-it": list(range(1, 11)),
        "gas-locked-in": list(range(11, 21)),
        "hits-different": list(range(60, 50, -1)),
        "gs-going-crazy": list(range(50, 40, -1)),
        "sips-tea": list(range(21, 31)),
    }
    assert len({identifier for group in ids.values() for identifier in group}) == 50
    assert groups[3]["competitors"][0]["leaderboard_rank"] == 11
    assert groups[3]["skipped_higher_priority"][0]["assigned_category"] == "clock-it"


def test_lowest_priority_allocation_preserves_metric_order_and_backfills(planner: ModuleType, export: Path) -> None:
    groups, warnings = planner.select_groups(export, "lowest")
    assert not warnings
    ids = {g["id"]: [int(c["submission_id"]) for c in g["competitors"]] for g in groups}
    assert ids == {
        "sips-tea": list(range(1, 11)),
        "gs-going-crazy": list(range(60, 50, -1)),
        "hits-different": list(range(50, 40, -1)),
        "gas-locked-in": list(range(11, 21)),
        "clock-it": list(range(21, 31)),
    }
    assert len({identifier for group in ids.values() for identifier in group}) == 50
    assert [g["allocation_rank"] for g in groups] == [1, 2, 3, 4, 5]
    assert groups[4]["competitors"][0]["leaderboard_rank"] == 21
    assert groups[4]["skipped_already_assigned"][0]["assigned_category"] == "sips-tea"
    assert all(not g["skipped_higher_priority"] for g in groups)


@pytest.mark.parametrize("selection_priority", ["highest", "lowest"])
def test_short_field_fails_instead_of_reusing_previously_selected_cars(
    planner: ModuleType, export: Path, selection_priority: str
) -> None:
    path = export / "submission_metadata.yml"
    records = yaml.safe_load(path.read_text())
    path.write_text(yaml.safe_dump({k: v for k, v in records.items() if int(k.split("_")[1]) < 50}))
    with pytest.raises(ValueError, match="need 10 after earlier category"):
        planner.select_groups(export, selection_priority)


def fake_race(spec: Any, _weights: Any, _timeout: float) -> dict[str, Any]:
    # Winners deliberately come from the bottom of the qualifying leaderboard.
    lap_seconds = 30 if spec.track == "bahrain" else 10 + (spec.track_seed or 0) % 3
    duration = spec.round_laps * lap_seconds + 5
    standings = [
        {
            "entrant_index": i,
            "name": spec.names[i],
            "finish_position": place,
            "finish_time_seconds": duration - len(spec.submissions) + place,
            "dnf": False,
        }
        for place, i in enumerate(reversed(range(len(spec.submissions))), start=1)
    ]
    return {
        "duration_seconds": duration,
        "interest": {"score": spec.seed / 10 + (spec.track_seed or 0), "finishers": len(spec.submissions)},
        "result": {"races": [{"standings": standings}]},
        "events": [],
    }


def setup_workers(planner: ModuleType, monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    tried: list[Any] = []

    def worker(spec: Any, weights: Any, timeout: float) -> dict[str, Any]:
        tried.append(spec)
        return fake_race(spec, weights, timeout)

    monkeypatch.setattr(planner, "run_worker", worker)

    def inspect(_path: str, _timeout: float) -> dict[str, Any]:
        return {"racing_name": "Demo", "declared_color_rgba": [0.1, 0.2, 0.3, 1.0]}

    monkeypatch.setattr(planner, "inspect_controller", inspect)
    return tried


@pytest.mark.parametrize("selection_priority", ["highest", "lowest"])
@pytest.mark.parametrize("bahrain_cars", [10, 20])
def test_full_pipeline_bahrain_fields_commands_and_resume(
    planner: ModuleType,
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selection_priority: str,
    bahrain_cars: int,
) -> None:
    tried = setup_workers(planner, monkeypatch)
    output = tmp_path / "show plan.json"
    args = [
        "--no-calibrate-laps",
        "--selection-priority",
        selection_priority,
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--track-count",
        "2",
        "--starts-per-track",
        "2",
        "--fixed-seed-count",
        "3",
        "--bahrain-seed-count",
        "3",
        "--bahrain-cars",
        str(bahrain_cars),
        "--jobs",
        "2",
        "--marshal-penalty-m",
        "12.5",
        "--marshal-stuck-seconds",
        "2.5",
        "--marshal-cooldown-seconds",
        "4",
        "--laps",
        "clock-it=12",
        "--camera",
        "helicopter",
        "--no-fullscreen",
        "--no-music",
        "--muted",
    ]
    assert planner.main(args) == 0
    plan = json.loads(output.read_text())
    assert plan["status"] == "complete"
    assert plan["settings"]["selection_priority"] == selection_priority
    category_ids = [g["id"] for g in plan["groups"]]
    assert plan["allocation_order"] == (category_ids if selection_priority == "lowest" else category_ids[::-1])
    assert plan["selection_policy"].startswith(selection_priority.capitalize())
    assert len(tried) == 25  # 4 procedural groups * 4 + Clock It * 3 + 2 Bahrain fields * 3.
    assert len(plan["show_runner"]) == 7
    assert len({c["submission_id"] for g in plan["groups"] for c in g["competitors"]}) == 50
    assert all(
        [c["submission_id"] for c in group["competitors"]] == [str(i) for i in range(1, bahrain_cars + 1)]
        for group in plan["bahrain_groups"]
    )
    assert plan["schema_version"] == 2
    assert "do-not-export@example.com" not in output.read_text()
    assert '"private"' not in output.read_text()
    for stage in [*plan["groups"], *plan["bahrain_groups"]]:
        assert len(stage["top_races"]) == 3
        for race in stage["top_races"]:
            launch = race["launch"]
            argv = launch["argv"]
            assert argv == shlex.split(launch["shell_command"])
            assert argv[argv.index("--marshal-penalty-m") + 1] == "12.5"
            assert argv[argv.index("--camera") + 1] == "helicopter"
            assert argv[argv.index("--title") + 1] == stage["title"]
            assert "--fullscreen" not in argv
            assert "--no-music" in argv and "--muted" in argv
            assert "--no-damage" in argv
            assert len(race["classification"]) == (bahrain_cars if stage["track"] == "bahrain" else 10)
            assert [c["finish_position"] for c in race["podium"]] == [1, 2, 3]
    assert all(spec.rules.marshal_penalty_m == 12.5 for spec in tried)
    assert all(spec.rules.marshal_stuck_seconds == 2.5 for spec in tried)
    assert all(spec.rules.marshal_cooldown_seconds == 4 for spec in tried)
    assert {spec.round_laps for spec in tried if spec.label == "Clock It"} == {12}
    assert {spec.track_seed for spec in tried if spec.label == "Clock It"} == {None}
    assert {spec.track for spec in tried if spec.label == "Clock It"} == {"mugello-short"}
    clock_argv = plan["groups"][4]["top_races"][0]["launch"]["argv"]
    assert "--track-seed" not in clock_argv
    assert clock_argv[clock_argv.index("--track") + 1] == "mugello-short"
    assert {spec.round_laps for spec in tried if spec.track == "bahrain"} == {3}

    replay = plan["bahrain_groups"][0]["top_races"][0]["launch"]["argv"]
    script_index = replay.index("python") + 1
    dry_run = subprocess.run(
        [sys.executable, *replay[script_index:], "--dry-run"], capture_output=True, text=True, check=True
    )
    resolved = shlex.split(dry_run.stdout)
    parsed = cli.build_argument_parser().parse_args(resolved[resolved.index("racing") + 1 :])
    assert parsed.marshal_penalty_m == 12.5 and parsed.no_music and parsed.camera == "helicopter"

    assert planner.main([*args, "--resume", "--camera", "drone", "--fullscreen"]) == 0
    assert len(tried) == 25
    resumed = json.loads(output.read_text())
    argv = resumed["bahrain_groups"][0]["top_races"][0]["launch"]["argv"]
    assert argv[argv.index("--camera") + 1] == "drone" and "--fullscreen" in argv
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume", "--marshal-penalty-m", "0"])
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume", "--bahrain-cars", "15"])
    with pytest.raises(SystemExit):
        planner.main(
            [
                *args,
                "--resume",
                "--selection-priority",
                "highest" if selection_priority == "lowest" else "lowest",
            ]
        )
    (export / "submission_1" / "controllers" / "driver.py").write_text("# changed\n")
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume"])


def test_defaults_plan_hundred_trials_per_stage_and_two_bahrain_fields_without_running_races(
    planner: ModuleType,
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tried = setup_workers(planner, monkeypatch)
    output = tmp_path / "plan.json"
    args = ["--export-dir", str(export), "--output", str(output), "--plan-only", "--no-marshal", "--damage"]
    assert planner.main(args) == 0
    assert not tried
    plan = json.loads(output.read_text())
    assert [g["search"]["planned"] for g in plan["groups"]] == [100] * 5
    assert [g["search"]["planned"] for g in plan["bahrain_groups"]] == [100, 100]
    assert plan["settings"]["bahrain_seed_count"] == 100
    assert [g["round_laps"] for g in plan["groups"]] == [6, 6, 6, 8, 10]
    assert [g["track"] for g in plan["groups"]] == ["procedural"] * 4 + ["mugello-short"]
    assert plan["settings"]["rules"]["marshal_stuck_seconds"] == 1.0
    specs = planner.stage_specs(plan["groups"][0], plan["settings"])
    assert {(s.track_seed, s.seed) for s in specs} == {(t, s) for t in range(10) for s in range(110, 120)}
    clock_specs = planner.stage_specs(plan["groups"][4], plan["settings"] | {"start_track_seed": 42})
    assert {(s.track_seed, s.seed) for s in clock_specs} == {(None, s) for s in range(110, 210)}
    assert all(s.track == "mugello-short" for s in clock_specs)
    assert plan["groups"][4]["search"]["track_count"] == 1
    assert all(g["calibrate_laps"] for g in plan["groups"])
    assert all(not g["calibrate_laps"] and g["round_laps"] == 3 for g in plan["bahrain_groups"])
    assert all(not spec.rules.marshal_enabled and spec.rules.damage_enabled for spec in specs)
    launch = planner.launch_details(specs[0], plan["settings"]["playback"])
    assert "--no-marshal" in launch["argv"] and "--no-damage" not in launch["argv"]


def test_failed_trials_are_saved_and_retried_without_repeating_successes(
    planner: ModuleType,
    export: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_workers(planner, monkeypatch)
    calls: list[tuple[str, int]] = []

    def flaky(spec: Any, weights: Any, timeout: float) -> dict[str, Any]:
        calls.append((spec.label, spec.seed))
        if spec.seed == 110:
            raise subprocess.TimeoutExpired("worker", timeout)
        return fake_race(spec, weights, timeout)

    monkeypatch.setattr(planner, "run_worker", flaky)
    output = tmp_path / "retry.json"
    args = [
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--track-count",
        "1",
        "--starts-per-track",
        "3",
        "--bahrain-seed-count",
        "3",
        "--no-calibrate-laps",
    ]
    assert planner.main(args) == 2
    assert len(calls) == 21
    plan = json.loads(output.read_text())
    assert plan["status"] == "complete_with_errors"
    assert all(g["search"]["failed"] == 1 for g in plan["groups"])
    log = [json.loads(line) for line in Path(plan["trial_log"]).read_text().splitlines()]
    assert sum(record["status"] == "timeout" for record in log) == 7

    def recovered(spec: Any, weights: Any, timeout: float) -> dict[str, Any]:
        calls.append((spec.label, spec.seed))
        return fake_race(spec, weights, timeout)

    monkeypatch.setattr(planner, "run_worker", recovered)
    assert planner.main([*args, "--resume", "--retry-failed"]) == 0
    assert len(calls) == 28
    assert json.loads(output.read_text())["status"] == "complete"


@pytest.mark.parametrize("car_count", [10, 20])
def test_bahrain_qualifies_directly_from_all_spawns_and_only_closed_field_excludes_juiced(
    planner: ModuleType, export: Path, car_count: int
) -> None:
    path = export / "submission_metadata.yml"
    payload = yaml.safe_load(path.read_text())
    for key, record in payload.items():
        record[":results"]["leaderboard"][-1]["value"] = int(key.removeprefix("submission_"))
    path.write_text(yaml.safe_dump(payload))
    groups = planner.build_bahrain_groups(export, {"60", "58"}, car_count)
    assert [c["submission_id"] for c in groups[0]["competitors"]] == ["59", *map(str, range(57, 58 - car_count, -1))]
    assert [c["submission_id"] for c in groups[1]["competitors"]] == list(map(str, range(60, 60 - car_count, -1)))
    assert all(g["round_laps"] == 3 and g["track"] == "bahrain" for g in groups)
    assert all(g["leaderboard"]["order"] == "desc" and "All Spawns, No Crumbs" in g["title"] for g in groups)
    assert {c["submission_id"] for c in groups[1]["competitors"] if c["is_juiced"]} == {"60", "58"}


def test_fallback_colors_match_grid_slots_and_declared_colors_survive_finale(planner: ModuleType) -> None:
    declared = {"submission_id": "1", "declared_color_rgba": [0.1, 0.2, 0.3, 1]}
    assert planner.color_metadata(declared, 9)["team_color_hex"] == "#1A334C"
    fallback = planner.color_metadata({"submission_id": "2", "declared_color_rgba": None}, 9)
    assert fallback["team_color_rgba"] == list(planner.DEFAULT_HEAT_COLORS[9])
    assert fallback["runtime_car_id"] == "heat-9:0"


def write_reviews(export: Path, held_count: int) -> Path:
    path = export.parent / "review scores.csv"
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["submission_id", "total_score", "clean_lap_seconds"])
        writer.writerows((i, 2 if i <= held_count else 0, 10) for i in range(1, 61))
    return path


@pytest.mark.parametrize("selection_priority", ["highest", "lowest"])
def test_holdouts_are_removed_before_priority_allocation_and_threshold_is_inclusive(
    planner: ModuleType, export: Path, selection_priority: str
) -> None:
    scores = {str(i): 0.0 for i in range(1, 61)}
    scores.update({"1": 1.0, "2": 1.5})
    groups, _ = planner.select_groups(export, selection_priority, review_scores=scores, juiced_threshold=1)
    selected = {c["submission_id"] for g in groups for c in g["competitors"]}
    assert "1" not in selected and "2" not in selected and len(selected) == 50
    assert any(row["submission_id"] == "2" for g in groups for row in g["skipped_juiced"])
    scores.update({str(i): 2.0 for i in range(1, 12)})
    with pytest.raises(ValueError, match=r"need 10.*juiced holdouts"):
        planner.select_groups(export, selection_priority, review_scores=scores)


@pytest.mark.parametrize("score", [None, "missing"])
def test_missing_review_cannot_silently_enter_a_category(planner: ModuleType, export: Path, score: Any) -> None:
    scores = {str(i): 0.0 for i in range(1, 61)}
    if score == "missing":
        del scores["1"]
    else:
        scores["1"] = score
    with pytest.raises(ValueError, match=r"missing/blank total_score.*1"):
        planner.select_groups(export, review_scores=scores)


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("submission_id,score\n1,0\n", "CSV columns"),
        ("submission_id,total_score,total_score\n1,0,1\n", "unique"),
        ("submission_id,total_score\n1,1\n1,0\n", "duplicate submission_id"),
        ("submission_id,total_score\n1,nan\n", "finite and nonnegative"),
        ("submission_id,total_score\n1,inf\n", "finite and nonnegative"),
        ("submission_id,total_score\n1,-1\n", "finite and nonnegative"),
        ("submission_id,total_score\n1,broken\n", "invalid total_score"),
        ("submission_id,total_score\n999,0\n", "not in this export"),
        ("submission_id,total_score\n../1,0\n", "invalid submission_id"),
    ],
)
def test_invalid_review_csv_fails_clearly(planner: ModuleType, export: Path, body: str, error: str) -> None:
    path = export.parent / "invalid.csv"
    path.write_text(body)
    with pytest.raises(ValueError, match=error):
        planner.read_review_scores(path, export)


def test_review_script_csv_is_consumed_without_conversion(planner: ModuleType, export: Path) -> None:
    controller = export / "submission_1/controllers/driver.py"
    controller.write_text(controller.read_text() + "\nfor i in range(1):\n    pass\n")
    output = export.parent / "actual-review.csv"
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts/review_submissions.py"),
            str(export),
            "--csv",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    scores = planner.read_review_scores(output, export)
    assert len(scores) == 60 and scores["1"] == 3 and scores["2"] == 0


@pytest.mark.parametrize("count", [0, 1, 2, 7, 10, 11, 21])
def test_juiced_fields_include_every_holdout_without_regular_fillers(
    planner: ModuleType, export: Path, count: int
) -> None:
    held = {str(i): 1.0 for i in range(1, count + 1)}
    groups, warnings = planner.build_juiced_groups(export, held, track="procedural", laps=6)
    if count < 2:
        assert not groups
        assert bool(warnings) == (count == 1)
        return
    assert not warnings
    ids = [c["submission_id"] for g in groups for c in g["competitors"]]
    assert len(ids) == len(set(ids)) == count and set(ids) == set(held)
    assert all(2 <= len(g["competitors"]) <= 10 for g in groups)
    assert all(c["category_id"] == "juiced" for g in groups for c in g["competitors"])


@pytest.mark.parametrize(("count", "track", "laps"), [(7, "procedural", 6), (2, "bahrain", 3)])
def test_juiced_pipeline_commands_metadata_and_bahrain_eligibility(
    planner: ModuleType, export: Path, monkeypatch: pytest.MonkeyPatch, count: int, track: str, laps: int
) -> None:
    tried = setup_workers(planner, monkeypatch)
    reviews = write_reviews(export, count)
    output = export.parent / "juiced-plan.json"
    args = [
        "--no-calibrate-laps",
        "--bahrain-seed-count",
        "3",
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--review-csv",
        str(reviews),
        "--selection-priority",
        "lowest",
        "--juiced-track",
        track,
        "--laps",
        f"juiced={laps}",
        "--track-count",
        "1",
        "--starts-per-track",
        "3",
    ]
    assert planner.main(args) == 0
    plan = json.loads(output.read_text())
    held = {str(i) for i in range(1, count + 1)}
    assert plan["status"] == "complete" and len(tried) == 24
    assert {r["submission_id"] for r in plan["holdout"]["submissions"]} == held
    assert plan["source"]["review_csv_sha256"] == planner.file_digest(reviews)
    assert len(plan["source"]["submission_sha256"]) == 50 + count
    assert len(plan["show_runner"]) == 8
    assert plan["show_order"][-3:] == ["juiced", "bahrain-no-juiced", "bahrain-open"]
    ordinary = {c["submission_id"] for g in plan["groups"] for c in g["competitors"]}
    closed_ids = {c["submission_id"] for c in plan["bahrain_groups"][0]["competitors"]}
    open_ids = {c["submission_id"] for c in plan["bahrain_groups"][1]["competitors"]}
    assert len(ordinary) == 50 and not held & ordinary and not held & closed_ids
    assert held.issubset(open_ids)
    juiced = plan["juiced_groups"][0]
    assert juiced["track"] == track and juiced["round_laps"] == laps
    assert len(juiced["top_races"]) == 3
    assert {c["submission_id"] for c in juiced["competitors"]} == held
    assert all(c["review_total_score"] == 2 for c in juiced["competitors"])
    for race in juiced["top_races"]:
        assert len(race["classification"]) == count
        assert set(race["spec"]["submissions"]) == held
        argv = race["launch"]["argv"]
        resolved = subprocess.run(
            [sys.executable, *argv[argv.index("python") + 1 :], "--dry-run"],
            capture_output=True,
            text=True,
            check=True,
        )
        command = shlex.split(resolved.stdout)
        parsed = cli.build_argument_parser().parse_args(command[command.index("racing") + 1 :])
        assert parsed.command == "heat" and len(parsed.module) == count and parsed.round_laps == laps
    assert planner.main([*args, "--resume"]) == 0 and len(tried) == 24
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume", "--juiced-threshold", "2"])
    # Even CSV edits that leave parsed scores alone invalidate saved review inputs.
    reviews.write_text(reviews.read_text() + "\n")
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume"])


def test_single_holdout_stays_out_without_creating_a_fake_opponent(
    planner: ModuleType, export: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_workers(planner, monkeypatch)
    reviews = write_reviews(export, 1)
    output = export.parent / "single.json"
    assert (
        planner.main(
            ["--export-dir", str(export), "--output", str(output), "--review-csv", str(reviews), "--plan-only"]
        )
        == 0
    )
    plan = json.loads(output.read_text())
    assert plan["holdout"]["submissions"] == [{"submission_id": "1", "total_score": 2.0}]
    assert not plan["juiced_groups"] and any("at least two" in w for w in plan["warnings"])
    assert all(c["submission_id"] != "1" for g in plan["groups"] for c in g["competitors"])


def test_inclusive_juiced_filters_require_both_values_and_preserve_zero(planner: ModuleType) -> None:
    scores = {"1": 10.0, "2": 11.0, "3": 9.9, "4": 11.0, "5": 11.0, "6": None, "7": 0.0}
    laps = {"1": 3.0, "2": 3.1, "3": 5.0, "4": 2.9, "5": None, "6": 5.0, "7": 0.0}
    assert planner.filter_juiced_scores(scores, 10.0, lap_scores=laps, min_laps=3.0) == {"1": 10.0, "2": 11.0}
    assert planner.filter_juiced_scores(scores, 0.0, lap_scores=laps, min_laps=0.0) == {
        "1": 10.0,
        "2": 11.0,
        "3": 9.9,
        "4": 11.0,
        "7": 0.0,
    }
    assert planner.filter_juiced_scores(scores, 10.0, lap_scores={}, min_laps=0.0) == {}


@pytest.mark.parametrize("selection_priority", ["highest", "lowest"])
def test_combined_juiced_filter_selection_exhibition_metadata_and_resume(
    planner: ModuleType, export: Path, monkeypatch: pytest.MonkeyPatch, selection_priority: str
) -> None:
    tried = setup_workers(planner, monkeypatch)
    reviews = export.parent / "filtered-reviews.csv"
    values = {1: (2, 3), 2: (3, 4), 3: (3, 2.9), 4: (3, ""), 5: (1.9, 10)}
    with reviews.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["submission_id", "total_score", "all_spawns_no_crumbs_laps"])
        writer.writerows((i, *values.get(i, (0, 3))) for i in range(1, 61))
    output = export.parent / "filtered-plan.json"
    args = [
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--review-csv",
        str(reviews),
        "--selection-priority",
        selection_priority,
        "--juiced-threshold",
        "2",
        "--juiced-min-laps",
        "3",
        "--track-count",
        "1",
        "--starts-per-track",
        "3",
        "--no-calibrate-laps",
        "--bahrain-seed-count",
        "3",
    ]
    assert planner.main(args) == 0
    plan = json.loads(output.read_text())
    assert len(tried) == 24
    assert plan["settings"]["juiced_threshold"] == 2
    assert plan["settings"]["juiced_min_laps"] == 3
    assert plan["holdout"]["comparison"] == plan["holdout"]["lap_comparison"] == ">="
    assert plan["holdout"]["min_laps"] == 3 and plan["holdout"]["criteria_operator"] == "and"
    assert plan["holdout"]["submissions"] == [
        {"submission_id": "1", "total_score": 2.0, "all_spawns_no_crumbs_laps": 3.0},
        {"submission_id": "2", "total_score": 3.0, "all_spawns_no_crumbs_laps": 4.0},
    ]
    regular = {c["submission_id"] for g in plan["groups"] for c in g["competitors"]}
    assert {"3", "4", "5"}.issubset(regular)
    assert not {"1", "2"} & regular
    juiced = plan["juiced_groups"][0]
    assert {c["submission_id"] for c in juiced["competitors"]} == {"1", "2"}
    assert [c["all_spawns_no_crumbs_laps"] for c in juiced["competitors"]] == [3.0, 4.0]
    assert not {"1", "2"} & {c["submission_id"] for c in plan["bahrain_groups"][0]["competitors"]}
    assert {"1", "2"}.issubset({c["submission_id"] for c in plan["bahrain_groups"][1]["competitors"]})
    assert planner.main([*args, "--resume"]) == 0 and len(tried) == 24
    with pytest.raises(SystemExit):
        planner.main([*args, "--resume", "--juiced-min-laps", "3.1"])


def test_lap_filter_requires_csv_and_metric_column(planner: ModuleType, export: Path) -> None:
    output = export.parent / "missing-laps.json"
    args = ["--export-dir", str(export), "--output", str(output), "--juiced-min-laps", "0", "--plan-only"]
    with pytest.raises(SystemExit):
        planner.main(args)
    reviews = write_reviews(export, 2)  # Older CSVs have no lap metric.
    with pytest.raises(SystemExit):
        planner.main([*args, "--review-csv", str(reviews)])
    assert not output.exists()


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "broken"])
def test_invalid_lap_scores_are_reported(planner: ModuleType, export: Path, value: str) -> None:
    reviews = export.parent / "invalid-laps.csv"
    reviews.write_text(f"submission_id,total_score,all_spawns_no_crumbs_laps\n1,10,{value}\n")
    with pytest.raises(ValueError, match="all_spawns_no_crumbs_laps"):
        planner.read_review_scores(reviews, export, column="all_spawns_no_crumbs_laps")


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_invalid_lap_threshold_is_rejected(planner: ModuleType, value: str) -> None:
    with pytest.raises(SystemExit):
        planner.build_parser().parse_args(["--juiced-min-laps", value])


def test_duration_quality_rewards_target_and_does_not_reward_all_dnfs(planner: ModuleType) -> None:
    settings = {"target_seconds": 90, "duration_tolerance_seconds": 10, "duration_weight": 0.2}
    record = {
        "duration_seconds": 90,
        "interest": {"score": 50},
        "result": {"races": [{"standings": [{"dnf": False}, {"dnf": False}]}]},
    }
    ideal = planner.race_quality(record, settings)
    assert ideal["score"] == 60 and ideal["duration_points"] == 20 and ideal["in_duration_window"]
    short = planner.race_quality({**record, "duration_seconds": 60}, settings)
    long = planner.race_quality({**record, "duration_seconds": 120}, settings)
    assert short["score"] == long["score"] < ideal["score"]
    assert not short["in_duration_window"]
    record["result"]["races"][0]["standings"] = [{"dnf": True}, {"dnf": True}]
    dnf = planner.race_quality(record, settings)
    assert dnf["duration_score"] == 0 and not dnf["in_duration_window"]


def test_calibration_selects_laps_per_layout_reuses_pilots_and_resumes_without_work(
    planner: ModuleType, export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tried = setup_workers(planner, monkeypatch)
    output = tmp_path / "calibrated.json"
    args = [
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--track-count",
        "3",
        "--starts-per-track",
        "3",
        "--bahrain-seed-count",
        "3",
    ]
    assert planner.main(args) == 0
    plan = json.loads(output.read_text())
    for stage in plan["groups"]:
        assert stage["search"]["completed"] == stage["search"]["planned"] == 9
        assert set(stage["lap_calibration"]) == ({"None"} if stage["id"] == "clock-it" else {"0", "1", "2"})
        for calibration in stage["lap_calibration"].values():
            assert calibration["status"] == "in_window"
            assert 80 <= calibration["duration_seconds"] <= 100
            assert 1 <= len(calibration["attempts"]) <= 6
        for race in stage["top_races"]:
            assert race["spec"]["round_laps"] == stage["lap_calibration"][str(race["spec"]["track_seed"])]["laps"]
            assert race["quality"]["in_duration_window"]
            assert race["launch"]["argv"][-2:] == ["--title", stage["title"]]
    assert len({c["laps"] for c in plan["groups"][0]["lap_calibration"].values()}) > 1
    assert all(not stage.get("lap_calibration") and stage["round_laps"] == 3 for stage in plan["bahrain_groups"])
    assert 51 < len(tried) <= 51 + (4 * 3 + 1) * 6
    keys = [(spec.label, spec.track_seed, spec.seed, spec.round_laps) for spec in tried]
    assert len(keys) == len(set(keys))  # Chosen pilots count toward the 10x10 search, not duplicate simulations.
    count = len(tried)
    assert planner.main([*args, "--resume"]) == 0 and len(tried) == count
    resumed = json.loads(output.read_text())
    assert [g["lap_calibration"] for g in resumed["groups"]] == [g["lap_calibration"] for g in plan["groups"]]
    for option, value in [("--duration-weight", "0.3"), ("--target-seconds", "95"), ("--calibration-attempts", "4")]:
        with pytest.raises(SystemExit):
            planner.main([*args, "--resume", option, value])


def test_interrupted_calibration_reuses_logged_pilots(
    planner: ModuleType, export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tried = setup_workers(planner, monkeypatch)
    search = planner.search_stages

    def interrupted(*args: Any, **kwargs: Any) -> None:
        search(*args, **kwargs)
        if kwargs.get("phase") == "calibration":
            raise KeyboardInterrupt

    monkeypatch.setattr(planner, "search_stages", interrupted)
    output = tmp_path / "interrupted.json"
    args = [
        "--export-dir",
        str(export),
        "--output",
        str(output),
        "--track-count",
        "1",
        "--starts-per-track",
        "2",
        "--bahrain-seed-count",
        "2",
    ]
    assert planner.main(args) == 130
    saved = json.loads(output.read_text())
    assert saved["status"] == "interrupted" and len(tried) == 5
    monkeypatch.setattr(planner, "search_stages", search)
    assert planner.main([*args, "--resume"]) == 0
    keys = [(spec.label, spec.track_seed, spec.seed, spec.round_laps) for spec in tried]
    assert len(keys) == len(set(keys))


@pytest.mark.parametrize("successful", [True, False])
def test_calibration_is_bounded_and_reports_unreachable_window(planner: ModuleType, successful: bool) -> None:
    settings = {
        "target_seconds": 90,
        "duration_tolerance_seconds": 5,
        "duration_weight": 0.2,
        "max_calibration_laps": 2,
        "calibration_attempts": 2,
    }
    stage = {"id": "sips-tea"}
    base = planner.RaceSpec(submissions=("1", "2"), round_laps=1)
    records = {}
    for laps in (1, 2):
        spec = planner.replace(base, round_laps=laps)
        record = {
            "status": "completed",
            "duration_seconds": laps * 60,
            "interest": {"score": 50},
            "result": {"races": [{"standings": [{"dnf": not successful}]}]},
        }
        record["quality"] = planner.race_quality(record, settings)
        records[planner.trial_id(stage["id"], spec)] = record
    calibration, pending = planner.calibration_step(stage, base, records, settings)
    assert pending is None and len(calibration["attempts"]) == 2
    assert calibration["laps"] == 1
    assert calibration["status"] == ("outside_window" if successful else "unavailable")


def test_fixed_lap_override_skips_calibration_and_bahrain_rejects_other_lengths(
    planner: ModuleType, export: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    setup_workers(planner, monkeypatch)
    output = tmp_path / "fixed.json"
    args = ["--export-dir", str(export), "--output", str(output), "--plan-only"]
    assert planner.main([*args, "--laps", "sips-tea=8"]) == 0
    stage = json.loads(output.read_text())["groups"][0]
    assert stage["round_laps"] == 8 and not stage["calibrate_laps"]
    with pytest.raises(SystemExit):
        planner.main([*args, "--output", str(tmp_path / "invalid.json"), "--laps", "bahrain-open=4"])
