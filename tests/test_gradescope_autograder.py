from __future__ import annotations

import importlib.util
import json
import stat
import subprocess
import zipfile
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from racing.race.progress import default_track_progress_model
from racing.race.runtime import race_spawn_poses


def load_builder() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "build_gradescope_autograder.py"
    spec = importlib.util.spec_from_file_location("build_gradescope_autograder", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_exporter() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "export_student_controllers.py"
    spec = importlib.util.spec_from_file_location("export_student_controllers", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_grader() -> ModuleType:
    path = Path(__file__).parents[1] / "autograder" / "gradescope" / "grade.py"
    spec = importlib.util.spec_from_file_location("formula110_gradescope_grade", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_race_worker() -> ModuleType:
    path = Path(__file__).parents[1] / "autograder" / "gradescope" / "race_worker.py"
    spec = importlib.util.spec_from_file_location("formula110_race_worker", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_gradescope_archive_has_required_root_files_and_config(tmp_path: Path) -> None:
    builder = load_builder()
    output = tmp_path / "autograder.zip"

    built = builder.build_archive(output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "setup.sh",
            "run_autograder",
            "grade.py",
            "race_worker.py",
            "control_worker.py",
            "config.json",
        } <= names
        assert "trusted/racing/student/api.py" in names
        assert (
            archive.read("trusted/racing/physics/engine.py")
            == (Path(__file__).parents[1] / "src" / "racing" / "physics" / "engine.py").read_bytes()
        )
        assert not any(name.startswith("trusted/racing/assets/") for name in names)
        assert all(not name.startswith("formula110-autograder/") for name in names)
        config = json.loads(archive.read("config.json"))
        assert config["submission_manifest"] == "formula110-submission.json"
        assert config["seeds"] == [110, 2026, 1893, 7656, 9340]
        assert config["duration_seconds"] == 30.0
        assert config["trial_timeout_seconds"] == 60.0
        assert config["marshal"] == {
            "stuck_seconds": 2.0,
            "distance_penalty_m": 5.0,
            "cooldown_seconds": 2.0,
        }
        assert sum(config["rubric"].values()) == 100.0
        assert config["rubric"] == {"completion_with_forward_progress": 100.0}
        for executable in ("setup.sh", "run_autograder", "race_worker.py", "control_worker.py"):
            mode = archive.getinfo(executable).external_attr >> 16
            assert mode & stat.S_IXUSR
        setup = archive.read("setup.sh").decode()
        assert "UV_PYTHON_INSTALL_DIR=/opt/formula110-python" in setup
        assert "formula110-runtime.pth" in setup


def test_gradescope_seeds_cover_each_fifth_of_the_track() -> None:
    builder = load_builder()
    seeds = builder.build_config()["seeds"]
    assert isinstance(seeds, list)
    seeds = cast(list[int], seeds)
    model = default_track_progress_model()

    starting_fifths = {
        min(
            4,
            int(
                race_spawn_poses(
                    1,
                    model=model,
                    random_seed=int(seed),
                    race_index=1,
                )[0].progress_distance_m
                / model.total_length_m
                * 5
            ),
        )
        for seed in seeds
    }

    assert starting_fifths == set(range(5))


def test_controller_worker_declares_cpu_only_1536_mib_runtime() -> None:
    worker = load_race_worker()

    assert worker.CONTROLLER_STARTUP_TIMEOUT_SECONDS == 30.0
    assert worker.COMMAND_TIMEOUT_SECONDS == 0.5
    assert worker.CONTROLLER_MEMORY_LIMIT_BYTES == 1536 * 1024 * 1024
    assert worker.CPU_ONLY_ENVIRONMENT["FORMULA110_DEVICE"] == "cpu"
    assert worker.CPU_ONLY_ENVIRONMENT["CUDA_VISIBLE_DEVICES"] == ""
    assert worker.CPU_ONLY_ENVIRONMENT["JAX_PLATFORMS"] == "cpu"


def test_race_worker_horizontal_g_and_slip_angle_helpers() -> None:
    worker = load_race_worker()

    forward_g_seconds = worker.horizontal_g_seconds_for_step(
        speed_before_mps=0.0,
        speed_after_mps=worker.STANDARD_GRAVITY_MPS2,
        heading_before_degrees=0.0,
        heading_after_degrees=0.0,
        delta_seconds=1.0,
    )
    lateral_g_seconds = worker.horizontal_g_seconds_for_step(
        speed_before_mps=worker.STANDARD_GRAVITY_MPS2,
        speed_after_mps=worker.STANDARD_GRAVITY_MPS2,
        heading_before_degrees=0.0,
        heading_after_degrees=180.0 / 3.141592653589793,
        delta_seconds=1.0,
    )

    assert forward_g_seconds == pytest.approx(1.0)
    assert lateral_g_seconds == pytest.approx(1.0)
    assert worker.absolute_slip_angle_degrees(
        velocity_x_mps=10.0,
        velocity_z_mps=0.0,
        heading_degrees=0.0,
    ) == pytest.approx(90.0)


def test_race_worker_lap_accumulator_records_trusted_metrics() -> None:
    worker = load_race_worker()
    accumulator = worker.LapTelemetryAccumulator(started_at_seconds=2.0, damage_at_start=0.1)

    accumulator.record_step(
        delta_seconds=0.5,
        wall_contact=True,
        horizontal_g_seconds=1.25,
        brake_applied=True,
        drift_distance_m=2.0,
        forward_speed_mps=12.0,
    )
    lap = accumulator.completed_lap(ended_at_seconds=12.0, damage_at_end=0.4)

    assert lap == {
        "duration_seconds": 10.0,
        "damage_delta": pytest.approx(0.3),
        "wall_contact_seconds": 0.5,
        "horizontal_g_seconds": 1.25,
        "brake_applied": True,
        "drift_distance_m": 2.0,
        "sustained_top_speed_mps": 0.0,
    }


def test_race_worker_lap_accumulator_uses_one_second_rolling_average_speed() -> None:
    worker = load_race_worker()
    accumulator = worker.LapTelemetryAccumulator(started_at_seconds=0.0, damage_at_start=0.0)

    for speed_mps in (4.0, 8.0, 12.0, 16.0, 20.0):
        accumulator.record_step(
            delta_seconds=0.25,
            wall_contact=False,
            horizontal_g_seconds=0.0,
            brake_applied=False,
            drift_distance_m=0.0,
            forward_speed_mps=speed_mps,
        )

    lap = accumulator.completed_lap(ended_at_seconds=1.25, damage_at_end=0.0)

    assert lap["sustained_top_speed_mps"] == pytest.approx(14.0)


def test_race_worker_uses_rss_limit_without_torch_hostile_address_space_cap() -> None:
    grader = load_grader()

    command = grader.worker_command([])

    assert "--cpu=25" in command
    assert "--fsize=1048576" in command
    assert "--nproc=128" in command
    assert not any(argument.startswith("--as=") for argument in command)


def test_export_selected_student_controller_uses_package_layout(tmp_path: Path) -> None:
    exporter = load_exporter()
    output = tmp_path / "submission.zip"

    built = exporter.export_controller("controllers.crash_fast", output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "controllers/__init__.py",
            "controllers/crash_fast.py",
            "formula110-submission.json",
            "pyproject.toml",
            "uv.lock",
        } <= names
        manifest = json.loads(archive.read("formula110-submission.json"))
        assert manifest == {"schema_version": 1, "controller_module": "controllers.crash_fast"}


@pytest.mark.parametrize("arguments", [[], ["controllers.one", "controllers.two"]])
def test_export_cli_requires_exactly_one_controller(arguments: list[str]) -> None:
    exporter = load_exporter()

    with pytest.raises(SystemExit):
        exporter.build_parser().parse_args(arguments)


def test_export_excludes_python_cache_and_type_marker(tmp_path: Path) -> None:
    exporter = load_exporter()
    output = tmp_path / "submission.zip"

    exporter.export_controller("controllers.crash_fast", output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert "controllers/crash_fast.py" in names
        assert not any("__pycache__" in name or name.endswith(".pyc") or name.endswith("/py.typed") for name in names)


def test_export_selected_controller_includes_complete_runtime_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = load_exporter()
    source_root = tmp_path / "src"
    controllers_root = source_root / "controllers"
    controllers_root.mkdir(parents=True)
    (controllers_root / "__init__.py").write_text("", encoding="utf-8")
    (controllers_root / "main.py").write_text("from controllers.helper import VALUE\n", encoding="utf-8")
    (controllers_root / "helper.py").write_text("VALUE: int = 1\n", encoding="utf-8")
    (controllers_root / "weights.bin").write_bytes(b"checkpoint")
    (controllers_root / "py.typed").write_text("", encoding="utf-8")
    cache = controllers_root / "__pycache__"
    cache.mkdir()
    (cache / "main.cpython-311.pyc").write_bytes(b"cache")
    output = tmp_path / "submission.zip"
    monkeypatch.setattr(exporter, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(exporter, "CONTROLLERS_ROOT", controllers_root)

    exporter.export_controller("controllers.main", output)

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "controllers/__init__.py",
            "controllers/main.py",
            "controllers/helper.py",
            "controllers/weights.bin",
            "formula110-submission.json",
            "pyproject.toml",
            "uv.lock",
        } <= names
        assert "controllers/py.typed" not in names
        assert not any("__pycache__" in name or name.endswith(".pyc") for name in names)


def test_submission_dependency_sync_uses_project_and_preserves_grader_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    grader = load_grader()
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "pyproject.toml").write_text(
        '[project]\nname = "student-controller"\nversion = "0.1.0"\n', encoding="utf-8"
    )
    (submission / "uv.lock").write_text('version = 1\nrevision = 3\nrequires-python = ">=3.11"\n', encoding="utf-8")
    commands: list[list[str]] = []

    def record_command(command: list[str], *, timeout_seconds: float) -> tuple[bool, str]:
        commands.append(command)
        return True, "Audited 4 packages"

    monkeypatch.setattr(grader, "SUBMISSION_PATH", submission)
    monkeypatch.setattr(grader, "run_command", record_command)

    synced, message = grader.sync_submission_dependencies()

    assert synced
    assert message == "found pyproject.toml; runtime dependencies synchronized"
    assert "Audited 4 packages" not in message
    assert len(commands) == 1
    command = commands[0]
    assert command[:2] == [str(grader.UV_PATH), "sync"]
    assert command[command.index("--project") + 1] == str(submission)
    assert {"--active", "--inexact", "--no-dev", "--no-install-project", "--locked"} <= set(command)


def test_failed_dependency_sync_stops_before_controller_grading(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = load_builder()
    grader = load_grader()
    controller_file = Path("/submission/controllers/target.py")

    def selected_controller(manifest_name: str) -> tuple[str, str]:
        return "controllers.target", f"{manifest_name} selects controllers.target"

    def located_controller(name: str) -> tuple[Path, str]:
        return controller_file, f"found {name}"

    def controller_must_not_run(module_file: Path | None, function_name: str) -> dict[str, object]:
        pytest.fail(f"{module_file} {function_name} ran after failed sync")

    monkeypatch.setattr(grader, "read_config", builder.build_config)
    monkeypatch.setattr(grader, "read_submission_controller", selected_controller)
    monkeypatch.setattr(grader, "locate_module", located_controller)
    monkeypatch.setattr(grader, "sync_submission_dependencies", lambda: (False, "could not install widgets"))
    monkeypatch.setattr(grader, "validate_control", controller_must_not_run)

    results = grader.grade()

    assert results["tests"][0]["status"] == "failed"
    assert "could not install widgets" in results["output"]


def test_export_student_controllers_reports_missing_module(tmp_path: Path) -> None:
    exporter = load_exporter()

    with pytest.raises(FileNotFoundError, match="not found"):
        exporter.export_controller("controllers.does_not_exist", tmp_path / "submission.zip")


def test_grader_evaluates_only_manifest_controller_and_awards_forward_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    builder = load_builder()
    grader = load_grader()
    config = builder.build_config()
    controller_file = Path("/submission/controllers/selected.py")
    requested_modules: list[str] = []

    def selected_controller(manifest_name: str) -> tuple[str, str]:
        return "controllers.selected", f"{manifest_name} selects controllers.selected"

    monkeypatch.setattr(grader, "read_config", lambda: config)
    monkeypatch.setattr(grader, "read_submission_controller", selected_controller)
    monkeypatch.setattr(grader, "sync_submission_dependencies", lambda: (True, "dependencies ready"))

    def locate_module(name: str) -> tuple[Path | None, str]:
        requested_modules.append(name)
        return controller_file, "found controllers/selected.py"

    def validate_control(module_file: Path | None, function_name: str) -> dict[str, object]:
        return {"ok": module_file == controller_file}

    monkeypatch.setattr(grader, "locate_module", locate_module)
    monkeypatch.setattr(grader, "validate_control", validate_control)

    def passing_trials(
        module_file: Path,
        function_name: str,
        seeds: list[int],
        duration_seconds: float,
        timeout_seconds: float,
        marshal_stuck_seconds: float,
        marshal_penalty_m: float,
        marshal_cooldown_seconds: float,
    ) -> list[dict[str, object]]:
        assert marshal_stuck_seconds == 2.0
        assert marshal_penalty_m == 5.0
        assert marshal_cooldown_seconds == 2.0
        return [
            {
                "ok": True,
                "seed": seed,
                "elapsed_seconds": duration_seconds,
                "raw_distance_m": 200.0,
                "partial_laps": 1.1,
                "lap_count": 1,
                "damage": 0.0,
                "survived": True,
                "wall_contact_seconds": 0.0,
                "max_speed_mps": 10.0,
                "first_lap_time_seconds": 30.0,
                "best_lap_time_seconds": 30.0,
                "laps": [
                    {
                        "duration_seconds": 30.0,
                        "damage_delta": 0.0,
                        "wall_contact_seconds": 0.0,
                        "horizontal_g_seconds": 2.0,
                        "brake_applied": False,
                        "drift_distance_m": 1.0,
                        "sustained_top_speed_mps": 10.0,
                    }
                ],
            }
            for seed in seeds
        ]

    monkeypatch.setattr(grader, "run_trials", passing_trials)

    results = grader.grade()

    assert requested_modules == ["controllers.selected"]
    assert results["tests"][0]["score"] == 100.0
    assert results["tests"][0]["status"] == "passed"
    assert "Dependencies:" not in results["output"]
    assert "dependencies ready" not in results["output"]
    assert results["extra_data"]["controller_module"] == "controllers.selected"
    assert [trial["seed"] for trial in results["extra_data"]["controller_trials"]] == [1, 2, 3, 4, 5]
    assert results["leaderboard"][0] == {"name": "All Spawns, No Crumbs (Laps)", "value": 1.1}
    assert "Leaderboard metrics:" in results["output"]
    assert "- All Spawns, No Crumbs (Laps): 1.1" in results["output"]


def test_trial_summary_prints_seed_serial_numbers() -> None:
    grader = load_grader()
    summary = grader.trial_summary(
        "controllers.selected",
        [
            {"ok": False, "seed": 110, "error": "first failure"},
            {"ok": False, "seed": 2026, "error": "second failure"},
        ],
    )

    assert "seed 1: ERROR" in summary
    assert "seed 2: ERROR" in summary
    assert "110" not in summary
    assert "2026" not in summary


def test_trial_summary_prints_top_speed_in_mph() -> None:
    grader = load_grader()
    summary = grader.trial_summary(
        "controllers.selected",
        [
            {
                "ok": True,
                "elapsed_seconds": 30.0,
                "partial_laps": 1.0,
                "raw_distance_m": 100.0,
                "max_speed_mps": 10.0,
                "damage": 0.0,
                "wall_contact_seconds": 0.0,
            }
        ],
    )

    assert "top 22.37 mph" in summary
    assert "m/s" not in summary


def test_recommended_leaderboard_metrics_select_and_aggregate_laps() -> None:
    grader = load_grader()

    def lap(
        duration: float,
        damage: float,
        wall_contact: float,
        g_seconds: float,
        brake_applied: bool,
        drift_distance: float,
        sustained_top_speed: float,
        *,
        off_track_seconds: float = 0.0,
    ) -> dict[str, object]:
        return {
            "duration_seconds": duration,
            "damage_delta": damage,
            "wall_contact_seconds": wall_contact,
            "horizontal_g_seconds": g_seconds,
            "brake_applied": brake_applied,
            "drift_distance_m": drift_distance,
            "sustained_top_speed_mps": sustained_top_speed,
            "off_track_seconds": off_track_seconds,
        }

    trials = [
        {
            "ok": True,
            "elapsed_seconds": 30.0,
            "raw_distance_m": 256.0,
            "partial_laps": 1.4,
            "damage": 0.4,
            "survived": True,
            "laps": [
                lap(12.0, 0.0, 0.0, 3.0, False, 2.0, 20.0, off_track_seconds=8.0),
                lap(10.0, 0.03, 0.1, 9.0, True, 4.0, 30.0),
                lap(11.0, 0.01, 0.0, 6.0, True, 3.0, 25.0),
            ],
        },
        {
            "ok": True,
            "elapsed_seconds": 30.0,
            "raw_distance_m": 220.0,
            "partial_laps": 1.2,
            "damage": 0.5,
            "survived": True,
            "laps": [
                lap(14.0, 0.0, 0.0, 4.0, False, 1.0, 22.0),
                lap(9.0, 0.05, 0.2, 8.0, True, 5.0, 28.0),
                lap(13.0, 0.01, 0.0, 10.0, True, 7.0, 35.0),
            ],
        },
    ]

    leaderboard, message = grader.leaderboard_for(trials, 30.0)

    assert leaderboard == [
        {"name": "All Spawns, No Crumbs (Laps)", "value": 1.2},
        {"name": "Clock It (s)", "value": 13.0, "order": "asc"},
        {"name": "Hits Different (damage %)", "value": 90.0},
        {"name": "Sips Tea (g-s)", "value": 3.5, "order": "asc"},
        {"name": "Gs Going Crazy (g-s)", "value": 8.0},
        {"name": "Gas Locked In (s)", "value": 13.0, "order": "asc"},
        {"name": "Serving Sideways (m)", "value": 5.5},
        {"name": "Speedmaxxing (mph)", "value": 72.7},
    ]
    assert "2 starting offsets" in message


def test_leaderboard_report_prints_unavailable_metrics_as_na() -> None:
    grader = load_grader()

    report = grader.leaderboard_report(
        [
            {"name": "All Spawns, No Crumbs (Laps)", "value": 1.2345},
            {"name": "Speedmaxxing (mph)", "value": 72.7},
        ],
        "Leaderboard: qualified; metrics aggregate 5 starting offsets.",
    )

    assert report == (
        "Leaderboard: qualified; metrics aggregate 5 starting offsets.\n\n"
        "Leaderboard metrics:\n"
        "- All Spawns, No Crumbs (Laps): 1.2345\n"
        "- Clock It (s): N/A\n"
        "- Hits Different (damage %): N/A\n"
        "- Sips Tea (g-s): N/A\n"
        "- Gs Going Crazy (g-s): N/A\n"
        "- Gas Locked In (s): N/A\n"
        "- Serving Sideways (m): N/A\n"
        "- Speedmaxxing (mph): 72.7"
    )


def test_missing_eligible_laps_do_not_receive_false_winning_values() -> None:
    grader = load_grader()
    trial = {
        "ok": True,
        "elapsed_seconds": 30.0,
        "raw_distance_m": 190.0,
        "partial_laps": 1.04,
        "damage": 0.2,
        "survived": True,
        "laps": [
            {
                "duration_seconds": 20.0,
                "damage_delta": 0.2,
                "wall_contact_seconds": 0.5,
                "horizontal_g_seconds": 4.0,
                "brake_applied": True,
                "drift_distance_m": 0.0,
                "sustained_top_speed_mps": 20.0,
            }
        ],
    }

    leaderboard, _message = grader.leaderboard_for([trial], 30.0)
    values = {entry["name"]: entry["value"] for entry in leaderboard}

    assert "Clock It (s)" not in values
    assert "Gs Going Crazy (g-s)" not in values
    assert "Gas Locked In (s)" not in values
    assert values["Hits Different (damage %)"] == 20.0
    assert values["Sips Tea (g-s)"] == 4.0
    assert values["Serving Sideways (m)"] == 0.0
    assert values["Speedmaxxing (mph)"] == 44.739
    assert all(isinstance(value, (int, float)) for value in values.values())


def test_qualified_controller_without_laps_still_appears_on_leaderboard() -> None:
    grader = load_grader()
    trials: list[dict[str, object]] = [
        {
            "ok": True,
            "elapsed_seconds": 30.0,
            "raw_distance_m": 10.0 + seed_number,
            "partial_laps": 0.0,
            "damage": 0.01,
            "survived": True,
            "laps": [],
        }
        for seed_number in range(5)
    ]

    leaderboard, message = grader.leaderboard_for(trials, 30.0)

    assert message == "Leaderboard: qualified; metrics aggregate 5 starting offsets."
    assert leaderboard == [
        {"name": "All Spawns, No Crumbs (Laps)", "value": 0.0},
        {"name": "Hits Different (damage %)", "value": 5.0},
    ]


def test_submission_manifest_names_exactly_one_controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grader = load_grader()
    submission = tmp_path / "submission"
    submission.mkdir()
    manifest = submission / "formula110-submission.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "controller_module": "controllers.chosen"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(grader, "SUBMISSION_PATH", submission)

    controller, message = grader.read_submission_controller("formula110-submission.json")

    assert controller == "controllers.chosen"
    assert "controllers.chosen" in message


def test_submission_manifest_rejects_multiple_controllers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    grader = load_grader()
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "formula110-submission.json").write_text(
        json.dumps({"schema_version": 1, "controller_module": ["controllers.one", "controllers.two"]}),
        encoding="utf-8",
    )
    monkeypatch.setattr(grader, "SUBMISSION_PATH", submission)

    controller, message = grader.read_submission_controller("formula110-submission.json")

    assert controller is None
    assert "valid controllers.* module" in message


def test_zero_progress_does_not_qualify_for_check_or_leaderboard() -> None:
    grader = load_grader()
    trial = {
        "ok": True,
        "elapsed_seconds": 30.0,
        "raw_distance_m": 0.0,
        "partial_laps": 0.0,
        "max_speed_mps": 0.0,
        "damage": 0.0,
        "survived": True,
        "first_lap_time_seconds": None,
        "best_lap_time_seconds": None,
    }

    assert not grader.completed_with_forward_progress(trial, 30.0)
    leaderboard, message = grader.leaderboard_for([trial], 30.0)
    assert leaderboard == []
    assert "DQ" in message


def test_controller_startup_diagnostic_includes_child_exit_and_stderr() -> None:
    module = load_race_worker()
    client = module.ControllerClient(
        submission=Path("/submission"), module_file=Path("/submission/controller.py"), function_name="control"
    )
    client.process = subprocess.Popen(
        ["bash", "-c", "echo interpreter-permission-denied >&2; exit 126"],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    client.process.wait(timeout=2.0)

    message = client._unexpected_exit_message()

    assert "exit 126" in message
    assert "interpreter-permission-denied" in message


def test_controller_startup_has_separate_timeout_from_control_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_race_worker()
    controller_file = tmp_path / "slow_start.py"
    controller_file.write_text(
        "import time\n"
        "time.sleep(0.75)\n"
        "from racing import RobotCommand\n"
        "def control(sensors):\n"
        "    return RobotCommand(throttle=0.25, steer=0.0)\n",
        encoding="utf-8",
    )
    control_worker = Path(__file__).parents[1] / "autograder" / "gradescope" / "control_worker.py"
    monkeypatch.setenv("FORMULA110_LOCAL_CONTROL", "1")
    monkeypatch.setattr(module, "CONTROL_WORKER_PATH", control_worker)

    from racing import RobotCommand, RobotSensors

    with module.ControllerClient(
        submission=tmp_path,
        module_file=controller_file,
        function_name="control",
    ) as client:
        assert client.command(RobotSensors()) == RobotCommand(throttle=0.25, steer=0.0)


def test_controller_tick_still_has_half_second_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = load_race_worker()
    controller_file = tmp_path / "slow_tick.py"
    controller_file.write_text(
        "import time\n"
        "from racing import RobotCommand\n"
        "def control(sensors):\n"
        "    time.sleep(0.75)\n"
        "    return RobotCommand(throttle=0.25, steer=0.0)\n",
        encoding="utf-8",
    )
    control_worker = Path(__file__).parents[1] / "autograder" / "gradescope" / "control_worker.py"
    monkeypatch.setenv("FORMULA110_LOCAL_CONTROL", "1")
    monkeypatch.setattr(module, "CONTROL_WORKER_PATH", control_worker)

    from racing import RobotSensors

    with (
        module.ControllerClient(
            submission=tmp_path,
            module_file=controller_file,
            function_name="control",
        ) as client,
        pytest.raises(TimeoutError, match=r"control call exceeded 0\.5 seconds"),
    ):
        client.command(RobotSensors())
