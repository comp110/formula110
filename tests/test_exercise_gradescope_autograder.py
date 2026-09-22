from __future__ import annotations

import importlib.util
import json
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from racing.race.head_to_head import (
    HeadToHeadRaceResult,
    HeadToHeadResult,
    HeadToHeadTeamRaceStats,
    classify_head_to_head_winner,
)

PROJECT_ROOT = Path(__file__).parents[1]


def load_module(relative_path: str, name: str) -> ModuleType:
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def controller_source(level: int) -> str:
    if level == 0:
        throttle = "throttle = 0.1"
    elif level == 1:
        throttle = "if sensors.odometry.speed_mps < 10.0:\n        throttle = 1.0\n    else:\n        throttle = 0.0"
    elif level == 2:
        throttle = "throttle = (15.0 - sensors.odometry.speed_mps) / 15.0"
    else:
        return (
            '"""Level 3 controller."""\n\n'
            'from racing import RobotCommand, RobotSensors\n\n__author__: str = "123456789"\n'
            'RACING_NAME = "Level 3"\n\n'
            "def control(sensors: RobotSensors) -> RobotCommand:\n"
            '    """Use the camera to steer around the track."""\n'
            "    throttle = 0.8\n"
            "    steer = sensors.camera.heading_error_degrees / 45.0\n"
            "    return RobotCommand(throttle=throttle, steer=steer)\n"
        )
    return (
        f'"""Level {level} controller."""\n\n'
        'from racing import RobotCommand, RobotSensors\n\n__author__: str = "123456789"\n'
        f'RACING_NAME = "Level {level}"\n\n'
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        '    """Use wall sensors to steer around the track."""\n'
        "    throttle = 0.0\n"
        "    steer = 0.0\n"
        "    if sensors.wall_lidar.front_left_m < 4.0:\n"
        "        steer = 1.0\n"
        "    elif sensors.wall_lidar.front_right_m < 4.0:\n"
        "        steer = -1.0\n"
        f"    {throttle}\n"
        "    return RobotCommand(throttle=throttle, steer=steer)\n"
    )


def rename_controller_variables(source: str, throttle_name: str = "power", steer_name: str = "turn") -> str:
    """Rename fixture identifiers while preserving the RobotCommand parameter order."""
    source = source.replace("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(throttle, steer)")
    names = {"throttle": throttle_name, "steer": steer_name, "sensors": "readings"}
    return re.sub(r"\b(throttle|steer|sensors)\b", lambda match: names[match.group()], source)


def head_to_head_fixture(score_pairs: tuple[tuple[float, float], ...]) -> dict[str, object]:
    """Serialize known scores through the simulator's actual result types."""
    races: list[HeadToHeadRaceResult] = []
    for index, (challenger_m, incumbent_m) in enumerate(score_pairs):
        races.append(
            HeadToHeadRaceResult(
                race_index=index,
                winner=classify_head_to_head_winner(margin_m=challenger_m - incumbent_m),
                challenger=HeadToHeadTeamRaceStats((challenger_m,), (0,), (0.0,), (0.0,)),
                incumbent=HeadToHeadTeamRaceStats((incumbent_m,), (0,), (0.0,), (0.0,)),
            )
        )
    result = HeadToHeadResult("next level", "previous level", 30.0, 1.0, tuple(races))
    return {"ok": True, **result.to_dict()}


@pytest.fixture(scope="module")
def exercise_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the actual upload bundle used by the behavioral regression tests."""
    builder = load_module("scripts/build_formula110_exercise_autograder.py", "exercise_behavior_builder")
    root = tmp_path_factory.mktemp("exercise-bundle")
    archive_path = builder.build_archive(root / "autograder.zip")
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(root / "bundle")
    return root / "bundle"


@pytest.fixture
def local_exercise_worker(exercise_bundle: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run generated test controllers with the existing local subprocess mode."""
    monkeypatch.setenv("FORMULA110_LOCAL_CONTROL", "1")
    monkeypatch.setenv("FORMULA110_CONTROL_WORKER", str(exercise_bundle / "control_worker.py"))
    monkeypatch.setenv("PYTHONPATH", str(exercise_bundle / "trusted"))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    return exercise_bundle / "progression_worker.py"


@pytest.mark.parametrize(
    "throttle_code",
    (
        "throttle = (15.0 - sensors.odometry.speed_mps) / 15.0",
        "speed = sensors.odometry.speed_mps\n    throttle = (15.0 - speed) / 15.0",
        "speed: float = sensors.odometry.speed_mps\n    throttle: float = (15 - speed) / 15",
        "speed = sensors.odometry.speed_mps\n    throttle = 1.0 - speed / 15.0",
        "maximum = 15.0\n    speed = sensors.odometry.speed_mps\n    throttle = (maximum - speed) / maximum",
        "speed = sensors.odometry.speed_mps\n    throttle = -speed / 15.0 + 1.0",
    ),
)
def test_packaged_level_2_throttle_accepts_equivalent_calculations(
    tmp_path: Path,
    local_exercise_worker: Path,
    throttle_code: str,
) -> None:
    source = controller_source(2).replace("throttle = (15.0 - sensors.odometry.speed_mps) / 15.0", throttle_code)

    result = run_packaged_worker(local_exercise_worker, tmp_path, source)

    assert result == {"ok": True, "cases_checked": 36}


def run_packaged_worker(
    worker: Path, submission: Path, source: str, mode: str = "level-2-throttle"
) -> dict[str, object]:
    module_file = submission / "controller.py"
    module_file.write_text(source, encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            str(worker),
            "--mode",
            mode,
            "--submission",
            str(submission),
            "--module-file",
            str(module_file),
        ],
        cwd=submission,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=20.0,
        check=True,
    )
    prefix = "FORMULA110_RESULT="
    result_line = next(line for line in completed.stdout.splitlines() if line.startswith(prefix))
    return cast(dict[str, object], json.loads(result_line[len(prefix) :]))


@pytest.mark.parametrize(
    ("throttle_code", "expected_error"),
    (
        ("throttle = 1.0", "At speed 7.5 m/s"),
        ("throttle = 1.0 if sensors.odometry.speed_mps < 15.0 else 0.0", "At speed 7.5 m/s"),
        ("throttle = (12.0 - sensors.odometry.speed_mps) / 12.0", "At speed 7.5 m/s"),
        ("throttle = (sensors.odometry.speed_mps - 15.0) / 15.0", "control returned throttle -1"),
        ("throttle = max(0.0, 1.0 - sensors.odometry.speed_mps / 15.0)", "At speed 22.5 m/s"),
        (
            "throttle = {0.0: 1.0, 7.5: 0.5, 15.0: 0.0, 22.5: -0.5}.get(sensors.odometry.speed_mps, 0.0)",
            "At speed 2.3 m/s",
        ),
        ("throttle = round((15.0 - sensors.odometry.speed_mps) / 15.0, 2)", "At speed 2.3 m/s"),
        (
            "throttle = (15.0 - sensors.odometry.speed_mps) / 15.0\n    throttle = 0.1",
            "control returned throttle 0.1",
        ),
        (
            "if False:\n        throttle = (15.0 - sensors.odometry.speed_mps) / 15.0",
            "control returned throttle 0",
        ),
        (
            "throttle = (15.0 - sensors.odometry.speed_mps) / 15.0\n"
            "    if sensors.wall_lidar.front_left_m < 4.0:\n        throttle = 0.0",
            "with front-left wall 1 m and front-right wall 20 m",
        ),
        ("throttle = float('nan')", "Out of range float values"),
        ("raise ValueError('cannot calculate throttle')", "ValueError: cannot calculate throttle"),
        ("while True:\n        pass", "control call exceeded 0.5 seconds"),
    ),
)
def test_packaged_level_2_throttle_rejects_incorrect_outputs_and_controller_failures(
    tmp_path: Path,
    local_exercise_worker: Path,
    throttle_code: str,
    expected_error: str,
) -> None:
    source = controller_source(2).replace("throttle = (15.0 - sensors.odometry.speed_mps) / 15.0", throttle_code)

    result = run_packaged_worker(local_exercise_worker, tmp_path, source)

    assert result["ok"] is False
    assert expected_error in str(result["error"])
    assert "Traceback" not in str(result["error"])
    assert "expected throttle" not in str(result["error"])
    assert "(15.0 -" not in str(result["error"])


def test_level_2_throttle_reports_missing_file() -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_missing_throttle")

    assert grader.validate_level_2_throttle(None, "control") == {"ok": False, "error": "module file is missing"}


@pytest.mark.parametrize(
    "source",
    (
        controller_source(0),
        rename_controller_variables(controller_source(0)),
        rename_controller_variables(controller_source(0), "steer", "throttle"),
        rename_controller_variables(controller_source(0))
        .replace("< 4.0", "< 6.5")
        .replace("power = 0.1", "power = 0.18")
        .replace("turn = 1.0", "turn = 0.7")
        .replace("turn = -1.0", "turn = -0.7"),
    ),
)
def test_packaged_level_0_accepts_variable_names_and_tuned_controls(
    tmp_path: Path, local_exercise_worker: Path, source: str
) -> None:
    result = run_packaged_worker(local_exercise_worker, tmp_path, source, "level-0-behavior")

    assert result == {"ok": True, "cases_checked": 12}


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(steer, throttle)"),
        ("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(throttle, 0.0)"),
        ("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(throttle, -steer)"),
        ("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(0.0, steer)"),
        ("throttle = 0.1", "throttle = -0.1"),
        ("throttle = 0.1", "throttle = 0.0 if sensors.odometry.speed_mps == 8.0 else 0.1"),
        ("elif sensors.wall_lidar.front_right_m", "if sensors.wall_lidar.front_right_m"),
    ),
)
def test_packaged_level_0_rejects_wrong_outputs_without_revealing_the_solution(
    tmp_path: Path, local_exercise_worker: Path, old: str, new: str
) -> None:
    source = controller_source(0).replace(old, new)

    result = run_packaged_worker(local_exercise_worker, tmp_path, source, "level-0-behavior")

    assert result["ok"] is False
    assert "control returned throttle" in str(result["error"])
    assert "Review the Level 0" in str(result["error"])
    assert "expected" not in str(result["error"])
    assert "Traceback" not in str(result["error"])


def test_level_0_behavior_reports_missing_file() -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_missing_level_0")

    assert grader.validate_level_0_behavior(None, "control") == {"ok": False, "error": "module file is missing"}


def test_build_exercise_archive_has_separate_root_files_and_100_point_config(tmp_path: Path) -> None:
    builder = load_module(
        "scripts/build_formula110_exercise_autograder.py",
        "build_formula110_exercise_autograder",
    )
    output = tmp_path / "exercise-autograder.zip"

    built = builder.build_archive(output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {
            "setup.sh",
            "run_autograder",
            "grade.py",
            "progression_worker.py",
            "race_worker.py",
            "control_worker.py",
            "config.json",
        } <= names
        assert "trusted/racing/student/api.py" in names
        config = json.loads(archive.read("config.json"))
        assert config["levels"] == {
            "0": "controllers.level_0",
            "1": "controllers.level_1",
            "2": "controllers.level_2",
            "3": "controllers.level_3",
        }
        assert sum(config["rubric"].values()) == 100.0
        assert all(config["rubric"][f"level_{level}_submission_requirements"] == 2.0 for level in range(4))
        assert config["head_to_head"]["minimum_level_1_distance_m"] == 250.0
        assert config["head_to_head"]["win_tolerance_fraction"] == 0.05
        for executable in (
            "setup.sh",
            "run_autograder",
            "progression_worker.py",
            "race_worker.py",
            "control_worker.py",
        ):
            mode = archive.getinfo(executable).external_attr >> 16
            assert mode & stat.S_IXUSR

    assert (PROJECT_ROOT / "autograder" / "gradescope" / "grade.py").is_file()


def test_export_exercise_requires_and_packages_all_four_levels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exporter = load_module("scripts/export_formula110_exercise.py", "export_formula110_exercise")
    source_root = tmp_path / "src"
    controllers_root = source_root / "controllers"
    controllers_root.mkdir(parents=True)
    (controllers_root / "__init__.py").write_text("", encoding="utf-8")
    for level in range(4):
        (controllers_root / f"level_{level}.py").write_text(controller_source(level), encoding="utf-8")
    (controllers_root / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(exporter, "CONTROLLERS_ROOT", controllers_root)
    output = tmp_path / "submission.zip"

    built = exporter.export_exercise(output)

    assert built == output.resolve()
    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
        assert {f"controllers/level_{level}.py" for level in range(4)} <= names
        assert "controllers/helper.py" in names
        manifest = json.loads(archive.read("formula110-exercise-submission.json"))
        assert manifest["exercise"] == "formula110-progression"


def test_export_exercise_reports_missing_level(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = load_module("scripts/export_formula110_exercise.py", "export_formula110_exercise_missing")
    source_root = tmp_path / "src"
    controllers_root = source_root / "controllers"
    controllers_root.mkdir(parents=True)
    monkeypatch.setattr(exporter, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(exporter, "SOURCE_ROOT", source_root)
    monkeypatch.setattr(exporter, "CONTROLLERS_ROOT", controllers_root)

    with pytest.raises(FileNotFoundError, match=r"level_0\.py"):
        exporter.export_exercise(tmp_path / "submission.zip")


@pytest.mark.parametrize("level", range(4))
def test_structure_checks_accept_the_required_level_progression(tmp_path: Path, level: int) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_level_{level}")
    module_file = tmp_path / f"level_{level}.py"
    module_file.write_text(controller_source(level), encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, level)

    assert passed, output


def test_level_1_accepts_the_reported_half_throttle_max_speed_strategy(tmp_path: Path) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_reported_level_1")
    module_file = tmp_path / "level_1.py"
    module_file.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        'RACING_NAME = "Level 1"\n'
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        '    """This is a slightly more viable self-driving controller"""\n'
        "    throttle: float\n"
        "    steer: float\n"
        "    if sensors.odometry.speed_mps < 10.0:\n"
        "        throttle = 0.5\n"
        "    else:\n"
        "        throttle = 0.1\n"
        "    if sensors.wall_lidar.front_left_m < 5.0:\n"
        "        steer = 1.0\n"
        "    elif sensors.wall_lidar.front_right_m < 5.0:\n"
        "        steer = -1.0\n"
        "    else:\n"
        "        steer = 0.0\n"
        "    return RobotCommand(throttle=throttle, steer=steer)\n",
        encoding="utf-8",
    )

    passed, output = grader.inspect_structure(module_file, 1)

    assert passed, output


@pytest.mark.parametrize("level", range(4))
@pytest.mark.parametrize("keyword_arguments", (False, True))
@pytest.mark.parametrize(("throttle_name", "steer_name"), (("power", "turn"), ("steer", "throttle")))
def test_structure_accepts_other_variable_and_sensor_parameter_names(
    tmp_path: Path, level: int, throttle_name: str, steer_name: str, keyword_arguments: bool
) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_renamed_variables")
    module_file = tmp_path / f"level_{level}.py"
    source = rename_controller_variables(controller_source(level), throttle_name, steer_name)
    if keyword_arguments:
        source = source.replace(
            f"RobotCommand({throttle_name}, {steer_name})",
            f"RobotCommand(throttle={throttle_name}, steer={steer_name})",
        )
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, level)

    assert passed, output


@pytest.mark.parametrize("constructor", ("RobotCommand", "racing.RobotCommand"))
@pytest.mark.parametrize(
    "arguments",
    (
        "throttle, steer",
        "throttle, steer=steer",
        "throttle=throttle, steer=steer",
        "steer=steer, throttle=throttle",
    ),
)
def test_level_0_accepts_positional_keyword_and_mixed_command_arguments(
    tmp_path: Path,
    constructor: str,
    arguments: str,
) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_level_0_arguments")
    module_file = tmp_path / "level_0.py"
    source = controller_source(0).replace("RobotCommand(throttle=throttle, steer=steer)", f"{constructor}({arguments})")
    if constructor.startswith("racing."):
        source = source.replace("from racing import", "import racing\nfrom racing import")
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 0)

    assert passed, output
    assert "PASS: control returns those variables in RobotCommand" in output


@pytest.mark.parametrize(
    "arguments",
    (
        "throttle, 0.0",
        "0.1, steer",
        "throttle",
        "throttle, steer, 0.0",
        "throttle, throttle=throttle, steer=steer",
        "throttle, steer=throttle",
        "unassigned_power, steer",
    ),
)
def test_level_0_requires_two_assigned_variables_in_valid_command_arguments(tmp_path: Path, arguments: str) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_level_0_wrong_arguments")
    module_file = tmp_path / "level_0.py"
    source = controller_source(0).replace("RobotCommand(throttle=throttle, steer=steer)", f"RobotCommand({arguments})")
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 0)

    assert not passed
    assert "MISSING: control returns those variables in RobotCommand" in output


@pytest.mark.parametrize("level", range(4))
def test_submission_requirements_accept_documented_typed_controllers(tmp_path: Path, level: int) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_requirements_{level}")
    module_file = tmp_path / f"level_{level}.py"
    module_file.write_text(controller_source(level), encoding="utf-8")

    passed, output = grader.inspect_submission_requirements(module_file)

    assert passed, output
    assert "PASS: a non-empty module-level docstring" in output
    assert "PASS: a module-level __author__ string with exactly 9 numerical digits" in output
    assert "PASS: control has typed shape" in output
    assert "PASS: every function parameter and return type is annotated" in output


@pytest.mark.parametrize(
    ("old", "new", "expected_problem"),
    (
        (
            '"""Level 0 controller."""\n\n',
            "",
            "MISSING: a non-empty module-level docstring",
        ),
        (
            '"123456789"',
            '"PID"',
            "MISSING: a module-level __author__ string with exactly 9 numerical digits",
        ),
        (
            '__author__: str = "123456789"\n',
            "",
            "MISSING: a module-level __author__ string with exactly 9 numerical digits",
        ),
        (
            '__author__: str = "123456789"',
            "__author__: str = 123456789",
            "MISSING: a module-level __author__ string with exactly 9 numerical digits",
        ),
        (
            "sensors: RobotSensors",
            "sensors",
            "MISSING: control has typed shape `control(RobotSensors) -> RobotCommand`",
        ),
        (
            '    """Use wall sensors to steer around the track."""\n',
            "",
            "MISSING: control has a non-empty docstring",
        ),
        (
            "def control(sensors: RobotSensors) -> RobotCommand:\n",
            "def helper(value):\n    return value\n\n\ndef control(sensors: RobotSensors) -> RobotCommand:\n",
            "function 'helper' is missing type annotations",
        ),
    ),
)
def test_submission_requirements_report_metadata_contract_and_annotation_problems(
    tmp_path: Path,
    old: str,
    new: str,
    expected_problem: str,
) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_invalid_{expected_problem[:8]}")
    module_file = tmp_path / "level_0.py"
    module_file.write_text(controller_source(0).replace(old, new, 1), encoding="utf-8")

    passed, output = grader.inspect_submission_requirements(module_file)

    assert not passed
    assert expected_problem in output


def test_submission_requirements_report_missing_and_unparseable_files(tmp_path: Path) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_unavailable_requirements")
    invalid = tmp_path / "level_0.py"
    invalid.write_text("def control(:\n", encoding="utf-8")

    missing_passed, missing_output = grader.inspect_submission_requirements(None)
    invalid_passed, invalid_output = grader.inspect_submission_requirements(invalid)

    assert not missing_passed
    assert missing_output == "The required level file is missing."
    assert not invalid_passed
    assert "SyntaxError" in invalid_output


def test_blank_results_is_a_valid_zero_score_payload() -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_blank_results")

    results = grader.blank_results("infrastructure failure")

    assert results["score"] == 0.0
    assert results["tests"] == []
    assert results["output"] == "infrastructure failure"


@pytest.mark.parametrize(
    "game_file",
    (
        "racing/__init__.py",
        "src/racing/game/app.py",
        "formula110/src/racing/assets/car.png",
        "upload/formula110/src/racing/__pycache__/main.cpython-311.pyc",
    ),
)
def test_full_project_upload_stops_before_grading_and_writes_resubmission_instructions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, game_file: str
) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_full_project_upload")
    submission = tmp_path / "submission"
    submitted_game_file = submission / game_file
    submitted_game_file.parent.mkdir(parents=True)
    submitted_game_file.write_text("raise AssertionError('submission must not execute')\n", encoding="utf-8")
    results_file = tmp_path / "results" / "results.json"
    monkeypatch.setattr(grader, "SUBMISSION_PATH", submission)
    monkeypatch.setattr(grader, "RESULTS_PATH", results_file)

    def unexpected_grading(*_args: object, **_kwargs: object) -> None:
        pytest.fail("A full-project upload must stop before any grading work")

    for function in ("read_config", "locate_module", "inspect_submission_requirements", "run_worker"):
        monkeypatch.setattr(grader, function, unexpected_grading)

    grader.main()

    result = json.loads(results_file.read_text(encoding="utf-8"))
    assert result["score"] == 0.0
    assert result["tests"] == []
    assert "No level checks or races were run" in result["output"]
    assert game_file in result["output"]
    assert "Create your Gradescope submission" in result["output"]
    assert "artifacts/formula110-exercise-submission.zip" in result["output"]
    assert "resubmit" in result["output"]
    assert "infrastructure error" not in result["output"]


@pytest.mark.parametrize("prefix", ("", "controllers/", "src/controllers/", "formula110/src/controllers/"))
def test_packaging_check_allows_controller_uploads_and_similarly_named_helpers(tmp_path: Path, prefix: str) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_controller_upload")
    for name in ("level_0.py", "level_1.py", "level_2.py", "level_3.py", "racing_strategy.py"):
        source = tmp_path / prefix / name
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text("", encoding="utf-8")
    (tmp_path / "racing").mkdir()

    assert grader.find_submitted_racing_file(tmp_path) is None


@pytest.mark.parametrize(
    "prohibited_expression",
    (
        "sensors.wall_lidar.front_left_m",
        "sensors.wall_lidar.front_right_m",
    ),
)
def test_level_3_structure_rejects_each_prohibited_sensor(
    tmp_path: Path,
    prohibited_expression: str,
) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_{prohibited_expression.rsplit('.', 1)[-1]}")
    source = controller_source(3).replace("throttle = 0.8", f"throttle = {prohibited_expression}")
    module_file = tmp_path / "level_3.py"
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 3)

    assert not passed
    assert "MISSING: the program does not use" in output


def test_level_3_structure_allows_odometry_speed_sensor(tmp_path: Path) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_level_3_speed")
    source = controller_source(3).replace(
        "throttle = 0.8",
        "throttle = (15.0 - sensors.odometry.speed_mps) / 15.0",
    )
    module_file = tmp_path / "level_3.py"
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 3)

    assert passed, output


@pytest.mark.parametrize(
    "camera_property",
    (
        "visible",
        "center_offset_m",
        "heading_error_degrees",
        "lookahead_offsets_m",
        "lookahead_distances_m",
        "competitors",
    ),
)
def test_level_3_structure_accepts_each_documented_camera_sensor(
    tmp_path: Path,
    camera_property: str,
) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_camera_{camera_property}")
    source = controller_source(3).replace("heading_error_degrees", camera_property)
    module_file = tmp_path / "level_3.py"
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 3)

    assert passed, output


def test_level_3_structure_requires_a_camera_sensor(tmp_path: Path) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_level_3_only_speed")
    source = (
        'from racing import RobotCommand, RobotSensors\n\nRACING_NAME = "Level 3"\n\n'
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    throttle = (15.0 - sensors.odometry.speed_mps) / 15.0\n"
        "    steer = 0.0\n"
        "    return RobotCommand(throttle=throttle, steer=steer)\n"
    )
    module_file = tmp_path / "level_3.py"
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 3)

    assert not passed
    assert "MISSING: control reads at least one documented sensors.camera property" in output


def test_level_3_structure_does_not_accept_a_non_camera_sensor(tmp_path: Path) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_level_3_contact")
    source = (
        'from racing import RobotCommand, RobotSensors\n\nRACING_NAME = "Level 3"\n\n'
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    throttle = (15.0 - sensors.odometry.speed_mps) / 15.0\n"
        "    steer = sensors.contact.wall\n"
        "    return RobotCommand(throttle=throttle, steer=steer)\n"
    )
    module_file = tmp_path / "level_3.py"
    module_file.write_text(source, encoding="utf-8")

    passed, output = grader.inspect_structure(module_file, 3)

    assert not passed
    assert "MISSING: control reads at least one documented sensors.camera property" in output


@pytest.mark.parametrize("passes_level_0", (True, False))
@pytest.mark.parametrize(
    ("throttle_code", "passes_throttle"),
    (
        ("speed = sensors.odometry.speed_mps\n    throttle = (15.0 - speed) / 15.0", True),
        ("throttle = 1.0 - sensors.odometry.speed_mps / 15.0", True),
        ("throttle = (12.0 - sensors.odometry.speed_mps) / 12.0", False),
        ("throttle = (15.0 - sensors.odometry.speed_mps) / 15.0\n    throttle = 0.1", False),
    ),
)
def test_grade_scores_throttle_behavior_independently_of_control_and_races(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    local_exercise_worker: Path,
    throttle_code: str,
    passes_throttle: bool,
    passes_level_0: bool,
) -> None:
    builder = load_module(
        "scripts/build_formula110_exercise_autograder.py",
        "build_formula110_exercise_autograder_grade_test",
    )
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_full_credit")
    files: dict[str, Path] = {}
    for level in range(4):
        module_file = tmp_path / f"level_{level}.py"
        source = controller_source(level)
        if level == 0:
            source = source.replace("RobotCommand(throttle=throttle, steer=steer)", "RobotCommand(throttle, steer)")
            if not passes_level_0:
                source = source.replace("RobotCommand(throttle, steer)", "RobotCommand(steer, throttle)")
        if level == 2:
            source = source.replace("throttle = (15.0 - sensors.odometry.speed_mps) / 15.0", throttle_code)
        source = rename_controller_variables(source)
        module_file.write_text(source, encoding="utf-8")
        files[f"controllers.level_{level}"] = module_file

    def worker_command(arguments: list[str]) -> list[str]:
        return [sys.executable, str(local_exercise_worker), *arguments]

    monkeypatch.setattr(grader, "worker_command", worker_command)
    monkeypatch.setattr(grader, "SUBMISSION_PATH", tmp_path)
    monkeypatch.setattr(grader, "RESULTS_PATH", tmp_path / "results.json")

    def locate_module(name: str) -> tuple[Path | None, str]:
        return files[name], f"found {files[name].name}"

    def validate_control(_module_file: Path | None, _function_name: str) -> dict[str, object]:
        return {"ok": True}

    solo_calls: list[tuple[str, float]] = []

    def passing_solo(module_file: Path, _function_name: str, **kwargs: object) -> dict[str, object]:
        solo_calls.append((module_file.name, float(cast(float, kwargs["duration_seconds"]))))
        return {
            "ok": True,
            "lap_count": 1,
            "raw_distance_m": 300.0,
            "max_speed_mps": 10.0,
            "best_lap_time_seconds": 25.0,
        }

    monkeypatch.setattr(grader, "read_config", builder.build_config)
    monkeypatch.setattr(grader, "locate_module", locate_module)
    monkeypatch.setattr(grader, "validate_control", validate_control)
    monkeypatch.setattr(grader, "run_solo", passing_solo)
    metadata = tmp_path / "submission_metadata.json"
    deadline = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
    metadata.write_text(
        json.dumps(
            {
                "created_at": (deadline - timedelta(hours=48)).isoformat(),
                "assignment": {"due_date": deadline.isoformat()},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(grader, "SUBMISSION_METADATA_PATH", metadata)

    race_challengers: list[str] = []

    def passing_race(challenger_file: Path, *_args: object, **_kwargs: object) -> dict[str, object]:
        race_challengers.append(challenger_file.name)
        return head_to_head_fixture(((275.0, 225.0),) * 3)

    monkeypatch.setattr(grader, "run_head_to_head", passing_race)

    results = grader.grade()

    expected_base_score = 100.0 - (0.0 if passes_throttle else 6.0) - (0.0 if passes_level_0 else 6.0)
    assert results["score"] == expected_base_score + 5.0
    assert sum(float(test["score"]) for test in results["tests"]) == expected_base_score
    failed_checks = [test["number"] for test in results["tests"] if test["status"] != "passed"]
    assert failed_checks == ([] if passes_level_0 else ["1.1"]) + ([] if passes_throttle else ["3.1"])
    level_0_check = next(test for test in results["tests"] if test["number"] == "1.1")
    assert "PASS: control returns those variables in RobotCommand" in level_0_check["output"]
    if passes_level_0:
        assert "12 sensor inputs checked" in level_0_check["output"]
    else:
        assert "control returned throttle" in level_0_check["output"]
    assert "expected" not in level_0_check["output"]
    throttle_check = next(test for test in results["tests"] if test["number"] == "3.1")
    if passes_throttle:
        assert "36 sensor inputs checked" in throttle_check["output"]
    else:
        assert "control returned throttle" in throttle_check["output"]
        assert "review the throttle examples" in throttle_check["output"]
    assert "expected throttle" not in throttle_check["output"]
    assert "(15.0 -" not in throttle_check["output"]
    assert [test["name"] for test in results["tests"][:4]] == [
        f"Submission requirements: Level {level}" for level in range(4)
    ]
    assert results["extra_data"]["early_submission_bonus"] == 5.0
    assert results["extra_data"]["hours_before_deadline"] == 48.0
    assert "Early submission extra credit: **+5 percentage points**" in results["output"]
    assert solo_calls == [("level_0.py", 60.0)]
    assert race_challengers == ["level_1.py", "level_2.py", "level_3.py"]


@pytest.mark.parametrize(
    ("challenger_m", "incumbent_m", "passes"),
    (
        (315.0, 300.0, True),
        (300.0, 300.0, True),
        (285.0, 300.0, True),
        (284.999999, 300.0, False),
        (284.335, 299.3, True),
        (284.334999, 299.3, False),
        (0.0, 0.0, False),
        (1.0, 0.0, True),
        (0.0, 1.0, False),
    ),
)
def test_race_win_credit_includes_the_exact_five_percent_boundary(
    challenger_m: float, incumbent_m: float, passes: bool
) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_win_tolerance")
    result = head_to_head_fixture(((challenger_m, incumbent_m),))

    passed, output = grader.assess_head_to_head(result, tolerance_fraction=0.05)

    assert passed is passes, output
    assert "5% tolerance" in output


@pytest.mark.parametrize(
    ("pairs", "passes"),
    (
        (((285.0, 300.0), (285.0, 300.0), (100.0, 300.0)), True),
        (((600.0, 300.0), (280.0, 300.0), (280.0, 300.0)), False),
        (((285.0, 300.0), (280.0, 300.0)), False),
    ),
)
def test_race_tolerance_requires_a_majority_of_individual_races(
    pairs: tuple[tuple[float, float], ...], passes: bool
) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_race_majority")

    passed, output = grader.assess_head_to_head(head_to_head_fixture(pairs), tolerance_fraction=0.05)

    assert passed is passes, output
    assert "a majority is required" in output


def test_race_win_tolerance_uses_scored_distance_after_penalties() -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_scored_distance")
    result = head_to_head_fixture(((280.0, 300.0),))
    race = cast(list[dict[str, object]], result["races"])[0]
    cast(dict[str, object], race["challenger"])["raw_distances_m"] = [500.0]

    passed, output = grader.assess_head_to_head(result, tolerance_fraction=0.05)

    assert not passed
    assert "challenger 280.0 m" in output
    assert grader.head_to_head_distances(result, "challenger") == [500.0]


@pytest.mark.parametrize(
    "result",
    (
        {"ok": False, "races": []},
        {"ok": True, "summary": {"winner": "challenger"}},
        {"ok": True, "races": []},
        {"ok": True, "races": [None]},
        {"ok": True, "races": [{"challenger": {"raw_distances_m": [100.0]}, "incumbent": {}}]},
        {"ok": True, "races": [{"challenger": {"distances_m": [True]}, "incumbent": {"distances_m": [100.0]}}]},
        {"ok": True, "races": [{"challenger": {"distances_m": [float("nan")]}, "incumbent": {"distances_m": [100.0]}}]},
        {"ok": True, "races": [{"challenger": {"distances_m": [float("inf")]}, "incumbent": {"distances_m": [100.0]}}]},
    ),
)
def test_missing_or_invalid_race_scores_cannot_earn_win_credit(result: dict[str, object]) -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_invalid_race")

    passed, _output = grader.assess_head_to_head(result, tolerance_fraction=0.05)

    assert not passed


@pytest.mark.parametrize(
    ("challenger_m", "incumbent_m", "passes_win", "passes_distance"),
    ((285.0, 300.0, True, True), (284.999999, 300.0, False, True), (190.0, 200.0, True, False)),
)
def test_packaged_grader_applies_tolerance_to_all_three_win_checks_only(
    tmp_path: Path,
    exercise_bundle: Path,
    monkeypatch: pytest.MonkeyPatch,
    challenger_m: float,
    incumbent_m: float,
    passes_win: bool,
    passes_distance: bool,
) -> None:
    grader = load_module(str(exercise_bundle / "grade.py"), "exercise_packaged_grade_win_tolerance")
    for level in range(4):
        (tmp_path / f"level_{level}.py").write_text(controller_source(level), encoding="utf-8")
    monkeypatch.setattr(grader, "CONFIG_PATH", exercise_bundle / "config.json")
    monkeypatch.setattr(grader, "SUBMISSION_PATH", tmp_path)
    monkeypatch.setattr(grader, "SUBMISSION_METADATA_PATH", tmp_path / "absent_metadata.json")

    def passing_check(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"ok": True, "cases_checked": 36}

    def passing_solo(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"ok": True, "lap_count": 1, "raw_distance_m": 300.0, "max_speed_mps": 10.0}

    def near_win(*_args: object, **_kwargs: object) -> dict[str, object]:
        return head_to_head_fixture(((challenger_m, incumbent_m),))

    for name in ("validate_control", "validate_level_0_behavior", "validate_level_2_throttle"):
        monkeypatch.setattr(grader, name, passing_check)
    monkeypatch.setattr(grader, "run_solo", passing_solo)
    monkeypatch.setattr(grader, "run_head_to_head", near_win)

    result = grader.grade()

    for check in result["tests"]:
        if check["number"] in ("2.4", "3.3", "4.3"):
            assert (check["status"] == "passed") is passes_win
            assert "Suite winner: incumbent" in check["output"]
            assert "5% tolerance" in check["output"]
        elif check["number"] == "2.3":
            assert (check["status"] == "passed") is passes_distance
        else:
            assert check["status"] == "passed"
    assert result["extra_data"]["head_to_head_win_tolerance_fraction"] == 0.05


def test_scores_round_half_up_to_a_tenth_and_promote_99_9() -> None:
    grader = load_module("autograder/exercise/grade.py", "exercise_grade_rounding")
    round_score = cast(Callable[[float], float], grader.round_score)

    assert round_score(42.44) == 42.4
    assert round_score(42.45) == 42.5
    assert round_score(99.89) == 100.0
    assert round_score(99.9) == 100.0


@pytest.mark.parametrize(
    ("hours_early", "expected_bonus"),
    (
        (48.0, 5.0),
        (47.99, 3.0),
        (24.0, 3.0),
        (23.99, 0.0),
    ),
)
def test_early_submission_extra_credit_uses_gradescope_metadata(
    tmp_path: Path,
    hours_early: float,
    expected_bonus: float,
) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_bonus_{hours_early}")
    metadata = tmp_path / "submission_metadata.json"
    deadline = datetime(2026, 9, 10, 17, 0, tzinfo=UTC)
    submitted_at = deadline - timedelta(hours=hours_early)
    metadata.write_text(
        json.dumps(
            {
                "created_at": submitted_at.isoformat(),
                "assignment": {"due_date": deadline.isoformat()},
            }
        ),
        encoding="utf-8",
    )

    bonus, actual_hours_early = grader.early_submission_bonus(metadata)

    assert bonus == expected_bonus
    assert actual_hours_early == pytest.approx(hours_early)


@pytest.mark.parametrize(
    "contents",
    (
        "not json",
        "[]",
        '{"created_at": "2026-09-01T00:00:00Z"}',
        '{"created_at": "not-a-timestamp", "assignment": {"due_date": "also-invalid"}}',
    ),
)
def test_invalid_or_incomplete_metadata_awards_no_early_bonus(tmp_path: Path, contents: str) -> None:
    grader = load_module("autograder/exercise/grade.py", f"exercise_grade_bad_metadata_{len(contents)}")
    metadata = tmp_path / "submission_metadata.json"
    metadata.write_text(contents, encoding="utf-8")

    assert grader.early_submission_bonus(metadata) == (0.0, None)
