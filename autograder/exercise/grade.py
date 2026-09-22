#!/usr/bin/env python3
"""Trusted Gradescope driver for the four-level Formula 110 exercise."""

from __future__ import annotations

import ast
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, cast

AUTOGRADER_ROOT = Path("/autograder")
CONFIG_PATH = Path("/opt/formula110-exercise-autograder/config.json")
WORKER_PATH = Path("/opt/formula110-exercise-autograder/progression_worker.py")
CONTROL_WORKER_PATH = Path("/opt/formula110-exercise-autograder/control_worker.py")
RUNTIME_PATH = Path("/opt/formula110-runtime")
VENV_PATH = AUTOGRADER_ROOT / "venv"
SUBMISSION_PATH = AUTOGRADER_ROOT / "submission"
RESULTS_PATH = AUTOGRADER_ROOT / "results" / "results.json"
SUBMISSION_METADATA_PATH = AUTOGRADER_ROOT / "submission_metadata.json"
RESULT_PREFIX = "FORMULA110_RESULT="
MAX_DIAGNOSTIC_CHARS = 6000
FORTY_EIGHT_HOUR_BONUS = 5.0
TWENTY_FOUR_HOUR_BONUS = 3.0
_MISSING = object()

FRONT_LEFT = ("sensors", "wall_lidar", "front_left_m")
FRONT_RIGHT = ("sensors", "wall_lidar", "front_right_m")
SPEED = ("sensors", "odometry", "speed_mps")
PROHIBITED_LEVEL_3_SENSORS = {FRONT_LEFT, FRONT_RIGHT}
LEVEL_3_CAMERA_SENSORS = {
    ("sensors", "camera", "visible"),
    ("sensors", "camera", "center_offset_m"),
    ("sensors", "camera", "heading_error_degrees"),
    ("sensors", "camera", "lookahead_offsets_m"),
    ("sensors", "camera", "lookahead_distances_m"),
    ("sensors", "camera", "competitors"),
}


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_results(results: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(RESULTS_PATH)


def blank_results(message: str) -> dict[str, Any]:
    return {
        "score": 0.0,
        "output": message,
        "output_format": "text",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": [],
    }


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed


def early_submission_bonus(metadata: Path | None) -> tuple[float, float | None]:
    """Return the Gradescope early-submission bonus and hours before the due date."""
    if metadata is None or not metadata.is_file():
        return 0.0, None
    try:
        content: object = json.loads(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0.0, None
    if not isinstance(content, dict):
        return 0.0, None
    metadata_content = cast(dict[str, object], content)
    assignment_value = metadata_content.get("assignment")
    if not isinstance(assignment_value, dict):
        return 0.0, None
    assignment = cast(dict[str, object], assignment_value)
    submitted_at = _timestamp(metadata_content.get("created_at"))
    due_at = _timestamp(assignment.get("due_date"))
    if submitted_at is None or due_at is None:
        return 0.0, None

    hours_early = (due_at - submitted_at).total_seconds() / 3600.0
    if hours_early >= 48.0:
        return FORTY_EIGHT_HOUR_BONUS, hours_early
    if hours_early >= 24.0:
        return TWENTY_FOUR_HOUR_BONUS, hours_early
    return 0.0, hours_early


def round_score(points: float) -> float:
    """Round points half-up to one decimal place, promoting 99.9 to 100."""
    rounded = Decimal(str(points)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if rounded == Decimal("99.9"):
        rounded = Decimal("100.0")
    return float(rounded)


def find_submitted_racing_file(submission: Path) -> Path | None:
    """Find a simulator file without reading or executing submitted code."""
    for directory, _subdirectories, filenames in os.walk(submission, followlinks=False):
        relative = Path(directory).relative_to(submission)
        if "racing" in relative.parts and filenames:
            return relative / filenames[0]
    return None


def locate_module(module_name: str) -> tuple[Path | None, str]:
    relative = Path(*module_name.split(".")).with_suffix(".py")
    direct_candidates = [SUBMISSION_PATH / relative, SUBMISSION_PATH / "src" / relative]
    for candidate in direct_candidates:
        if candidate.is_file() and not candidate.is_symlink():
            return candidate.resolve(), f"found {candidate.relative_to(SUBMISSION_PATH)}"

    matches = sorted(
        path.resolve() for path in SUBMISSION_PATH.rglob(relative.name) if path.is_file() and not path.is_symlink()
    )
    if len(matches) == 1:
        return matches[0], f"found flattened upload {matches[0].relative_to(SUBMISSION_PATH)}"
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(SUBMISSION_PATH)) for path in matches[:8])
        return None, f"multiple possible files named {relative.name}: {names}"
    return None, f"expected {relative} (or src/{relative}) in the submission"


def dotted_name(node: ast.AST) -> tuple[str, ...] | None:
    """Return the components of a simple attribute chain."""
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    return tuple(reversed(parts))


def attribute_paths(tree: ast.AST, sensor_name: str = "sensors") -> set[tuple[str, ...]]:
    return {
        ("sensors", *path[1:]) if path[0] == sensor_name else path
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and (path := dotted_name(node)) is not None
    }


def control_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for statement in tree.body:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)) and statement.name == "control":
            return statement
    return None


def _module_author(tree: ast.Module) -> object:
    author: object = _MISSING
    for statement in tree.body:
        value: ast.expr | None = None
        assigns_author = False
        if isinstance(statement, ast.Assign):
            assigns_author = any(
                isinstance(target, ast.Name) and target.id == "__author__" for target in statement.targets
            )
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            assigns_author = isinstance(statement.target, ast.Name) and statement.target.id == "__author__"
            value = statement.value
        if assigns_author:
            author = value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None
    return author


def _annotation_text(annotation: ast.expr | None) -> str | None:
    return ast.unparse(annotation) if annotation is not None else None


def _has_required_control_signature(function: ast.FunctionDef | ast.AsyncFunctionDef | None) -> bool:
    if not isinstance(function, ast.FunctionDef):
        return False
    arguments = function.args
    has_unsupported_parameters = bool(
        arguments.posonlyargs
        or arguments.vararg
        or arguments.kwonlyargs
        or arguments.kwarg
        or arguments.defaults
        or any(default is not None for default in arguments.kw_defaults)
    )
    parameter_types = tuple(_annotation_text(parameter.annotation) for parameter in arguments.args)
    return (
        not has_unsupported_parameters
        and parameter_types == ("RobotSensors",)
        and _annotation_text(function.returns) == "RobotCommand"
    )


def _function_parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[ast.arg, ...]:
    positional = (*function.args.posonlyargs, *function.args.args)
    if positional and positional[0].arg in {"self", "cls"}:
        positional = positional[1:]
    variadic = tuple(parameter for parameter in (function.args.vararg, function.args.kwarg) if parameter is not None)
    return (*positional, *variadic, *function.args.kwonlyargs)


def _annotation_problems(tree: ast.Module) -> list[str]:
    problems: list[str] = []
    functions = sorted(
        (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))),
        key=lambda function: function.lineno,
    )
    for function in functions:
        missing_parameters = [
            parameter.arg for parameter in _function_parameters(function) if parameter.annotation is None
        ]
        missing_parts: list[str] = []
        if missing_parameters:
            label = "parameter" if len(missing_parameters) == 1 else "parameters"
            names = ", ".join(f"{name!r}" for name in missing_parameters)
            missing_parts.append(f"{label} {names}")
        if function.returns is None:
            missing_parts.append("return type")
        if missing_parts:
            problems.append(
                f"line {function.lineno}, function {function.name!r} is missing type annotations for "
                + " and ".join(missing_parts)
            )
    return problems


def inspect_submission_requirements(module_file: Path | None) -> tuple[bool, str]:
    """Check the common COMP110 documentation, authorship, and typing requirements."""
    if module_file is None:
        return False, "The required level file is missing."
    try:
        source = module_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_file))
    except (OSError, SyntaxError, UnicodeError) as error:
        return False, f"Could not inspect the program: {type(error).__name__}: {error}"

    control = control_function(tree)
    author = _module_author(tree)
    annotation_problems = _annotation_problems(tree)
    annotation_description = "every function parameter and return type is annotated"
    if annotation_problems:
        annotation_description += "; " + "; ".join(annotation_problems)
    return _format_checklist(
        [
            (
                "a non-empty module-level docstring is the first statement",
                bool((ast.get_docstring(tree, clean=False) or "").strip()),
            ),
            (
                "a module-level __author__ string with exactly 9 numerical digits",
                isinstance(author, str) and re.fullmatch(r"[0-9]{9}", author) is not None,
            ),
            (
                "control has typed shape `control(RobotSensors) -> RobotCommand`",
                _has_required_control_signature(control),
            ),
            (
                "control has a non-empty docstring",
                control is not None and bool((ast.get_docstring(control, clean=False) or "").strip()),
            ),
            (annotation_description, not annotation_problems),
        ]
    )


def racing_name(tree: ast.Module) -> str | None:
    for statement in tree.body:
        value: ast.AST | None = None
        if isinstance(statement, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "RACING_NAME" for target in statement.targets
        ):
            value = statement.value
        if (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.target.id == "RACING_NAME"
        ):
            value = statement.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return value.value
    return None


def assigned_names(function: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(function) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def speed_appears_in_comparison(function: ast.AST, sensor_name: str = "sensors") -> bool:
    return any(
        isinstance(node, ast.Compare) and SPEED in attribute_paths(node, sensor_name) for node in ast.walk(function)
    )


def robot_command_uses_variables(function: ast.AST) -> bool:
    names = assigned_names(function)
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        called = dotted_name(node.func)
        if called not in {("RobotCommand",), ("racing", "RobotCommand")}:
            continue
        if len(node.args) > 2:
            continue
        positional = dict(zip(("throttle", "steer"), node.args, strict=False))
        if any(keyword.arg not in {"throttle", "steer"} or keyword.arg in positional for keyword in node.keywords):
            continue
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
        if len(keywords) != len(node.keywords):
            continue
        arguments = {**positional, **keywords}
        throttle_value = arguments.get("throttle")
        steer_value = arguments.get("steer")
        if (
            isinstance(throttle_value, ast.Name)
            and throttle_value.id in names
            and isinstance(steer_value, ast.Name)
            and steer_value.id in names
            and throttle_value.id != steer_value.id
        ):
            return True
    return False


def _format_checklist(checks: list[tuple[str, bool]]) -> tuple[bool, str]:
    lines = [f"{'PASS' if passed else 'MISSING'}: {description}" for description, passed in checks]
    return all(passed for _description, passed in checks), "\n".join(lines)


def inspect_structure(module_file: Path | None, level: int) -> tuple[bool, str]:
    if module_file is None:
        return False, "The required level file is missing."
    try:
        source = module_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(module_file))
    except (OSError, SyntaxError, UnicodeError) as error:
        return False, f"Could not inspect the program: {type(error).__name__}: {error}"

    function = control_function(tree)
    if function is None:
        return False, "MISSING: a top-level control function"
    sensor_name = function.args.args[0].arg if function.args.args else "sensors"
    paths = attribute_paths(function, sensor_name)
    names = assigned_names(function)
    expected_name = f"Level {level}"

    if level == 0:
        return _format_checklist(
            [
                (f'RACING_NAME is "{expected_name}"', racing_name(tree) == expected_name),
                ("control assigns at least two variables (names are your choice)", len(names) >= 2),
                ("control returns those variables in RobotCommand", robot_command_uses_variables(function)),
                (
                    "control contains a conditional statement",
                    any(isinstance(node, ast.If) for node in ast.walk(function)),
                ),
                ("control reads the forward-left wall sensor", FRONT_LEFT in paths),
                ("control reads the forward-right wall sensor", FRONT_RIGHT in paths),
            ]
        )
    if level == 1:
        return _format_checklist(
            [
                (f'RACING_NAME is "{expected_name}"', racing_name(tree) == expected_name),
                (
                    "control uses variables for throttle and steering (names are your choice)",
                    robot_command_uses_variables(function),
                ),
                ("control reads both forward wall sensors", {FRONT_LEFT, FRONT_RIGHT} <= paths),
                (
                    "control compares the odometry speed in a conditional",
                    speed_appears_in_comparison(function, sensor_name),
                ),
            ]
        )
    if level == 2:
        return _format_checklist(
            [
                (f'RACING_NAME is "{expected_name}"', racing_name(tree) == expected_name),
                (
                    "control uses variables for throttle and steering (names are your choice)",
                    robot_command_uses_variables(function),
                ),
                ("control reads both forward wall sensors", {FRONT_LEFT, FRONT_RIGHT} <= paths),
            ]
        )

    prohibited_used = PROHIBITED_LEVEL_3_SENSORS & paths
    camera_sensors_used = LEVEL_3_CAMERA_SENSORS & paths
    return _format_checklist(
        [
            (f'RACING_NAME is "{expected_name}"', racing_name(tree) == expected_name),
            (
                "the program does not use the prohibited front_left_m or front_right_m wall sensors",
                not prohibited_used,
            ),
            (
                "control reads at least one documented sensors.camera property",
                bool(camera_sensors_used),
            ),
        ]
    )


def worker_command(arguments: list[str]) -> list[str]:
    prlimit = shutil.which("prlimit") or "/usr/bin/prlimit"
    return [
        prlimit,
        "--cpu=50",
        "--as=3221225472",
        "--fsize=1048576",
        "--nproc=128",
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "PATH=/autograder/venv/bin:/usr/bin:/bin",
        "PYTHONPATH=/opt/formula110-runtime:/opt/formula110-exercise-autograder",
        f"FORMULA110_CONTROL_WORKER={CONTROL_WORKER_PATH}",
        str(VENV_PATH / "bin" / "python"),
        str(WORKER_PATH),
        *arguments,
    ]


def run_worker(arguments: list[str], *, timeout_seconds: float) -> dict[str, Any]:
    output_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=RESULTS_PATH.parent, prefix="worker-", delete=False) as output_file:
            output_path = Path(output_file.name)
            try:
                completed = subprocess.run(
                    worker_command(arguments),
                    cwd=SUBMISSION_PATH,
                    stdin=subprocess.DEVNULL,
                    stdout=output_file,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"ok": False, "error": f"timed out after {timeout_seconds:.0f} seconds"}

        output = output_path.read_bytes()[-1_048_576:].decode("utf-8", errors="replace")
        payload_line = next((line for line in reversed(output.splitlines()) if line.startswith(RESULT_PREFIX)), None)
        if payload_line is None:
            diagnostic = output[-2000:].strip()
            return {
                "ok": False,
                "error": f"worker exited {completed.returncode} without a result",
                "diagnostic": diagnostic,
            }
        payload: object = json.loads(payload_line[len(RESULT_PREFIX) :])
        if isinstance(payload, dict):
            return cast(dict[str, Any], payload)
        return {"ok": False, "error": "worker result was not an object"}
    except (OSError, json.JSONDecodeError) as error:
        return {"ok": False, "error": f"worker failure: {type(error).__name__}: {error}"}
    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def validate_control(module_file: Path | None, function_name: str) -> dict[str, Any]:
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--mode",
            "validate",
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
        ],
        timeout_seconds=10.0,
    )


def validate_level_0_behavior(module_file: Path | None, function_name: str) -> dict[str, Any]:
    """Check Level 0 wall responses in the isolated controller process."""
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--mode",
            "level-0-behavior",
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
        ],
        timeout_seconds=10.0,
    )


def validate_level_2_throttle(module_file: Path | None, function_name: str) -> dict[str, Any]:
    """Check returned throttle values in the isolated controller process."""
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--mode",
            "level-2-throttle",
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
        ],
        timeout_seconds=20.0,
    )


def run_solo(
    module_file: Path | None,
    function_name: str,
    *,
    seed: int,
    duration_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--mode",
            "solo",
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
            "--seed",
            str(seed),
            "--seconds",
            str(duration_seconds),
        ],
        timeout_seconds=timeout_seconds,
    )


def run_head_to_head(
    challenger_file: Path | None,
    incumbent_file: Path | None,
    function_name: str,
    *,
    seed: int,
    race_count: int,
    round_seconds: float,
    win_margin_m: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    if challenger_file is None or incumbent_file is None:
        return {"ok": False, "error": "one or both level files are missing"}
    return run_worker(
        [
            "--mode",
            "head-to-head",
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(challenger_file),
            "--opponent-module-file",
            str(incumbent_file),
            "--function",
            function_name,
            "--seed",
            str(seed),
            "--seconds",
            str(round_seconds),
            "--races",
            str(race_count),
            "--win-margin-m",
            str(win_margin_m),
        ],
        timeout_seconds=timeout_seconds,
    )


def test_case(name: str, passed: bool, points: float, output: str, number: str) -> dict[str, Any]:
    return {
        "name": name,
        "number": number,
        "score": points if passed else 0.0,
        "max_score": points,
        "status": "passed" if passed else "failed",
        "output": output[-MAX_DIAGNOSTIC_CHARS:],
        "visibility": "visible",
    }


def validation_output(module_name: str, result: dict[str, Any]) -> str:
    if result.get("ok") is True:
        return f"{module_name}.control loaded, accepted RobotSensors, and returned RobotCommand."
    return f"{module_name}: {result.get('error', 'validation failed')}"


def level_0_behavior_output(result: dict[str, Any]) -> str:
    if result.get("ok") is True:
        return (
            "PASS: returned commands follow the Level 0 wall-sensor strategy "
            f"({result['cases_checked']} sensor inputs checked)."
        )
    return f"FAIL: Level 0 wall-sensor behavior: {result.get('error', 'check failed')}"


def level_2_throttle_output(result: dict[str, Any]) -> str:
    if result.get("ok") is True:
        return (
            "PASS: returned throttle matches the Level 2 speed-scaling requirements "
            f"({result['cases_checked']} sensor inputs checked)."
        )
    return f"FAIL: proportional throttle behavior: {result.get('error', 'check failed')}"


def solo_summary(module_name: str, result: dict[str, Any]) -> str:
    if result.get("ok") is not True:
        return f"{module_name}: ERROR — {result.get('error', 'trial failed')}"
    lap_time = result.get("best_lap_time_seconds")
    lap_time_summary = (
        f"best lap {float(lap_time):.2f} s"
        if isinstance(lap_time, (int, float)) and not isinstance(lap_time, bool)
        else "no completed lap time"
    )
    return (
        f"{module_name}: {int(result['lap_count'])} completed lap(s), "
        f"{float(result['raw_distance_m']):.1f} m forward progress, "
        f"top speed {float(result['max_speed_mps']):.2f} m/s, "
        f"{lap_time_summary}"
    )


def head_to_head_distances(result: dict[str, Any], role: str) -> list[float]:
    if result.get("ok") is not True:
        return []
    distances: list[float] = []
    races_value: object = result.get("races", [])
    if not isinstance(races_value, list):
        return distances
    races = cast(list[object], races_value)
    for race_value in races:
        if not isinstance(race_value, dict):
            continue
        race = cast(dict[str, Any], race_value)
        role_value: object = race.get(role, {})
        if not isinstance(role_value, dict):
            continue
        role_result = cast(dict[str, Any], role_value)
        raw_distances_value: object = role_result.get("raw_distances_m", [])
        if not isinstance(raw_distances_value, list):
            continue
        raw_distances = cast(list[object], raw_distances_value)
        numeric_distances = [
            float(distance)
            for distance in raw_distances
            if isinstance(distance, (int, float)) and not isinstance(distance, bool)
        ]
        if numeric_distances:
            distances.append(max(numeric_distances))
    return distances


def head_to_head_summary(challenger_name: str, incumbent_name: str, result: dict[str, Any]) -> str:
    if result.get("ok") is not True:
        return f"{challenger_name} vs {incumbent_name}: ERROR — {result.get('error', 'race failed')}"
    summary_value: object = result.get("summary", {})
    summary = cast(dict[str, Any], summary_value) if isinstance(summary_value, dict) else {}
    winner = summary.get("winner", "unknown")
    challenger_wins = summary.get("challenger_wins", "?")
    incumbent_wins = summary.get("incumbent_wins", "?")
    ties = summary.get("ties", "?")
    challenger_distances = head_to_head_distances(result, "challenger")
    incumbent_distances = head_to_head_distances(result, "incumbent")
    lines = [
        f"Suite winner: {winner}; record {challenger_wins}-{incumbent_wins}-{ties} "
        f"({challenger_name}-{incumbent_name}-ties)."
    ]
    for index, (challenger_m, incumbent_m) in enumerate(
        zip(challenger_distances, incumbent_distances, strict=False),
        start=1,
    ):
        lines.append(f"Race {index}: {challenger_name} {challenger_m:.1f} m; {incumbent_name} {incumbent_m:.1f} m")
    return "\n".join(lines)


def race_scored_distances(race_value: object) -> tuple[Decimal, Decimal] | None:
    """Read both scores from the same race, preserving its scoring rule."""
    if not isinstance(race_value, dict):
        return None
    race = cast(dict[str, Any], race_value)
    scoring = race.get("scoring", "team-sum")
    if scoring not in ("team-sum", "best-copy"):
        return None
    scores: list[Decimal] = []
    for role in ("challenger", "incumbent"):
        team_value: object = race.get(role)
        if not isinstance(team_value, dict):
            return None
        team = cast(dict[str, Any], team_value)
        distances_value: object = team.get("distances_m")
        if not isinstance(distances_value, list) or not distances_value:
            return None
        distances: list[Decimal] = []
        for value in cast(list[object], distances_value):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                return None
            distance = Decimal(str(value))
            if not distance.is_finite() or distance < 0:
                return None
            distances.append(distance)
        scores.append(max(distances) if scoring == "best-copy" else sum(distances, Decimal(0)))
    return scores[0], scores[1]


def assess_head_to_head(result: dict[str, Any], *, tolerance_fraction: float) -> tuple[bool, str]:
    """Award race credit for a majority of finishes within the configured tolerance."""
    tolerance = Decimal(str(tolerance_fraction))
    if not tolerance.is_finite() or not Decimal(0) <= tolerance < Decimal(1):
        raise ValueError("win_tolerance_fraction must be between 0 (inclusive) and 1 (exclusive)")
    races_value: object = result.get("races")
    if result.get("ok") is not True or not isinstance(races_value, list) or not races_value:
        return False, "Head-to-head grading: no completed race results available."
    races = cast(list[object], races_value)
    credited_wins = 0
    lines: list[str] = []
    for index, race in enumerate(races, start=1):
        scores = race_scored_distances(race)
        if scores is None:
            return False, f"Head-to-head grading: Race {index} has missing or invalid scored distances."
        challenger_m, incumbent_m = scores
        # Decimal arithmetic includes the exact 5% boundary without score rounding.
        credited = challenger_m > 0 and challenger_m >= incumbent_m * (Decimal(1) - tolerance)
        credited_wins += int(credited)
        lines.append(
            f"Race {index} scored distance: challenger {challenger_m:g} m; incumbent {incumbent_m:g} m. "
            + ("Counts as a win for grading." if credited else "Does not count as a win for grading.")
        )
    passed = credited_wins > len(races) // 2
    lines.insert(
        0,
        f"Head-to-head grading: {'PASS' if passed else 'FAIL'} with {tolerance_fraction:.0%} tolerance "
        f"({credited_wins}/{len(races)} races count as wins; a majority is required). "
        "A challenger with positive scored distance may finish this percentage behind the incumbent.",
    )
    return passed, "\n".join(lines)


def grade() -> dict[str, Any]:
    started = time.monotonic()
    racing_file = find_submitted_racing_file(SUBMISSION_PATH)
    if racing_file is not None:
        return blank_results(
            "Grading stopped: this upload contains game files in a racing directory "
            f"(for example, {racing_file.as_posix()}). No level checks or races were run.\n\n"
            "Follow the official 'Create your Gradescope submission' instructions in the "
            "Formula 110 project write-up (src/controllers/instructions.md). Use that workflow "
            "to create artifacts/formula110-exercise-submission.zip, then upload that ZIP "
            "to Gradescope and resubmit."
        )
    config = read_config()
    module_names = {int(level): str(name) for level, name in config["levels"].items()}
    function_name = str(config["control_function"])
    solo_config = config["solo"]
    race_config = config["head_to_head"]
    points = config["rubric"]

    files: dict[int, Path | None] = {}
    locations: dict[int, str] = {}
    requirements: dict[int, tuple[bool, str]] = {}
    structures: dict[int, tuple[bool, str]] = {}
    validations: dict[int, dict[str, Any]] = {}
    for level in range(4):
        files[level], locations[level] = locate_module(module_names[level])
        requirements[level] = inspect_submission_requirements(files[level])
        structures[level] = inspect_structure(files[level], level)
        validations[level] = validate_control(files[level], function_name)

    level_0_behavior = validate_level_0_behavior(files[0], function_name)
    level_2_throttle = validate_level_2_throttle(files[2], function_name)

    solo_result = (
        run_solo(
            files[0],
            function_name,
            seed=int(solo_config["seed"]),
            duration_seconds=float(solo_config["duration_seconds"]),
            timeout_seconds=float(solo_config["trial_timeout_seconds"]),
        )
        if validations[0].get("ok") is True
        else {"ok": False, "error": "Level 0 control validation failed"}
    )
    race_results: dict[int, dict[str, Any]] = {}
    for challenger_level in (1, 2, 3):
        incumbent_level = challenger_level - 1
        if validations[challenger_level].get("ok") is True and validations[incumbent_level].get("ok") is True:
            race_results[challenger_level] = run_head_to_head(
                files[challenger_level],
                files[incumbent_level],
                function_name,
                seed=int(race_config["seed"]),
                race_count=int(race_config["race_count"]),
                round_seconds=float(race_config["round_seconds"]),
                win_margin_m=float(race_config["win_margin_m"]),
                timeout_seconds=float(race_config["trial_timeout_seconds"]),
            )
        else:
            race_results[challenger_level] = {
                "ok": False,
                "error": f"Level {challenger_level} or Level {incumbent_level} control validation failed",
            }

    level_1_distances = head_to_head_distances(race_results[1], "challenger")
    minimum_level_1_distance = float(race_config["minimum_level_1_distance_m"])
    level_1_reaches_distance = bool(level_1_distances) and all(
        distance >= minimum_level_1_distance for distance in level_1_distances
    )
    race_summaries = {
        level: head_to_head_summary(
            module_names[level],
            module_names[level - 1],
            race_results[level],
        )
        for level in (1, 2, 3)
    }
    race_assessments = {
        level: assess_head_to_head(race_results[level], tolerance_fraction=float(race_config["win_tolerance_fraction"]))
        for level in (1, 2, 3)
    }

    tests = [
        *[
            test_case(
                f"Submission requirements: Level {level}",
                requirements[level][0],
                float(points[f"level_{level}_submission_requirements"]),
                requirements[level][1],
                f"0.{level + 1}",
            )
            for level in range(4)
        ],
        test_case(
            "Level 0: variables, conditionals, and wall sensors",
            structures[0][0] and level_0_behavior.get("ok") is True,
            float(points["level_0_structure"]),
            structures[0][1] + "\n" + level_0_behavior_output(level_0_behavior),
            "1.1",
        ),
        test_case(
            "Level 0: valid control function",
            validations[0].get("ok") is True,
            float(points["level_0_control"]),
            validation_output(module_names[0], validations[0]),
            "1.2",
        ),
        test_case(
            "Level 0: completes a lap",
            solo_result.get("ok") is True and int(solo_result.get("lap_count", 0)) >= 1,
            float(points["level_0_lap"]),
            solo_summary(module_names[0], solo_result),
            "1.3",
        ),
        test_case(
            "Level 1: conditional maximum-speed strategy",
            structures[1][0],
            float(points["level_1_structure"]),
            structures[1][1],
            "2.1",
        ),
        test_case(
            "Level 1: valid control function",
            validations[1].get("ok") is True,
            float(points["level_1_control"]),
            validation_output(module_names[1], validations[1]),
            "2.2",
        ),
        test_case(
            "Level 1: reaches 250 meters in the head-to-head race",
            level_1_reaches_distance,
            float(points["level_1_distance"]),
            race_summaries[1],
            "2.3",
        ),
        test_case(
            "Level 1: beats Level 0",
            race_assessments[1][0],
            float(points["level_1_beats_level_0"]),
            race_summaries[1] + "\n" + race_assessments[1][1],
            "2.4",
        ),
        test_case(
            "Level 2: proportional throttle expression",
            structures[2][0] and level_2_throttle.get("ok") is True,
            float(points["level_2_structure"]),
            structures[2][1] + "\n" + level_2_throttle_output(level_2_throttle),
            "3.1",
        ),
        test_case(
            "Level 2: valid control function",
            validations[2].get("ok") is True,
            float(points["level_2_control"]),
            validation_output(module_names[2], validations[2]),
            "3.2",
        ),
        test_case(
            "Level 2: beats Level 1",
            race_assessments[2][0],
            float(points["level_2_beats_level_1"]),
            race_summaries[2] + "\n" + race_assessments[2][1],
            "3.3",
        ),
        test_case(
            "Level 3: camera sensor strategy",
            structures[3][0],
            float(points["level_3_structure"]),
            structures[3][1],
            "4.1",
        ),
        test_case(
            "Level 3: valid control function",
            validations[3].get("ok") is True,
            float(points["level_3_control"]),
            validation_output(module_names[3], validations[3]),
            "4.2",
        ),
        test_case(
            "Level 3: beats Level 2",
            race_assessments[3][0],
            float(points["level_3_beats_level_2"]),
            race_summaries[3] + "\n" + race_assessments[3][1],
            "4.3",
        ),
    ]
    base_score = sum(float(test["score"]) for test in tests)
    early_bonus, hours_early = early_submission_bonus(SUBMISSION_METADATA_PATH)
    final_score = round_score(base_score + early_bonus)
    module_lines = "\n".join(f"- Level {level}, {module_names[level]}: {locations[level]}" for level in range(4))
    bonus_message = ""
    if early_bonus and hours_early is not None:
        bonus_message = (
            f"\n\nEarly submission extra credit: **+{early_bonus:.0f} percentage points** "
            f"({hours_early:.1f} hours before the deadline)."
        )
    return {
        "score": final_score,
        "execution_time": round(time.monotonic() - started, 3),
        "output": f"## Required progression files\n\n{module_lines}{bonus_message}",
        "output_format": "md",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": tests,
        "extra_data": {
            "base_score": base_score,
            "early_submission_bonus": early_bonus,
            "hours_before_deadline": round(hours_early, 2) if hours_early is not None else None,
            "level_2_throttle": level_2_throttle,
            "level_0_behavior": level_0_behavior,
            "level_0_solo": solo_result,
            "head_to_head": race_results,
            "head_to_head_win_tolerance_fraction": float(race_config["win_tolerance_fraction"]),
        },
    }


def main() -> None:
    results = blank_results("The exercise autograder started but did not finish.")
    write_results(results)
    try:
        results = grade()
    except BaseException as error:
        results = blank_results(f"Autograder infrastructure error: {type(error).__name__}: {error}")
    write_results(results)


if __name__ == "__main__":
    main()
