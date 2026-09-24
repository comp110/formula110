#!/usr/bin/env python3
"""Trusted Formula 110 Gradescope grading driver."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

AUTOGRADER_ROOT = Path("/autograder")
CONFIG_PATH = Path("/opt/formula110-autograder/config.json")
WORKER_PATH = Path("/opt/formula110-autograder/race_worker.py")
RUNTIME_PATH = Path("/opt/formula110-runtime")
VENV_PATH = AUTOGRADER_ROOT / "venv"
SUBMISSION_PATH = AUTOGRADER_ROOT / "submission"
RESULTS_PATH = AUTOGRADER_ROOT / "results" / "results.json"
UV_PATH = Path("/usr/local/bin/uv")
RESULT_PREFIX = "FORMULA110_RESULT="
MAX_DIAGNOSTIC_CHARS = 6000
DEPENDENCY_SYNC_TIMEOUT_SECONDS = 300.0
CONTROL_VALIDATION_TIMEOUT_SECONDS = 35.0
MODULE_NAME_PATTERN = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
METRIC_ZERO_EPSILON = 1e-9
METERS_PER_SECOND_TO_MILES_PER_HOUR = 2.2369362920544
LEADERBOARD_METRIC_NAMES = (
    "All Spawns, No Crumbs (Laps)",
    "Clock It (s)",
    "Hits Different (damage %)",
    "Sips Tea (g-s)",
    "Gs Going Crazy (g-s)",
    "Gas Locked In (s)",
    "Serving Sideways (m)",
    "Speedmaxxing (mph)",
)


def read_config() -> dict[str, Any]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_results(results: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = RESULTS_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(RESULTS_PATH)


def blank_results(message: str) -> dict[str, Any]:
    return {
        "output": message,
        "output_format": "text",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": [],
        "leaderboard": [],
    }


def locate_module(module_name: str) -> tuple[Path | None, str]:
    relative = Path(*module_name.split(".")).with_suffix(".py")
    direct_candidates = [SUBMISSION_PATH / relative, SUBMISSION_PATH / "src" / relative]
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate.resolve(), f"found {candidate.relative_to(SUBMISSION_PATH)}"

    matches = sorted(
        path.resolve() for path in SUBMISSION_PATH.rglob(relative.name) if path.is_file() and not path.is_symlink()
    )
    suffix_parts = relative.parts
    exact_suffix_matches = [path for path in matches if path.parts[-len(suffix_parts) :] == suffix_parts]
    if len(exact_suffix_matches) == 1:
        return exact_suffix_matches[0], f"found {exact_suffix_matches[0].relative_to(SUBMISSION_PATH)}"
    if len(matches) == 1:
        return matches[0], f"found flattened upload {matches[0].relative_to(SUBMISSION_PATH)}"
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(SUBMISSION_PATH)) for path in matches[:8])
        return None, f"multiple possible files named {relative.name}: {names}"
    return None, f"expected {relative} (or src/{relative}) in the submission"


def read_submission_controller(manifest_name: str) -> tuple[str | None, str]:
    """Read and validate the one controller module selected by the exporter."""
    manifest = SUBMISSION_PATH / manifest_name
    if not manifest.is_file() or manifest.is_symlink():
        return None, f"expected {manifest_name} at the root of the submission"
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, f"could not read {manifest_name}: {error}"
    if not isinstance(payload, dict):
        return None, f"{manifest_name} has an unsupported schema"
    payload = cast(dict[str, object], payload)
    if payload.get("schema_version") != 1:
        return None, f"{manifest_name} has an unsupported schema"
    module_name = payload.get("controller_module")
    if (
        not isinstance(module_name, str)
        or module_name.endswith(".py")
        or MODULE_NAME_PATTERN.fullmatch(module_name) is None
        or not module_name.startswith("controllers.")
    ):
        return None, f"{manifest_name} does not name a valid controllers.* module"
    return module_name, f"{manifest_name} selects {module_name}"


def run_command(command: list[str], *, timeout_seconds: float) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            cwd=SUBMISSION_PATH,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_seconds,
            check=False,
            text=True,
            errors="replace",
            env={
                "HOME": "/tmp/formula110-grader",
                "PATH": f"{VENV_PATH / 'bin'}:/usr/bin:/bin",
                "PYTHONPATH": str(RUNTIME_PATH),
                "VIRTUAL_ENV": str(VENV_PATH),
                "UV_PYTHON_DOWNLOADS": "never",
            },
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout_seconds:.0f} seconds"
    output = completed.stdout[-MAX_DIAGNOSTIC_CHARS:].strip()
    if not output:
        output = "no diagnostics"
    return completed.returncode == 0, output


def locate_submission_pyproject() -> tuple[Path | None, str]:
    """Locate the project file without accepting an ambiguous upload."""
    direct = SUBMISSION_PATH / "pyproject.toml"
    if direct.is_file() and not direct.is_symlink():
        return direct.resolve(), "found pyproject.toml"

    matches = sorted(
        path.resolve() for path in SUBMISSION_PATH.rglob("pyproject.toml") if path.is_file() and not path.is_symlink()
    )
    if len(matches) == 1:
        relative = matches[0].relative_to(SUBMISSION_PATH)
        return matches[0], f"found {relative}"
    if len(matches) > 1:
        names = ", ".join(str(path.relative_to(SUBMISSION_PATH)) for path in matches[:8])
        return None, f"multiple possible pyproject.toml files: {names}"
    return None, "expected pyproject.toml in the submission"


def sync_submission_dependencies() -> tuple[bool, str]:
    """Install submission runtime dependencies into the existing grader venv."""
    pyproject, location = locate_submission_pyproject()
    if pyproject is None:
        return False, location

    command = [
        str(UV_PATH),
        "sync",
        "--project",
        str(pyproject.parent),
        "--active",
        "--inexact",
        "--no-dev",
        "--no-default-groups",
        "--no-install-project",
        "--no-python-downloads",
    ]
    lockfile = pyproject.with_name("uv.lock")
    if lockfile.is_file() and not lockfile.is_symlink():
        command.append("--locked")
    synced, diagnostic = run_command(command, timeout_seconds=DEPENDENCY_SYNC_TIMEOUT_SECONDS)
    if synced:
        return True, f"{location}; runtime dependencies synchronized"
    return False, f"{location}; dependency sync failed: {diagnostic}"


def worker_command(arguments: list[str]) -> list[str]:
    prlimit = shutil.which("prlimit") or "/usr/bin/prlimit"
    return [
        prlimit,
        "--cpu=25",
        "--fsize=1048576",
        "--nproc=128",
        "--",
        "/usr/bin/env",
        "-i",
        "HOME=/tmp",
        "PATH=/autograder/venv/bin:/usr/bin:/bin",
        "PYTHONPATH=/opt/formula110-runtime",
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
        try:
            payload = json.loads(payload_line[len(RESULT_PREFIX) :])
        except json.JSONDecodeError as error:
            return {"ok": False, "error": f"worker returned invalid JSON: {error}"}
        if not isinstance(payload, dict):
            return {"ok": False, "error": "worker result was not an object"}
        return cast(dict[str, Any], payload)
    finally:
        if output_path is not None:
            output_path.unlink(missing_ok=True)


def validate_control(module_file: Path | None, function_name: str) -> dict[str, Any]:
    if module_file is None:
        return {"ok": False, "error": "module file is missing"}
    return run_worker(
        [
            "--submission",
            str(SUBMISSION_PATH),
            "--module-file",
            str(module_file),
            "--function",
            function_name,
            "--validate-only",
        ],
        timeout_seconds=CONTROL_VALIDATION_TIMEOUT_SECONDS,
    )


def run_trials(
    module_file: Path | None,
    function_name: str,
    seeds: list[int],
    duration_seconds: float,
    timeout_seconds: float,
    marshal_stuck_seconds: float,
    marshal_penalty_m: float,
    marshal_cooldown_seconds: float,
) -> list[dict[str, Any]]:
    if module_file is None:
        return [{"ok": False, "seed": seed, "error": "module file is missing"} for seed in seeds]
    return [
        run_worker(
            [
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
                "--marshal-stuck-seconds",
                str(marshal_stuck_seconds),
                "--marshal-penalty-m",
                str(marshal_penalty_m),
                "--marshal-cooldown-seconds",
                str(marshal_cooldown_seconds),
            ],
            timeout_seconds=timeout_seconds,
        )
        for seed in seeds
    ]


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


def trial_summary(module_name: str, trials: list[dict[str, Any]]) -> str:
    lines = [module_name]
    for seed_number, trial in enumerate(trials, start=1):
        if trial.get("ok") is not True:
            lines.append(f"seed {seed_number}: ERROR — {trial.get('error', 'trial failed')}")
            continue
        lines.append(
            f"seed {seed_number}: {float(trial['elapsed_seconds']):.1f} s, "
            f"{float(trial['partial_laps']):.3f} laps, "
            f"{float(trial['raw_distance_m']):.1f} m, "
            f"top {float(trial['max_speed_mps']) * METERS_PER_SECOND_TO_MILES_PER_HOUR:.2f} mph, "
            f"damage {float(trial['damage']) * 100.0:.1f}%, "
            f"wall contact {float(trial['wall_contact_seconds']):.3f} s, "
            f"marshals {int(trial.get('marshal_count', 0))}"
        )
    return "\n".join(lines)


def serial_numbered_trials(trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Replace private seed values with their one-based run numbers."""
    return [{**trial, "seed": seed_number} for seed_number, trial in enumerate(trials, start=1)]


def all_trials(trials: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> bool:
    return bool(trials) and all(trial.get("ok") is True and predicate(trial) for trial in trials)


def completed_with_forward_progress(trial: dict[str, Any], duration_seconds: float) -> bool:
    """Return whether a trial ran for the configured duration and moved forward."""
    return float(trial["elapsed_seconds"]) >= duration_seconds - 1e-6 and float(trial["raw_distance_m"]) > 1e-6


def completed_laps(trial: dict[str, Any]) -> list[dict[str, Any]]:
    """Return trusted completed-lap telemetry from one worker result."""
    laps = trial.get("laps", [])
    if not isinstance(laps, list):
        return []
    return [cast(dict[str, Any], lap) for lap in cast(list[object], laps) if isinstance(lap, dict)]


def aggregate_lap_metric(
    trials: list[dict[str, Any]],
    *,
    key: str,
    eligible: Callable[[dict[str, Any]], bool] = lambda _lap: True,
    highest: bool = False,
) -> float | None:
    """Average each starting offset's best eligible completed-lap value."""
    per_trial_values: list[float] = []
    for trial in trials:
        values = [float(lap[key]) for lap in completed_laps(trial) if eligible(lap)]
        if not values:
            return None
        per_trial_values.append((max if highest else min)(values))
    return sum(per_trial_values) / len(per_trial_values) if per_trial_values else None


def leaderboard_entry(
    name: str,
    value: float | None,
    *,
    decimal_places: int,
    ascending: bool = False,
    scale: float = 1.0,
) -> dict[str, Any] | None:
    """Build a numeric Gradescope field, omitting unavailable metrics."""
    if value is None:
        return None
    entry: dict[str, Any] = {
        "name": name,
        "value": round(value * scale, decimal_places),
    }
    if ascending:
        entry["order"] = "asc"
    return entry


def leaderboard_report(leaderboard: list[dict[str, Any]], message: str) -> str:
    """Format the qualification message and all metric values for students."""
    if not leaderboard:
        return message
    values = {str(entry["name"]): entry["value"] for entry in leaderboard}
    metric_lines = [f"- {name}: {values.get(name, 'N/A')}" for name in LEADERBOARD_METRIC_NAMES]
    return "\n".join([message, "", "Leaderboard metrics:", *metric_lines])


def clean_lap(lap: dict[str, Any]) -> bool:
    """Return whether a lap had neither damage nor wall contact."""
    return (
        float(lap["damage_delta"]) <= METRIC_ZERO_EPSILON and float(lap["wall_contact_seconds"]) <= METRIC_ZERO_EPSILON
    )


def contact_free_lap(lap: dict[str, Any]) -> bool:
    """Return whether a lap had no wall contact."""
    return float(lap["wall_contact_seconds"]) <= METRIC_ZERO_EPSILON


def brake_free_lap(lap: dict[str, Any]) -> bool:
    """Return whether trusted actuation applied no brakes during a lap."""
    return not bool(lap["brake_applied"])


def leaderboard_for(trials: list[dict[str, Any]], duration_seconds: float) -> tuple[list[dict[str, Any]], str]:
    qualified = all_trials(
        trials,
        lambda trial: (
            completed_with_forward_progress(trial, duration_seconds)
            and bool(trial["survived"])
            and float(trial["damage"]) < 1.0
        ),
    )
    if not qualified:
        return [], "Leaderboard: DQ — the controller did not finish every starting offset with forward progress."

    endurance = min(float(trial["partial_laps"]) for trial in trials)
    clean_hot_lap = aggregate_lap_metric(
        trials,
        key="duration_seconds",
        eligible=clean_lap,
    )
    total_damage = sum(float(trial["damage"]) for trial in trials)
    smooth_operator = aggregate_lap_metric(trials, key="horizontal_g_seconds")
    g_force_junkie = aggregate_lap_metric(
        trials,
        key="horizontal_g_seconds",
        eligible=contact_free_lap,
        highest=True,
    )
    brake_is_lava = aggregate_lap_metric(
        trials,
        key="duration_seconds",
        eligible=brake_free_lap,
    )
    drift_queen = aggregate_lap_metric(trials, key="drift_distance_m", highest=True)
    full_send_mps = aggregate_lap_metric(trials, key="sustained_top_speed_mps", highest=True)
    full_send_mph = None if full_send_mps is None else full_send_mps * METERS_PER_SECOND_TO_MILES_PER_HOUR

    leaderboard = [
        entry
        for entry in (
            leaderboard_entry("All Spawns, No Crumbs (Laps)", endurance, decimal_places=4),
            leaderboard_entry("Clock It (s)", clean_hot_lap, decimal_places=3, ascending=True),
            leaderboard_entry("Hits Different (damage %)", total_damage, decimal_places=2, scale=100.0),
            leaderboard_entry("Sips Tea (g-s)", smooth_operator, decimal_places=3, ascending=True),
            leaderboard_entry("Gs Going Crazy (g-s)", g_force_junkie, decimal_places=3),
            leaderboard_entry("Gas Locked In (s)", brake_is_lava, decimal_places=3, ascending=True),
            leaderboard_entry("Serving Sideways (m)", drift_queen, decimal_places=3),
            leaderboard_entry("Speedmaxxing (mph)", full_send_mph, decimal_places=3),
        )
        if entry is not None
    ]
    return leaderboard, f"Leaderboard: qualified; metrics aggregate {len(trials)} starting offsets."


def grade() -> dict[str, Any]:
    started = time.monotonic()
    config = read_config()
    manifest_name = str(config["submission_manifest"])
    controller_name, manifest_message = read_submission_controller(manifest_name)
    points = float(config["rubric"]["completion_with_forward_progress"])
    if controller_name is None:
        results = blank_results(manifest_message)
        results["execution_time"] = round(time.monotonic() - started, 3)
        results["tests"] = [
            test_case(
                "Controller completes 30-second runs with forward progress",
                False,
                points,
                manifest_message,
                "1",
            )
        ]
        return results

    controller_file, controller_location = locate_module(controller_name)
    if controller_file is None:
        results = blank_results(f"{manifest_message}\n{controller_location}")
        results["execution_time"] = round(time.monotonic() - started, 3)
        results["tests"] = [
            test_case(
                "Controller completes 30-second runs with forward progress",
                False,
                points,
                controller_location,
                "1",
            )
        ]
        return results

    dependencies_synced, dependency_message = sync_submission_dependencies()
    if not dependencies_synced:
        results = blank_results(
            "Submission dependency installation failed before controller execution.\n\n" + dependency_message
        )
        results["execution_time"] = round(time.monotonic() - started, 3)
        results["tests"] = [
            test_case(
                "Controller completes 30-second runs with forward progress",
                False,
                points,
                dependency_message,
                "1",
            )
        ]
        return results

    function_name = str(config["control_function"])
    seeds = [int(seed) for seed in config["seeds"]]
    duration = float(config["duration_seconds"])
    trial_timeout = float(config["trial_timeout_seconds"])
    marshal = config["marshal"]
    if not isinstance(marshal, dict):
        raise TypeError("marshal configuration must be an object")
    marshal = cast(dict[str, Any], marshal)

    validation = validate_control(controller_file, function_name)
    trials = (
        run_trials(
            controller_file,
            function_name,
            seeds,
            duration,
            trial_timeout,
            float(marshal["stuck_seconds"]),
            float(marshal["distance_penalty_m"]),
            float(marshal["cooldown_seconds"]),
        )
        if validation.get("ok") is True
        else [
            {
                "ok": False,
                "seed": seed,
                "error": f"control validation failed: {validation.get('error', 'invalid controller')}",
            }
            for seed in seeds
        ]
    )
    summary = trial_summary(controller_name, trials)
    passed = all_trials(trials, lambda trial: completed_with_forward_progress(trial, duration))
    tests = [
        test_case(
            "Controller completes 30-second runs with forward progress",
            passed,
            points,
            summary,
            "1",
        )
    ]
    leaderboard, leaderboard_message = leaderboard_for(trials, duration)
    leaderboard_output = leaderboard_report(leaderboard, leaderboard_message)
    return {
        "execution_time": round(time.monotonic() - started, 3),
        "output": (f"Submission: {manifest_message}\nController: {controller_location}\n\n{leaderboard_output}"),
        "output_format": "text",
        "test_output_format": "text",
        "test_name_format": "text",
        "stdout_visibility": "hidden",
        "tests": tests,
        "leaderboard": leaderboard,
        "extra_data": {
            "controller_module": controller_name,
            "controller_trials": serial_numbered_trials(trials),
        },
    }


def main() -> None:
    results = blank_results("The autograder started but did not finish.")
    write_results(results)
    try:
        results = grade()
    except BaseException as error:
        results = blank_results(f"Autograder infrastructure error: {type(error).__name__}: {error}")
    write_results(results)


if __name__ == "__main__":
    main()
