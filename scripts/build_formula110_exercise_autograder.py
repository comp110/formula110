#!/usr/bin/env python3
"""Build the Gradescope autograder for the four-level Formula 110 exercise."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "autograder" / "exercise"
SHARED_WORKER_ROOT = PROJECT_ROOT / "autograder" / "gradescope"
RACING_SOURCE = PROJECT_ROOT / "src" / "racing"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-exercise-autograder.zip"


def build_config() -> dict[str, object]:
    """Return the instructor-visible exercise grading configuration."""
    return {
        "schema_version": 1,
        "levels": {
            "0": "controllers.level_0",
            "1": "controllers.level_1",
            "2": "controllers.level_2",
            "3": "controllers.level_3",
        },
        "control_function": "control",
        "solo": {
            "seed": 110,
            "duration_seconds": 60.0,
            "trial_timeout_seconds": 90.0,
        },
        "head_to_head": {
            "seed": 110,
            "race_count": 1,
            "round_seconds": 30.0,
            "trial_timeout_seconds": 120.0,
            "minimum_level_1_distance_m": 250.0,
            "win_margin_m": 1.0,
            "win_tolerance_fraction": 0.05,
        },
        "rubric": {
            "level_0_submission_requirements": 2.0,
            "level_0_structure": 6.0,
            "level_0_control": 4.0,
            "level_0_lap": 13.0,
            "level_1_submission_requirements": 2.0,
            "level_1_structure": 5.0,
            "level_1_control": 4.0,
            "level_1_distance": 7.0,
            "level_1_beats_level_0": 7.0,
            "level_2_submission_requirements": 2.0,
            "level_2_structure": 6.0,
            "level_2_control": 4.0,
            "level_2_beats_level_1": 13.0,
            "level_3_submission_requirements": 2.0,
            "level_3_structure": 8.0,
            "level_3_control": 4.0,
            "level_3_beats_level_2": 11.0,
        },
    }


def _template_sources() -> dict[str, Path]:
    return {
        "setup.sh": TEMPLATE_ROOT / "setup.sh",
        "run_autograder": TEMPLATE_ROOT / "run_autograder",
        "grade.py": TEMPLATE_ROOT / "grade.py",
        "progression_worker.py": TEMPLATE_ROOT / "progression_worker.py",
        "race_worker.py": SHARED_WORKER_ROOT / "race_worker.py",
        "control_worker.py": SHARED_WORKER_ROOT / "control_worker.py",
    }


def _write_bundle_tree(root: Path, config: dict[str, object]) -> None:
    for template_name, source in _template_sources().items():
        destination = root / template_name
        destination.write_bytes(source.read_bytes())
        if template_name in {
            "setup.sh",
            "run_autograder",
            "progression_worker.py",
            "race_worker.py",
            "control_worker.py",
        }:
            destination.chmod(0o755)

    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    trusted_root = root / "trusted" / "racing"
    for source in RACING_SOURCE.rglob("*"):
        relative = source.relative_to(RACING_SOURCE)
        if (
            not source.is_file()
            or "assets" in relative.parts
            or "__pycache__" in relative.parts
            or source.suffix == ".pyc"
        ):
            continue
        destination = trusted_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _write_zip(tree: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    executable_names = {
        "setup.sh",
        "run_autograder",
        "progression_worker.py",
        "race_worker.py",
        "control_worker.py",
    }
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in tree.rglob("*") if path.is_file()):
            relative = source.relative_to(tree)
            info = zipfile.ZipInfo.from_file(source, arcname=str(relative))
            info.compress_type = zipfile.ZIP_DEFLATED
            if relative.as_posix() in executable_names:
                info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_archive(output: Path) -> Path:
    """Create and return a Gradescope-ready exercise autograder archive."""
    missing = [path for path in (*_template_sources().values(), RACING_SOURCE) if not path.exists()]
    if missing:
        names = ", ".join(str(path.relative_to(PROJECT_ROOT)) for path in missing)
        raise FileNotFoundError(f"missing required autograder sources: {names}")

    with tempfile.TemporaryDirectory(prefix="formula110-exercise-autograder-") as temporary_directory:
        tree = Path(temporary_directory)
        _write_bundle_tree(tree, build_config())
        _write_zip(tree, output.resolve())
    return output.resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"archive path (default: {DEFAULT_OUTPUT.relative_to(PROJECT_ROOT)})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        output = build_archive(args.output)
    except FileNotFoundError as error:
        raise SystemExit(f"error: {error}") from error
    print(output)


if __name__ == "__main__":
    main()
