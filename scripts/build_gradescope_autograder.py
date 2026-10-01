#!/usr/bin/env python3
"""Build a Gradescope autograder for one submission-selected controller."""

from __future__ import annotations

import argparse
import json
import stat
import tempfile
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = PROJECT_ROOT / "autograder" / "gradescope"
RACING_SOURCE = PROJECT_ROOT / "src" / "racing"
DEFAULT_OUTPUT = PROJECT_ROOT / "artifacts" / "formula110-gradescope-autograder.zip"
GRADING_SEEDS = (110, 2026, 1893, 7656, 9340)


def build_config() -> dict[str, object]:
    """Return the instructor-visible grading configuration."""
    return {
        "schema_version": 2,
        "submission_manifest": "formula110-submission.json",
        "control_function": "control",
        "seeds": list(GRADING_SEEDS),
        "duration_seconds": 30.0,
        "trial_timeout_seconds": 60.0,
        "marshal": {
            "stuck_seconds": 2.0,
            "distance_penalty_m": 5.0,
            "cooldown_seconds": 2.0,
        },
        "rubric": {"event_qualification": 100.0},
    }


def _write_bundle_tree(root: Path, config: dict[str, object]) -> None:
    for template_name in ("setup.sh", "run_autograder", "grade.py", "race_worker.py", "control_worker.py"):
        source = TEMPLATE_ROOT / template_name
        destination = root / template_name
        destination.write_bytes(source.read_bytes())
        if template_name in ("setup.sh", "run_autograder", "race_worker.py", "control_worker.py"):
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
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sorted(path for path in tree.rglob("*") if path.is_file()):
            relative = source.relative_to(tree)
            info = zipfile.ZipInfo.from_file(source, arcname=str(relative))
            info.compress_type = zipfile.ZIP_DEFLATED
            if relative.as_posix() in {"setup.sh", "run_autograder", "race_worker.py", "control_worker.py"}:
                info.external_attr = (stat.S_IFREG | 0o755) << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def build_archive(output: Path) -> Path:
    """Create and return a Gradescope-ready zip archive."""
    if not TEMPLATE_ROOT.is_dir() or not RACING_SOURCE.is_dir():
        raise FileNotFoundError("run this script from a complete Formula 110 project checkout")

    config = build_config()
    with tempfile.TemporaryDirectory(prefix="formula110-autograder-") as temporary_directory:
        tree = Path(temporary_directory)
        _write_bundle_tree(tree, config)
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
