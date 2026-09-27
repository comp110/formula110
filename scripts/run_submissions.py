#!/usr/bin/env python3
"""Race two, four, or eight submissions from an extracted Gradescope assignment export."""

from __future__ import annotations

import argparse
import json
import secrets
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from racing.game import cli
from racing.race.heat import HEAT_ENTRANT_COUNTS
from racing.race.runtime import DEFAULT_RACE_RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = PROJECT_ROOT / "assignment_8706145_export"
SUBMISSION_MANIFEST = "formula110-submission.json"
EXERCISE_MANIFEST = "formula110-exercise-submission.json"


def submission_id(value: str) -> str:
    """Accept a numeric ID or the export's submission_ID directory name."""
    value = value.removeprefix("submission_")
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError("expected a numeric submission ID, such as 429149360")
    return value


def race_seed(value: str) -> int:
    """Resolve a repeatable integer seed or choose one for this invocation."""
    if value == "random":
        return secrets.randbelow(2**31)
    try:
        return int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected an integer seed or 'random'") from error


def resolve_controller(export_dir: Path, identifier: str) -> Path:
    """Find the manifest-selected controller, using level 3 for older exercise exports."""
    root = (export_dir / f"submission_{identifier}").resolve()
    if not root.is_dir():
        raise ValueError(f"submission {identifier} not found: {root}")

    manifest = root / SUBMISSION_MANIFEST
    if not manifest.is_file():
        manifest = root / EXERCISE_MANIFEST
    if not manifest.is_file():
        raise ValueError(f"submission {identifier}: expected {SUBMISSION_MANIFEST} or {EXERCISE_MANIFEST}")
    try:
        raw_payload: object = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"could not read {manifest}: {error}") from error
    if not isinstance(raw_payload, dict):
        raise ValueError(f"{manifest}: expected schema_version 1")
    payload = cast(dict[str, object], raw_payload)
    if payload.get("schema_version") != 1:
        raise ValueError(f"{manifest}: expected schema_version 1")

    if manifest.name == EXERCISE_MANIFEST:
        levels = payload.get("levels")
        module = cast(dict[str, object], levels).get("3") if isinstance(levels, dict) else None
    else:
        module = payload.get("controller_module")
    if (
        not isinstance(module, str)
        or not module.startswith("controllers.")
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError(f"{manifest}: expected a controllers.* module")

    relative = Path(*module.split(".")).with_suffix(".py")
    for source_root in (root, root / "src"):
        candidate = (source_root / relative).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    raise ValueError(f"submission {identifier}: {module} selected by {manifest.name}, but {relative} is missing")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog=(
            "Additional racing h2h/heat options after the IDs are forwarded, e.g. --races 3 --round-seconds 60 "
            "--camera follow --no-audio. Heats start in ID order, first ID on pole. "
            "--seed controls starting positions (and grid order in h2h); "
            "--track-seed INT generates a reproducible procedural track."
        ),
    )
    parser.add_argument(
        "submissions", type=submission_id, nargs="+", metavar="ID", help="two, four, or eight submission IDs"
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=DEFAULT_EXPORT_DIR,
        help=f"extracted assignment directory (default: {DEFAULT_EXPORT_DIR.name})",
    )
    parser.add_argument(
        "--seed",
        type=race_seed,
        default=DEFAULT_RACE_RANDOM_SEED,
        metavar="INT|random",
        help=f"starting-position seed (also grid order in h2h), or 'random' (default: {DEFAULT_RACE_RANDOM_SEED})",
    )
    parser.add_argument("--headless", action="store_true", help="run without the default watched three-quarter view")
    parser.add_argument(
        "--dry-run", action="store_true", help="print the resolved race command without loading controllers"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    identifiers = cast(list[str], args.submissions)
    if len(identifiers) not in (2, *HEAT_ENTRANT_COUNTS):
        parser.error("provide exactly two IDs for head-to-head or four or eight IDs for a heat")
    if len(identifiers) in HEAT_ENTRANT_COUNTS and len(set(identifiers)) != len(identifiers):
        parser.error(f"a {len(identifiers)}-car heat requires {len(identifiers)} distinct submission IDs")
    try:
        controllers = [resolve_controller(args.export_dir.expanduser(), identifier) for identifier in identifiers]
    except ValueError as error:
        parser.error(str(error))

    if len(identifiers) == 2:
        runner_args = [
            "h2h",
            "--challenger-module",
            str(controllers[0]),
            "--incumbent-module",
            str(controllers[1]),
            "--challenger-fallback-name",
            f"#{identifiers[0]}",
            "--incumbent-fallback-name",
            f"#{identifiers[1]}",
        ]
    else:
        runner_args = ["heat"]
        for identifier, controller in zip(identifiers, controllers, strict=True):
            runner_args.extend(["--module", str(controller), "--fallback-name", f"#{identifier}"])
    runner_args.extend(
        [
            "--seed",
            str(args.seed),
            "--camera",
            "three_quarter",
        ]
    )
    if not args.headless:
        runner_args.append("--watch")
    runner_args.extend(extra)
    race_args = cli.build_argument_parser().parse_args(runner_args)

    for index, (identifier, controller) in enumerate(zip(identifiers, controllers, strict=True)):
        role = ("Challenger", "Incumbent ")[index] if len(identifiers) == 2 else f"Entrant {index + 1}"
        print(f"{role} #{identifier}: {controller}", file=sys.stderr)
    print(f"Seed: {race_args.seed} (reuse with --seed {race_args.seed})", file=sys.stderr)
    if args.dry_run:
        print(shlex.join([sys.executable, "-m", "racing", *runner_args]))
        return
    cli.main(runner_args)


if __name__ == "__main__":
    main()
