#!/usr/bin/env python3
"""Search Bahrain starting seeds for LB+NN in P1 and JB in P2 in the specified eight-car heat."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from run_submissions import DEFAULT_EXPORT_DIR, PROJECT_ROOT, resolve_controller

# Input order is the starting-grid order, just as in run_submissions.py.
ENTRANTS = (
    ("431908390", "LB+NN"),
    ("431903376", "AR"),
    ("431372646", "YZ"),
    ("429929224", "JB"),
    ("431382355", "YH"),
    ("431896981", "KC"),
    ("431901622", "ZY+JL"),
    ("431906969", "JS+MH"),
)
TARGET_INDICES = (0, 3)
DEFAULT_ROUND_LAPS = 5


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def positive_seconds(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be finite and positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--start-seed", type=int, default=110, help="first seed to try (default: 110)")
    parser.add_argument("--count", type=positive_int, default=1000, help="maximum consecutive seeds (default: 1000)")
    parser.add_argument("--matches", type=positive_int, default=1, help="stop after this many matches (default: 1)")
    parser.add_argument(
        "--round-laps",
        "--laps",
        type=positive_int,
        default=DEFAULT_ROUND_LAPS,
        help=f"laps per race (default: {DEFAULT_ROUND_LAPS})",
    )
    parser.add_argument("--either-order", action="store_true", help="also accept JB in P1 and LB+NN in P2")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--timeout-seconds",
        type=positive_seconds,
        default=300.0,
        help="wall-clock limit per seed; timed-out seeds are skipped (default: 300)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        metavar="JSONL",
        help="append each trial, full results, and matching replay commands to this JSONL file",
    )
    return parser


def race_arguments(seed: int, export_dir: Path, *, round_laps: int = DEFAULT_ROUND_LAPS) -> list[str]:
    """Keep the headless search and watched replay on identical race settings."""
    args = [str(PROJECT_ROOT / "scripts" / "run_submissions.py"), *(identifier for identifier, _ in ENTRANTS)]
    for _, name in ENTRANTS:
        args.extend(["--name", name])
    args.extend(
        [
            "--export-dir",
            str(export_dir),
            "--track",
            "bahrain",
            "--seed",
            str(seed),
            "--races",
            "1",
            "--round-laps",
            str(round_laps),
            "--finish-timeout-seconds",
            "10",
            "--no-damage",
        ]
    )
    return args


def replay_command(seed: int, export_dir: Path, *, round_laps: int = DEFAULT_ROUND_LAPS) -> str:
    return shlex.join(
        [
            "uv",
            "run",
            "python",
            *race_arguments(seed, export_dir, round_laps=round_laps),
            "--fullscreen",
            "--camera",
            "cinematic",
        ]
    )


def is_match(standings: Sequence[dict[str, Any]], *, either_order: bool = False) -> bool:
    """Require actual P1/P2 finishes; distance-ranked DNFs must never match."""
    podium = tuple(
        next(
            (row["entrant_index"] for row in standings if row["finish_position"] == place and not row["dnf"]),
            None,
        )
        for place in (1, 2)
    )
    return podium == TARGET_INDICES or (either_order and podium == TARGET_INDICES[::-1])


def run_seed(
    seed: int,
    export_dir: Path,
    timeout_seconds: float,
    *,
    round_laps: int = DEFAULT_ROUND_LAPS,
) -> dict[str, Any]:
    # A new interpreter per seed also resets module-level controller state.
    command = [sys.executable, *race_arguments(seed, export_dir, round_laps=round_laps), "--headless", "--json"]
    process = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(f"race process exited {process.returncode}:\n{process.stderr[-4000:]}")
    try:
        result = json.loads(process.stdout)
        if (
            result["mode"] != "heat"
            or result["random_seed"] != seed
            or result["track_id"] != "bahrain"
            or result["round_laps"] != round_laps
            or len(result["races"]) != 1
            or len(result["races"][0]["standings"]) != len(ENTRANTS)
        ):
            raise ValueError("unexpected heat settings or standings")
    except (ValueError, KeyError, TypeError) as error:
        raise RuntimeError(f"invalid race JSON: {error}\n{process.stdout[-2000:]}") from error
    return result


def describe_standings(standings: Sequence[dict[str, Any]]) -> str:
    return ", ".join(
        f"{row['name']} DNF"
        if row["dnf"]
        else f"P{row['finish_position']} {row['name']} ({row['finish_time_seconds']:.3f}s)"
        for row in standings
    )


def save_trial(path: Path | None, record: dict[str, Any]) -> None:
    if path is not None:
        with path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(record) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    export_dir = args.export_dir.expanduser().resolve()
    try:
        for identifier, _ in ENTRANTS:
            resolve_controller(export_dir, identifier)
    except ValueError as error:
        parser.error(str(error))
    if args.output is not None:
        args.output = args.output.expanduser().resolve()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Discover unwritable output paths before spending time simulating races.
        with args.output.open("a", encoding="utf-8"):
            pass

    target = "LB+NN and JB in P1/P2, either order" if args.either_order else "P1 LB+NN, P2 JB"
    print(f"Searching Bahrain: {target}; {args.round_laps} laps, 10s finish timeout, no damage.", flush=True)
    found: list[int] = []
    skipped = 0
    for index, seed in enumerate(range(args.start_seed, args.start_seed + args.count), start=1):
        print(f"[{index}/{args.count}] Seed {seed} ...", flush=True)
        started = time.monotonic()
        try:
            result = run_seed(seed, export_dir, args.timeout_seconds, round_laps=args.round_laps)
        except subprocess.TimeoutExpired:
            skipped += 1
            print(f"  Skipped: exceeded {args.timeout_seconds:g}s wall-clock limit.", flush=True)
            save_trial(args.output, {"seed": seed, "status": "timeout", "timeout_seconds": args.timeout_seconds})
            continue
        except RuntimeError as error:
            print(f"Seed {seed}: {error}", file=sys.stderr, flush=True)
            return 2
        except KeyboardInterrupt:
            print(f"\nStopped; seed {seed} was not completed. Resume with --start-seed {seed}.", flush=True)
            return 130
        standings = result["races"][0]["standings"]
        matched = is_match(standings, either_order=args.either_order)
        command = replay_command(seed, export_dir, round_laps=args.round_laps) if matched else None
        save_trial(
            args.output,
            {
                "seed": seed,
                "status": "completed",
                "matched": matched,
                "either_order": args.either_order,
                "wall_seconds": time.monotonic() - started,
                "result": result,
                "replay_command": command,
            },
        )
        print(f"  {describe_standings(standings)}", flush=True)
        if matched:
            found.append(seed)
            print(f"\nMATCH seed {seed}\n{command}\n", flush=True)
            if len(found) >= args.matches:
                break

    print(f"Matched seeds: {', '.join(map(str, found)) or 'none'}; skipped timeouts: {skipped}.", flush=True)
    # 1 means a completed search with no match; 2 means some trials were inconclusive.
    return 0 if found else 2 if skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
