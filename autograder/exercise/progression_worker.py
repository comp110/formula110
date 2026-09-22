#!/usr/bin/env python3
"""Trusted worker for one exercise controller validation, solo run, or race."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from importlib import import_module
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self, cast

from racing import LidarSensors, OdometrySensors, RobotCommand, RobotSensors

os.environ.setdefault(
    "FORMULA110_CONTROL_WORKER",
    "/opt/formula110-exercise-autograder/control_worker.py",
)
SHARED_WORKER_ROOT = Path(__file__).resolve().parents[1] / "gradescope"
if SHARED_WORKER_ROOT.is_dir():
    sys.path.insert(0, str(SHARED_WORKER_ROOT))


class ControllerClientLike(Protocol):
    def __enter__(self) -> Self: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    def command(self, sensors: RobotSensors) -> RobotCommand: ...


class ControllerClientFactory(Protocol):
    def __call__(
        self,
        *,
        submission: Path,
        module_file: Path,
        function_name: str,
    ) -> ControllerClientLike: ...


race_worker = import_module("race_worker")
controller_client = cast(ControllerClientFactory, race_worker.ControllerClient)
run_solo_trial = cast(Callable[[argparse.Namespace], dict[str, object]], race_worker.run_trial)
validate_controller = cast(Callable[[argparse.Namespace], dict[str, object]], race_worker.validate_controller)

RESULT_PREFIX = "FORMULA110_RESULT="


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("validate", "level-0-behavior", "level-2-throttle", "solo", "head-to-head"), required=True
    )
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--module-file", type=Path, required=True)
    parser.add_argument("--opponent-module-file", type=Path)
    parser.add_argument("--function", default="control")
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--races", type=int, default=1)
    parser.add_argument("--win-margin-m", type=float, default=1.0)
    return parser.parse_args()


def validate_level_0_behavior(args: argparse.Namespace) -> dict[str, object]:
    """Check steering direction and forward throttle without fixing tuning values."""
    # Use clearly near/far walls so students can tune the thresholds and strengths.
    cases = ((100.0, 100.0, 0), (0.5, 100.0, 1), (100.0, 0.5, -1), (0.5, 0.5, 1))
    cases_checked = 0
    with controller_client(
        submission=args.submission.resolve(),
        module_file=args.module_file.resolve(),
        function_name=str(args.function),
    ) as client:
        for left_m, right_m, direction in cases:
            for speed_mps in (0.0, 4.0, 8.0):
                sensors = RobotSensors(
                    odometry=OdometrySensors(speed_mps=speed_mps),
                    wall_lidar=LidarSensors(
                        angles_degrees=(-20.0, 20.0),
                        distances_m=(left_m, right_m),
                    ),
                )
                command = client.command(sensors)
                cases_checked += 1
                steering_matches = (
                    math.isclose(command.steer, 0.0, abs_tol=1e-6)
                    if direction == 0
                    else 0.0 < command.steer * direction <= 1.0
                )
                if not (steering_matches and 0.0 < command.throttle <= 1.0):
                    return {
                        "ok": False,
                        "cases_checked": cases_checked,
                        "error": (
                            f"At speed {speed_mps:g} m/s with front-left wall {left_m:g} m "
                            f"and front-right wall {right_m:g} m, control returned "
                            f"throttle {command.throttle:g} and steer {command.steer:g}. "
                            "Review the Level 0 steering plan and forward-throttle instructions."
                        ),
                    }
    return {"ok": True, "cases_checked": cases_checked}


def validate_level_2_throttle(args: argparse.Namespace) -> dict[str, object]:
    """Evaluate proportional throttle at varied speeds and wall readings."""
    speeds = (0.0, 7.5, 15.0, 22.5, 2.3, 11.25, 14.85, 15.15, 18.75)
    wall_distances = ((20.0, 20.0), (1.0, 20.0), (20.0, 1.0), (1.0, 1.0))
    cases_checked = 0
    with controller_client(
        submission=args.submission.resolve(),
        module_file=args.module_file.resolve(),
        function_name=str(args.function),
    ) as client:
        for left_m, right_m in wall_distances:
            for speed_mps in speeds:
                sensors = RobotSensors(
                    odometry=OdometrySensors(speed_mps=speed_mps),
                    wall_lidar=LidarSensors(
                        angles_degrees=(-20.0, 20.0),
                        distances_m=(left_m, right_m),
                    ),
                )
                expected = (15.0 - speed_mps) / 15.0
                actual = client.command(sensors).throttle
                cases_checked += 1
                if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6):
                    return {
                        "ok": False,
                        "cases_checked": cases_checked,
                        "error": (
                            f"At speed {speed_mps:g} m/s with front-left wall {left_m:g} m "
                            f"and front-right wall {right_m:g} m, control returned throttle {actual:g}. "
                            "This does not match the Level 2 speed-scaling requirements; "
                            "review the throttle examples in the instructions."
                        ),
                    }
    return {"ok": True, "cases_checked": cases_checked}


def run_head_to_head(args: argparse.Namespace) -> dict[str, object]:
    """Race the primary module as challenger against the opponent module."""
    from racing.race.head_to_head import run_headless_head_to_head
    from racing.race.rules import HeadToHeadRaceRules

    opponent_file = args.opponent_module_file
    if opponent_file is None:
        raise ValueError("head-to-head mode requires --opponent-module-file")
    rules = HeadToHeadRaceRules(
        win_margin_m=float(args.win_margin_m),
        marshal_enabled=True,
    )
    with (
        controller_client(
            submission=args.submission.resolve(),
            module_file=args.module_file.resolve(),
            function_name=str(args.function),
        ) as challenger,
        controller_client(
            submission=args.submission.resolve(),
            module_file=opponent_file.resolve(),
            function_name=str(args.function),
        ) as incumbent,
    ):
        result = run_headless_head_to_head(
            challenger_controller=challenger.command,
            incumbent_controller=incumbent.command,
            challenger_name=args.module_file.stem,
            incumbent_name=opponent_file.stem,
            race_count=int(args.races),
            round_seconds=float(args.seconds),
            random_seed=int(args.seed),
            rules=rules,
            copies_per_side=1,
        )
    return {"ok": True, **result.to_dict()}


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.mode == "validate":
        return validate_controller(args)
    if args.mode == "level-0-behavior":
        return validate_level_0_behavior(args)
    if args.mode == "level-2-throttle":
        return validate_level_2_throttle(args)
    if args.mode == "solo":
        return run_solo_trial(args)
    return run_head_to_head(args)


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except BaseException as error:
        result = {
            "ok": False,
            "seed": args.seed,
            "error": f"{type(error).__name__}: {error}"[:1000],
        }
    print(RESULT_PREFIX + json.dumps(result, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
