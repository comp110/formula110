"""Compare live Bahrain frame rates with real All Spawns leaderboard controllers.

Each field runs in a fresh interpreter, with real-time physics, an overhead
camera, no audio, and uncapped offscreen rendering. This measures a running
race rather than a frozen scene; window/compositor and audio costs are excluded.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from importlib import import_module
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from plan_race_night import build_bahrain_groups
from run_submissions import DEFAULT_EXPORT_DIR
from score_races import RaceSpec, positive_float, replay_command

from racing.game.app import create_heat_viewer_app
from racing.game.config import CameraView, HeatViewerConfig, RacingAudioConfig, parse_window_size
from racing.race.heat import DEFAULT_HEAT_COLORS, HEAT_ENTRANT_COUNTS, HeatEntrant
from racing.race.rules import HeadToHeadRaceRules
from racing.race.start import LIGHTS_OUT_SECONDS
from racing.student.api import load_student_submission


def benchmark(args: argparse.Namespace, count: int) -> dict[str, Any]:
    field = build_bahrain_groups(args.export_dir.resolve(), set(), count)[1]
    entrants = []
    for index, competitor in enumerate(field["competitors"]):
        submission = load_student_submission(competitor["controller_path"], suppress_prints=True)
        entrants.append(
            HeatEntrant(
                name=competitor["display_name"],
                controller=submission.controller,
                team_color=submission.car_color or DEFAULT_HEAT_COLORS[index],
            )
        )
    rules = HeadToHeadRaceRules(
        damage_enabled=False,
        marshal_stuck_seconds=1.0,
        marshal_cooldown_seconds=0.5,
        marshal_penalty_m=3.0,
    )
    spec = RaceSpec(
        submissions=tuple(c["submission_id"] for c in field["competitors"]),
        names=tuple(c["display_name"] for c in field["competitors"]),
        label=field["title"],
        export_dir=str(args.export_dir.resolve()),
        seed=args.seed,
        track="bahrain",
        round_laps=3,
        rules=rules,
    )
    command = replay_command(spec, fullscreen=True, camera="top_down", title=spec.label)
    (args.output / f"replay-{count}.sh").write_text("#!/bin/sh\n" + command + "\n")
    app = cast(
        Any,
        create_heat_viewer_app(
            HeatViewerConfig(
                entrants=tuple(entrants),
                title=spec.label,
                track_id="bahrain",
                random_seed=args.seed,
                camera_view=CameraView.TOP_DOWN,
                round_laps=3,
                finish_timeout_seconds=spec.finish_timeout_seconds,
                rules=rules,
                size=args.size,
                window_type="offscreen",
                vsync=False,
                audio=RacingAudioConfig(enabled=False),
            )
        ),
    )
    try:
        core = cast(Any, import_module("panda3d.core"))
        clock = core.ClockObject.getGlobalClock()
        clock.setMode(core.ClockObject.MNormal)
        clock.tick()
        app.racing_start_sequence.elapsed_seconds = LIGHTS_OUT_SECONDS
        start = perf_counter()
        while perf_counter() - start < args.warmup_seconds:
            app.step()
        simulation_start = app.racing_timing._elapsed_seconds
        timings = []
        start = perf_counter()
        while perf_counter() - start < args.seconds:
            before = perf_counter()
            app.step()
            timings.append((perf_counter() - before) * 1000)
            if hasattr(app, "racing_result"):
                raise RuntimeError("Race ended during measurement; reduce --seconds")
        elapsed = perf_counter() - start
        result = {
            "cars": count,
            "track": "bahrain",
            "seed": args.seed,
            "camera": "top_down",
            "size": args.size,
            "window_type": "offscreen",
            "vsync": False,
            "audio": False,
            "clock": "real_time",
            "rules": asdict(rules),
            "warmup_seconds": args.warmup_seconds,
            "wall_seconds": elapsed,
            "simulation_seconds": app.racing_timing._elapsed_seconds - simulation_start,
            "frames": len(timings),
            "average_fps": len(timings) / elapsed,
            "median_frame_ms": median(timings),
            "p95_frame_ms": sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
            "frame_ms": timings,
            "spec": asdict(spec),
            "replay_command": command,
        }
        screenshot = args.output / f"race-{count}.png"
        if app.screenshot(namePrefix=str(screenshot.resolve()), defaultFilename=False) is None:
            raise RuntimeError(f"Could not capture {screenshot}")
        (args.output / f"benchmark-{count}.json").write_text(json.dumps(result, indent=2) + "\n")
        return result
    finally:
        app.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--counts", type=int, nargs="+", choices=HEAT_ENTRANT_COUNTS, default=[10, 20])
    parser.add_argument("--seed", type=int, default=110)
    parser.add_argument("--seconds", type=positive_float, default=20.0)
    parser.add_argument("--warmup-seconds", type=positive_float, default=3.0)
    parser.add_argument("--size", type=parse_window_size, default=(1920, 1080))
    parser.add_argument("--output", type=Path, default=Path("artifacts/bahrain-car-count-performance"))
    parser.add_argument("--worker", type=int, choices=HEAT_ENTRANT_COUNTS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        benchmark(args, args.worker)
        return
    results = []
    for count in args.counts:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(count),
            "--export-dir",
            str(args.export_dir),
            "--seed",
            str(args.seed),
            "--seconds",
            str(args.seconds),
            "--warmup-seconds",
            str(args.warmup_seconds),
            "--size",
            f"{args.size[0]}x{args.size[1]}",
            "--output",
            str(args.output),
        ]
        print(f"Benchmarking {count} cars in overhead view ...", flush=True)
        with (args.output / f"worker-{count}.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        result = json.loads((args.output / f"benchmark-{count}.json").read_text())
        results.append({key: value for key, value in result.items() if key != "frame_ms"})
        print(
            f"{count} cars: {result['average_fps']:.1f} FPS; median {result['median_frame_ms']:.1f} ms; "
            f"p95 {result['p95_frame_ms']:.1f} ms",
            flush=True,
        )
    (args.output / "comparison.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
