"""Compare stationary scene frame times and screenshots before/after batching.

Run with .venv/bin/python scripts/benchmark_track_rendering.py --track bahrain.
Physics is frozen to keep both passes identical; these are graphics benchmarks,
not predictions of frame rates during a race with student controllers.
"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any, cast

from racing.game.config import CameraView, GameConfig, RacingAudioConfig
from racing.graphics import track_rendering
from racing.graphics.camera import (
    FORMULA_DRONE_CAMERA_SETTINGS,
    FORMULA_FOLLOW_CAMERA_SETTINGS,
    apply_camera_view,
)
from racing.race.progress import resolve_track
from racing.track.procedural import TRACK_ID_PROCEDURAL
from racing.track.world import TRACK_ID_BAHRAIN, track_layout_ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=(*track_layout_ids(), TRACK_ID_PROCEDURAL), default=TRACK_ID_BAHRAIN)
    parser.add_argument("--track-seed", type=int)
    parser.add_argument("--output", type=Path, default=Path("artifacts/bahrain-performance"))
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    resolved = resolve_track(args.track, args.track_seed)

    # Defer the production optimization so both measurements use the same scene.
    pending: list[dict[str, Any]] = []
    batch = track_rendering.batch_static_track_entities

    def defer_batch(**kwargs: Any) -> None:
        pending.append(kwargs)

    track_rendering.batch_static_track_entities = defer_batch
    try:
        app_module = cast(Any, import_module("racing.game.app"))
        app = app_module.create_app(
            GameConfig(
                track_id=args.track,
                track_seed=args.track_seed,
                window_type="offscreen",
                vsync=False,
                size=(1280, 720),
                audio=RacingAudioConfig(enabled=False),
            )
        )
    finally:
        track_rendering.batch_static_track_entities = batch

    ursina = cast(Any, import_module("ursina"))
    core = cast(Any, import_module("panda3d.core"))
    try:
        # Clear uncovered pixels when switching cameras in the offscreen buffer.
        app.win.setClearColorActive(True)
        app.win.setClearColor((0, 0, 0, 1))
        app.win.setClearDepthActive(True)
        ursina.camera.display_region.setClearColor((0, 0, 0, 1))
        ursina.camera.display_region.setClearColorActive(True)
        ursina.camera.display_region.setClearDepthActive(True)
        core.ClockObject.getGlobalClock().setMode(core.ClockObject.MNonRealTime)
        core.ClockObject.getGlobalClock().setDt(1 / 60)
        for entity in ursina.scene.entities:
            if entity.name == "simulation_loop":
                entity.ignore = True

        def measure(phase: str) -> dict[str, Any]:
            analyzer = core.SceneGraphAnalyzer()
            analyzer.addNode(ursina.scene.node())
            result: dict[str, Any] = {
                "scene_entities": len(ursina.scene.entities),
                "geoms": analyzer.getNumGeoms(),
                "nodes": analyzer.getNumNodes(),
            }
            for view in (
                CameraView.TOP_DOWN,
                CameraView.THREE_QUARTER,
                CameraView.HELICOPTER,
                CameraView.DRONE,
                CameraView.FOLLOW,
            ):
                apply_camera_view(
                    ursina=ursina,
                    view=view,
                    target=app.racing_robot.chassis_np,
                    track_model=resolved.model,
                    follow_settings=(
                        FORMULA_DRONE_CAMERA_SETTINGS if view is CameraView.DRONE else FORMULA_FOLLOW_CAMERA_SETTINGS
                    ),
                )
                for _ in range(30):
                    app.step()
                output = args.output / f"{phase}-{view.value}.png"
                if app.screenshot(namePrefix=str(output.resolve()), defaultFilename=False) is None:
                    raise RuntimeError(f"Failed to capture {output}")
                timings: list[float] = []
                for _ in range(args.frames):
                    start = perf_counter()
                    app.step()
                    timings.append((perf_counter() - start) * 1000)
                result[view.value] = {
                    "median_ms": median(timings),
                    "p95_ms": sorted(timings)[min(len(timings) - 1, int(len(timings) * 0.95))],
                }
            print(f"{phase}: {json.dumps(result)}", flush=True)
            return result

        results = {"track": args.track, "frames_per_view": args.frames, "before": measure("before")}
        for kwargs in pending:
            batch(**kwargs)
        results["after"] = measure("after")
        (args.output / "benchmark.json").write_text(json.dumps(results, indent=2) + "\n")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
