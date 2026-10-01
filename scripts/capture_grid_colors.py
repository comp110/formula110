"""Verify fixed car/HUD colors across overtakes, recovery, round resets, and results."""

from __future__ import annotations

import argparse
from importlib import import_module
from math import isclose
from pathlib import Path
from typing import Any, cast

from racing.game.app import create_heat_viewer_app
from racing.game.config import CameraView, HeatViewerConfig, RacingAudioConfig
from racing.race.heat import HeatEntrant
from racing.race.rules import HeadToHeadRaceRules
from racing.race.start import LIGHTS_OUT_SECONDS
from racing.race.timing import TimingSample
from racing.student.api import RobotCommand, RobotSensors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/grid-colors"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    def controller(_sensors: RobotSensors) -> RobotCommand:
        return RobotCommand()

    app = cast(
        Any,
        create_heat_viewer_app(
            HeatViewerConfig(
                title="Fixed starting grid colors",
                starting_grid=True,
                camera_view=CameraView.THREE_QUARTER,
                round_seconds=0.1,
                race_count=2,
                window_type="offscreen",
                vsync=False,
                audio=RacingAudioConfig(enabled=False),
                rules=HeadToHeadRaceRules(damage_enabled=False),
                entrants=tuple(
                    HeatEntrant(f"Yellow {index + 1}", controller, (1.0, 0.8, 0.1, 1.0)) for index in range(4)
                ),
            )
        ),
    )
    try:
        ursina = cast(Any, import_module("ursina"))
        for _ in range(3):
            app.step()
        loop = next(entity for entity in ursina.scene.entities if entity.name == "head_to_head_viewer_loop")
        initial = dict(app.racing_display_colors)
        ids = tuple(row.car_id for row in app.racing_timing.rows)
        assert len(set(initial.values())) == len(ids)

        def verify_colors() -> None:
            assert app.racing_display_colors == initial
            for entry, runtime in zip(app.racing_entries, app.racing_runtimes, strict=True):
                expected = initial[entry.car_id]
                assert runtime.robot.team_paint_entities
                colors = [
                    *(entity.model.getColorScale() for entity in runtime.robot.team_paint_entities),
                    app.racing_timing_tower.row_buttons[entry.car_id].find("**/timing-tower-team-stripe").getColor(),
                ]
                for color in colors:
                    assert all(isclose(a, b, abs_tol=0.002) for a, b in zip(color, expected, strict=True))
                assert all(
                    isclose(a, b, abs_tol=0.002)
                    for a, b in zip(
                        tuple(runtime.label.background.getColor())[:3],
                        expected[:3],
                        strict=True,
                    )
                )

        def capture(name: str) -> None:
            for _ in range(3):
                app.graphicsEngine.renderFrame()
            path = args.output / f"{name}.png"
            assert app.screenshot(namePrefix=str(path.resolve()), defaultFilename=False) is not None
            print(path)

        def update(delta: float = 0.0) -> None:
            ursina.time.dt = delta
            loop.update()

        def release_grid() -> None:
            loop.input("space")
            sequence = app.racing_start_sequence
            sequence.elapsed_seconds = sequence.overview_hold_seconds + LIGHTS_OUT_SECONDS
            update()

        verify_colors()
        capture("01-starting-grid")
        release_grid()
        capture("02-before-overtake")
        # Drive the real timing model through position changes, without a physics tick.
        for elapsed, order in enumerate((tuple(reversed(ids)), ids[1:] + ids[:1]), start=1):
            app.racing_timing.update(
                elapsed_seconds=0.0,
                samples=tuple(
                    TimingSample(car_id, 100.0 - rank * 5, discontinuity=elapsed == 2)
                    for rank, car_id in enumerate(order)
                ),
            )
            update()
            assert tuple(row.car_id for row in app.racing_timing.rows) == order
            verify_colors()
        capture("03-after-overtakes")

        update(0.15)
        assert app.racing_start_sequence.waiting_for_start
        verify_colors()
        capture("04-next-grid")
        release_grid()
        update(0.15)
        assert hasattr(app, "racing_result")
        for _ in range(7):
            update(0.25)
        verify_colors()
        capture("05-results")
        print("Verified paint, label and timing-stripe colors stay fixed through position changes and round reset.")
    finally:
        app.destroy()


if __name__ == "__main__":
    main()
