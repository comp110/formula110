"""Presentation time for a standing start, independent of race simulation time."""

from dataclasses import dataclass

START_LIGHT_COUNT = 5
START_LIGHT_INTERVAL_SECONDS = 1.0
START_LIGHTS_HOLD_SECONDS = 1.5
LIGHTS_OUT_SECONDS = START_LIGHT_COUNT * START_LIGHT_INTERVAL_SECONDS + START_LIGHTS_HOLD_SECONDS
LIGHTS_OUT_DISPLAY_SECONDS = 0.65
GRID_CAMERA_TRANSITION_SECONDS = (START_LIGHT_COUNT - 2) * START_LIGHT_INTERVAL_SECONDS
GRID_OVERVIEW_HOLD_SECONDS = 2.0


@dataclass(slots=True)
class RaceStartSequence:
    """Illuminate five lights, hold, then release every car simultaneously."""

    elapsed_seconds: float = 0.0
    waiting_for_start: bool = False
    overview_hold_seconds: float = 0.0

    @property
    def countdown_elapsed_seconds(self) -> float:
        """Keep the camera move and lights on one clock, after the overhead hold."""
        return max(0.0, self.elapsed_seconds - self.overview_hold_seconds)

    @property
    def holding_overview(self) -> bool:
        return not self.waiting_for_start and self.elapsed_seconds < self.overview_hold_seconds

    @property
    def racing(self) -> bool:
        return not self.waiting_for_start and self.countdown_elapsed_seconds >= LIGHTS_OUT_SECONDS

    @property
    def light_count(self) -> int:
        if self.racing:
            return 0
        return min(START_LIGHT_COUNT, int(self.countdown_elapsed_seconds / START_LIGHT_INTERVAL_SECONDS))

    @property
    def visible(self) -> bool:
        return (
            not self.waiting_for_start and not self.holding_overview
            and self.countdown_elapsed_seconds < LIGHTS_OUT_SECONDS + LIGHTS_OUT_DISPLAY_SECONDS
        )

    def reset(self, *, wait_for_start: bool = False) -> None:
        self.elapsed_seconds = 0.0
        self.waiting_for_start = wait_for_start

    def begin(self) -> None:
        """Release the presentation clock without restarting an active countdown."""
        self.waiting_for_start = False

    def advance(self, delta_seconds: float) -> float:
        """Return only the part of this frame after lights-out for simulation."""
        if self.waiting_for_start:
            return 0.0
        before = max(0.0, self.countdown_elapsed_seconds - LIGHTS_OUT_SECONDS)
        self.elapsed_seconds = round(self.elapsed_seconds + max(0.0, delta_seconds), 12)
        return max(0.0, self.countdown_elapsed_seconds - LIGHTS_OUT_SECONDS) - before
