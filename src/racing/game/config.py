"""Small settings objects used when creating simulator apps."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol

from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
    ColorRGBA,
)
from racing.race.heat import HeatEntrant
from racing.race.laps import DEFAULT_FINISH_TIMEOUT_SECONDS
from racing.race.rules import HeadToHeadRaceRules
from racing.race.runtime import DEFAULT_RACE_RANDOM_SEED
from racing.student.api import RobotController
from racing.track.world import TRACK_ID_MUGELLO_SHORT

DEFAULT_RACE_SECONDS = 30.0


class CameraView(Enum):
    """Names for the camera views a player can cycle through."""

    TOP_DOWN = "top_down"
    THREE_QUARTER = "three_quarter"
    DRONE = "drone"
    HELICOPTER = "helicopter"
    CINEMATIC = "cinematic"
    FOLLOW = "follow"
    FOLLOW_CAR = "follow_car"
    SPLIT_FOLLOW = "split_follow"


class CarShowcaseView(Enum):
    """Names for camera poses in the car-only preview scene."""

    TOP = "top"
    THREE_QUARTER = "three_quarter"
    REAR_THREE_QUARTER = "rear_three_quarter"
    FRONT = "front"
    REAR = "rear"
    SIDE = "side"
    UPSIDE_DOWN = "upside_down"


@dataclass(frozen=True, slots=True)
class RacingAudioConfig:
    """Volume and mute settings for simulator sound."""

    enabled: bool = True
    muted: bool = False
    music_enabled: bool = True
    master_volume: float = 0.82
    engine_volume: float = 0.78
    music_volume: float = 0.26
    tire_squeal_volume: float = 0.82
    doppler_factor: float = 0.28
    distance_factor: float = 1.0
    drop_off_factor: float = 1.15
    engine_min_distance_m: float = 4.0
    engine_max_distance_m: float = 95.0
    tire_squeal_min_distance_m: float = 3.0
    tire_squeal_max_distance_m: float = 70.0


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Settings for the single-car playable scene."""

    title: str = "Racing"
    borderless: bool = False
    fullscreen: bool = False
    vsync: bool = True
    development_mode: bool = False
    size: tuple[int, int] = (1280, 720)
    camera_view: CameraView = CameraView.DRONE
    student_controller: RobotController | None = None
    window_type: str | None = None
    fixed_delta_seconds: float = 1 / 60
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    team_color: ColorRGBA = DEFAULT_FORMULA_TEAM_COLOR
    spawn_position: tuple[float, float, float] | None = None
    spawn_heading_degrees: float | None = None
    spawn_progress_distance_m: float | None = None
    human_recording_path: Path | None = None
    audio: RacingAudioConfig = field(default_factory=RacingAudioConfig)


@dataclass(frozen=True, slots=True)
class HeadToHeadViewerConfig:
    """Settings for the graphical race viewer with two controller teams."""

    title: str = "Racing Head-to-Head"
    borderless: bool = False
    fullscreen: bool = False
    vsync: bool = True
    development_mode: bool = False
    size: tuple[int, int] = (1280, 720)
    camera_view: CameraView = CameraView.DRONE
    challenger_name: str = "challenger"
    incumbent_name: str = "incumbent"
    starting_grid: bool = False
    grid_names: tuple[str, ...] = ()
    challenger_controller: RobotController | None = None
    incumbent_controller: RobotController | None = None
    challenger_keyboard: bool = False
    incumbent_keyboard: bool = False
    challenger_copies: int = 1
    incumbent_copies: int = 1
    race_count: int = 1
    round_seconds: float = DEFAULT_RACE_SECONDS
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    win_margin_m: float = 1.0
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60
    window_type: str | None = None
    challenger_team_color: ColorRGBA = DEFAULT_CHALLENGER_TEAM_COLOR
    incumbent_team_color: ColorRGBA = DEFAULT_INCUMBENT_TEAM_COLOR
    audio: RacingAudioConfig = field(default_factory=RacingAudioConfig)


@dataclass(frozen=True, slots=True)
class HeatViewerConfig:
    """Settings for a four-, eight-, or nine-controller race with individual standings."""

    entrants: tuple[HeatEntrant, ...] = ()
    starting_grid: bool = False
    grid_names: tuple[str, ...] = ()
    title: str = "Racing Heat"
    borderless: bool = False
    fullscreen: bool = False
    vsync: bool = True
    development_mode: bool = False
    size: tuple[int, int] = (1280, 720)
    camera_view: CameraView = CameraView.THREE_QUARTER
    race_count: int = 1
    round_seconds: float = DEFAULT_RACE_SECONDS
    round_laps: int | None = None
    finish_timeout_seconds: float = DEFAULT_FINISH_TIMEOUT_SECONDS
    random_seed: int = DEFAULT_RACE_RANDOM_SEED
    track_id: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60
    window_type: str | None = None
    audio: RacingAudioConfig = field(default_factory=RacingAudioConfig)


@dataclass(frozen=True, slots=True)
class CarShowcaseConfig:
    """Settings for the small scene that only shows the car model."""

    title: str = "Racing Car Art Showcase"
    size: tuple[int, int] = (1280, 720)
    view: CarShowcaseView = CarShowcaseView.THREE_QUARTER
    development_mode: bool = True
    window_type: str = "offscreen"
    team_color: ColorRGBA = DEFAULT_FORMULA_TEAM_COLOR


class RunnableApp(Protocol):
    """Small interface shared by Ursina apps returned from this package."""

    def run(self) -> None:
        """Start the app event loop."""

    def setBackgroundColor(self, red: float, green: float, blue: float, alpha: float = 1) -> None:
        """Set the window clear color."""


class WindowLike(Protocol):
    """Protocol for mutable Ursina window attributes configured at startup."""

    title: str
    borderless: bool
    fullscreen: bool
    vsync: bool


class WindowConfigLike(Protocol):
    """Protocol for configuration objects that provide window settings."""

    @property
    def title(self) -> str:
        """Text shown in the OS window title bar."""
        ...

    @property
    def borderless(self) -> bool:
        """Whether the window should be borderless."""
        ...

    @property
    def fullscreen(self) -> bool:
        """Whether the window should be fullscreen."""
        ...

    @property
    def vsync(self) -> bool:
        """Whether vertical sync should be enabled."""
        ...


def configure_window(window: WindowLike, config: WindowConfigLike) -> None:
    """Copy common window settings from a config object to Ursina."""
    window.title = config.title
    window.borderless = config.borderless
    window.fullscreen = config.fullscreen
    window.vsync = config.vsync


def fps_text_for_delta(delta_seconds: float) -> str:
    """Format a frame delta as an FPS display string."""
    if delta_seconds <= 0:
        return "FPS: --"
    return f"FPS: {round(1 / delta_seconds):>3}"


def parse_window_size(value: str) -> tuple[int, int]:
    """Parse a WIDTHxHEIGHT command-line window size."""
    width_text, height_text = value.lower().split("x", 1)
    return int(width_text), int(height_text)


def parse_color_rgba(value: str) -> ColorRGBA:
    """Parse a #RRGGBB, #RRGGBBAA, or comma-separated RGBA color."""
    text = value.strip()
    if text.startswith("#"):
        hex_text = text[1:]
        if len(hex_text) not in (6, 8):
            raise argparse.ArgumentTypeError("hex colors must be #RRGGBB or #RRGGBBAA")
        try:
            channels = tuple(int(hex_text[index : index + 2], 16) / 255 for index in range(0, len(hex_text), 2))
        except ValueError as error:
            raise argparse.ArgumentTypeError("hex colors must use hexadecimal digits") from error
        if len(channels) == 3:
            return (channels[0], channels[1], channels[2], 1.0)
        return (channels[0], channels[1], channels[2], channels[3])

    parts = tuple(part.strip() for part in text.split(","))
    if len(parts) not in (3, 4):
        raise argparse.ArgumentTypeError("colors must have three or four comma-separated channels")
    try:
        channels = tuple(float(part) for part in parts)
    except ValueError as error:
        raise argparse.ArgumentTypeError("color channels must be numbers") from error
    if any(channel < 0.0 or channel > 1.0 for channel in channels):
        raise argparse.ArgumentTypeError("comma-separated color channels must be between 0.0 and 1.0")
    if len(channels) == 3:
        return (channels[0], channels[1], channels[2], 1.0)
    return (channels[0], channels[1], channels[2], channels[3])


def positive_float(value: str) -> float:
    """Parse a positive floating-point command-line value."""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed
