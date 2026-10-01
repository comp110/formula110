from __future__ import annotations

import wave
from importlib import resources
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from racing.game.config import RacingAudioConfig
from racing.race.start import GRID_OVERVIEW_HOLD_SECONDS, LIGHTS_OUT_SECONDS, RaceStartSequence
from racing.sound.audio import START_LIGHT_AUDIO_FILENAME, NullRacingAudioRuntime, RacingAudioRuntime


def test_five_lights_hold_then_extinguish_together() -> None:
    start = RaceStartSequence()
    assert start.light_count == 0
    for count in range(1, 6):
        assert start.advance(1.0) == 0.0
        assert start.light_count == count
        assert not start.racing
    assert start.advance(1.49) == 0.0
    assert start.light_count == 5
    assert start.advance(0.01) == 0.0
    assert start.racing
    assert start.light_count == 0
    assert start.visible
    assert start.advance(0.65) == pytest.approx(0.65)
    assert not start.visible


@pytest.mark.parametrize("frame_seconds", [1 / 30, 1 / 60, 1 / 144, 0.25])
def test_only_post_start_time_reaches_simulation(frame_seconds: float) -> None:
    start = RaceStartSequence()
    simulated = 0.0
    transitions: list[int] = []
    for _ in range(round(8.0 / frame_seconds)):
        previous = start.light_count
        simulated += start.advance(frame_seconds)
        if start.light_count > previous:
            transitions.append(start.light_count)
        if not start.racing:
            assert simulated == 0.0
    assert transitions == [1, 2, 3, 4, 5]
    assert simulated == pytest.approx(8.0 - LIGHTS_OUT_SECONDS)


def test_frame_crossing_lights_out_only_releases_remaining_time() -> None:
    start = RaceStartSequence(elapsed_seconds=LIGHTS_OUT_SECONDS - 0.1)
    assert start.advance(0.25) == pytest.approx(0.15)
    assert start.advance(0.25) == pytest.approx(0.25)


def test_next_race_resets_the_entire_start_sequence() -> None:
    start = RaceStartSequence()
    start.advance(20.0)
    start.reset()
    assert not start.racing
    assert start.visible
    assert start.light_count == 0
    assert start.advance(1.0) == 0.0
    assert start.light_count == 1


def test_starting_grid_waits_indefinitely_and_each_round_requires_a_new_start() -> None:
    start = RaceStartSequence(waiting_for_start=True)
    assert start.advance(600.0) == 0.0
    assert start.elapsed_seconds == 0.0
    assert not start.visible
    assert not start.racing
    start.begin()
    assert start.visible
    assert start.advance(3.0) == 0.0
    assert start.light_count == 3
    start.begin()
    assert start.elapsed_seconds == 3.0
    assert start.advance(4.0) == pytest.approx(0.5)
    start.reset(wait_for_start=True)
    assert start.waiting_for_start
    assert start.advance(60.0) == 0.0


@pytest.mark.parametrize("frame_seconds", [1 / 30, 1 / 60, 1 / 144, 0.25])
def test_overview_hold_delays_lights_camera_clock_and_simulation(frame_seconds: float) -> None:
    start = RaceStartSequence(waiting_for_start=True, overview_hold_seconds=GRID_OVERVIEW_HOLD_SECONDS)
    start.advance(20.0)
    start.begin()
    for _ in range(round(GRID_OVERVIEW_HOLD_SECONDS / frame_seconds) - 1):
        assert start.advance(frame_seconds) == 0.0
        assert start.holding_overview
        assert start.countdown_elapsed_seconds == 0.0
        assert start.light_count == 0
        assert not start.visible
    start.advance(frame_seconds + 1e-9)
    assert not start.holding_overview
    assert start.visible
    assert start.light_count == 0
    assert start.advance(3.0) == 0.0
    assert start.light_count == 3
    assert start.advance(LIGHTS_OUT_SECONDS - 3.0 + 0.1) == pytest.approx(0.1)
    assert start.racing
    start.reset(wait_for_start=True)
    start.begin()
    assert start.advance(1.5) == 0.0
    assert start.holding_overview


def test_frame_crossing_overview_hold_only_advances_remaining_countdown_time() -> None:
    start = RaceStartSequence(elapsed_seconds=1.9, overview_hold_seconds=GRID_OVERVIEW_HOLD_SECONDS)
    assert start.advance(0.25) == 0.0
    assert start.countdown_elapsed_seconds == pytest.approx(0.15)
    assert start.advance(LIGHTS_OUT_SECONDS) == pytest.approx(0.15)


def test_beep_is_one_shot_cached_and_respects_mute(monkeypatch: pytest.MonkeyPatch) -> None:
    manager_class = Mock()
    monkeypatch.setattr(
        "racing.sound.audio.import_module", Mock(return_value=SimpleNamespace(Audio3DManager=manager_class))
    )
    loader = Mock()
    ursina = SimpleNamespace(
        application=SimpleNamespace(base=SimpleNamespace(loader=loader, sfxManagerList=[Mock()])),
        camera=Mock(), scene=Mock(),
    )
    runtime = RacingAudioRuntime(ursina=ursina, config=RacingAudioConfig(music_enabled=False))
    sound = loader.loadSfx.return_value
    try:
        runtime.set_muted(True)
        runtime.play_start_light_beep()
        loader.loadSfx.assert_not_called()
        runtime.toggle_muted()
        for _ in range(5):
            runtime.play_start_light_beep()
        loader.loadSfx.assert_called_once()
        assert loader.loadSfx.call_args.args[0].endswith(START_LIGHT_AUDIO_FILENAME)
        sound.setLoop.assert_called_once_with(False)
        assert sound.play.call_count == 5
        runtime.toggle_muted()
        sound.stop.assert_called_once()
        runtime.play_start_light_beep()
        assert sound.play.call_count == 5
    finally:
        runtime.destroy()


def test_no_audio_runtime_accepts_start_cues() -> None:
    runtime = NullRacingAudioRuntime()
    runtime.play_start_light_beep()
    assert not runtime.enabled


def test_bundled_beep_is_short_nonempty_pcm() -> None:
    with (
        resources.files("racing").joinpath("assets", "audio", START_LIGHT_AUDIO_FILENAME).open("rb") as source,
        wave.open(source) as beep,
    ):
        assert beep.getnchannels() == 1
        assert beep.getsampwidth() == 2
        assert beep.getnframes() / beep.getframerate() == pytest.approx(0.37)
        assert any(beep.readframes(beep.getnframes()))
