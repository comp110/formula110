"""Runtime audio support for graphical racing scenes."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib import import_module, resources
from math import exp, sqrt
from typing import Any, Protocol, cast

from racing.game.config import RacingAudioConfig
from racing.physics import RobotVehicle
from racing.student.api import RobotCommand
from racing.track.spatial import track_forward_vector, track_left_vector
from racing.track.world import clamp

ENGINE_AUDIO_FILENAME = "f1_engine_loop.wav"
MUSIC_AUDIO_FILENAME = "berlin_town_music.wav"
START_LIGHT_AUDIO_FILENAME = "f1_start_light_beep.wav"
TIRE_SQUEAL_AUDIO_FILENAMES = ("tire_squeal_1.wav", "tire_squeal_2.wav", "tire_squeal_3.wav")
FORMULA_ENGINE_AUDIO_FILENAMES = ("formula_engine_body_loop.wav",)
MUTE_BUTTON_TEXT = "Audio On"
MUTED_BUTTON_TEXT = "Muted"

MIN_ENGINE_PLAY_RATE = 0.58
MAX_ENGINE_PLAY_RATE = 2.32
ENGINE_REFERENCE_SPEED_KMH = 210.0
ENGINE_PLAY_RATE_RESPONSE_SECONDS = 0.48
ENGINE_VOLUME_RESPONSE_SECONDS = 0.18
TIRE_SQUEAL_ATTACK_SECONDS = 0.030
TIRE_SQUEAL_RELEASE_SECONDS = 0.075
TIRE_SQUEAL_REFERENCE_SPEED_MPS = 18.0
TIRE_SQUEAL_SLIP_THRESHOLD_MULTIPLIER = 4.0
TIRE_SQUEAL_SLIP_START_MPS = 0.55 * TIRE_SQUEAL_SLIP_THRESHOLD_MULTIPLIER
TIRE_SQUEAL_SLIP_FULL_MPS = 5.8 * TIRE_SQUEAL_SLIP_THRESHOLD_MULTIPLIER
TIRE_SQUEAL_SKID_START = 0.05
TIRE_SQUEAL_SKID_FULL_RANGE = 0.65
TIRE_SQUEAL_NO_BRAKE_SKID_REQUIREMENT_MULTIPLIER = 2.0
TIRE_SQUEAL_DECEL_START_MPS2 = 8.0
TIRE_SQUEAL_DECEL_FULL_MPS2 = 38.0
FORMULA_GEAR_TOP_SPEEDS_MPS = (8.0, 16.0, 28.0, 42.0, 58.0)
FORMULA_SHIFT_UP_RPM = 0.93
FORMULA_SHIFT_DOWN_RPM = 0.42
FORMULA_SHIFT_THROTTLE_THRESHOLD = 0.20
FORMULA_RPM_ATTACK_SECONDS = 0.050
FORMULA_RPM_RELEASE_SECONDS = 0.075
FORMULA_LOAD_RESPONSE_SECONDS = 0.070
FORMULA_HIGH_RPM_PITCH_START = 0.66


@dataclass(frozen=True, slots=True)
class EngineAudioState:
    """Computed sound controls for one car engine loop."""

    play_rate: float
    volume: float


@dataclass(frozen=True, slots=True)
class FormulaEngineRuntimeState:
    """Persistent state for the formula engine synth layers."""

    gear: int = 1
    rpm: float = 0.34
    load: float = 0.0


@dataclass(frozen=True, slots=True)
class FormulaEngineAudioState:
    """Computed sound controls for formula engine layers."""

    runtime_state: FormulaEngineRuntimeState
    layer_play_rates: tuple[float, ...]
    layer_volumes: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class TireSquealAudioState:
    """Computed sound controls for tire squeal loops."""

    intensity: float


@dataclass(frozen=True, slots=True)
class TireMotionState:
    """Motion signals used to drive tire audio."""

    speed_mps: float
    forward_speed_mps: float
    lateral_slip_mps: float
    wheel_skid_loss: float


@dataclass(slots=True)
class AudioKeyToggleState:
    """Edge-trigger state for the mute hotkey."""

    mute_key_was_down: bool = False


class RacingAudioRuntimeLike(Protocol):
    """Common runtime shape for active and null racing audio."""

    @property
    def enabled(self) -> bool:
        """Whether audio was created for the scene."""
        ...

    @property
    def muted(self) -> bool:
        """Whether all racing audio is currently muted."""
        ...

    def register_vehicle(self, robot: RobotVehicle) -> None:
        """Register one robot as an engine sound source."""
        ...

    def record_command(self, robot: RobotVehicle, command: RobotCommand) -> None:
        """Record the most recent drive command for a robot."""
        ...

    def update(self, delta_seconds: float) -> None:
        """Update sound positions, pitch, and volume for one frame."""
        ...

    def play_start_light_beep(self) -> None:
        """Play the non-spatial cue for one newly illuminated start light."""
        ...

    def set_muted(self, muted: bool) -> None:
        """Turn all simulator audio on or off."""
        ...

    def toggle_muted(self) -> bool:
        """Toggle global mute state and return the new state."""
        ...

    def button_text(self) -> str:
        """Choose the text shown on the audio mute button."""
        ...


@dataclass(slots=True)
class EngineAudioEmitter:
    """One registered car engine sound source."""

    robot: RobotVehicle
    engine_sound: Any | None = None
    formula_engine_sounds: tuple[Any, ...] = ()
    tire_squeal_sounds: tuple[Any, ...] = ()
    command: RobotCommand = field(default_factory=RobotCommand)
    engine_play_rate: float = 1.0
    engine_volume: float = 0.0
    formula_engine_state: FormulaEngineRuntimeState = field(default_factory=FormulaEngineRuntimeState)
    tire_squeal_intensity: float = 0.0
    previous_forward_speed_mps: float | None = None
    initialized: bool = False


class RacingAudioRuntime:
    """Audio runtime that plays spatial engine, tire, and music loops."""

    def __init__(self, *, ursina: Any, config: RacingAudioConfig) -> None:
        self._config = config
        self._muted = config.muted
        self._resource_stack = ExitStack()
        self._emitters: dict[int, EngineAudioEmitter] = {}

        base = ursina.application.base
        self._loader = base.loader
        self._start_light_sound: Any | None = None
        audio3d_module = cast(Any, import_module("direct.showbase.Audio3DManager"))
        audio3d_manager_class = audio3d_module.Audio3DManager
        self._audio3d = audio3d_manager_class(
            base.sfxManagerList[0],
            listener_target=ursina.camera,
            root=ursina.scene,
        )
        self._apply_spatial_settings()
        self._engine_path = self._resource_path(ENGINE_AUDIO_FILENAME)
        self._formula_engine_paths = tuple(self._resource_path(filename) for filename in FORMULA_ENGINE_AUDIO_FILENAMES)
        self._tire_squeal_paths = tuple(self._resource_path(filename) for filename in TIRE_SQUEAL_AUDIO_FILENAMES)

        self._music_sound: Any | None = None
        if config.music_enabled:
            music_path = self._resource_path(MUSIC_AUDIO_FILENAME)
            self._music_sound = base.loader.loadSfx(music_path)
            _call_if_available(self._music_sound, "setLoop", True)
            _call_if_available(self._music_sound, "setVolume", self._music_volume())
            _call_if_available(self._music_sound, "play")

    @property
    def enabled(self) -> bool:
        """Whether audio was created for the scene."""
        return True

    @property
    def muted(self) -> bool:
        """Whether all racing audio is currently muted."""
        return self._muted

    def register_vehicle(self, robot: RobotVehicle) -> None:
        """Register one robot as an engine sound source."""
        robot_id = id(robot)
        if robot_id in self._emitters:
            return
        engine_sound: Any | None = None
        formula_engine_sounds = tuple(
            self._load_spatial_loop(
                path=path,
                robot=robot,
                min_distance=self._config.engine_min_distance_m,
                max_distance=self._config.engine_max_distance_m,
            )
            for path in self._formula_engine_paths
        )
        tire_squeal_sounds = tuple(
            self._load_spatial_loop(
                path=path,
                robot=robot,
                min_distance=self._config.tire_squeal_min_distance_m,
                max_distance=self._config.tire_squeal_max_distance_m,
            )
            for path in self._tire_squeal_paths
        )
        self._emitters[robot_id] = EngineAudioEmitter(
            robot=robot,
            engine_sound=engine_sound,
            formula_engine_sounds=formula_engine_sounds,
            tire_squeal_sounds=tire_squeal_sounds,
        )

    def record_command(self, robot: RobotVehicle, command: RobotCommand) -> None:
        """Record the most recent drive command for a robot."""
        emitter = self._emitters.get(id(robot))
        if emitter is not None:
            emitter.command = command

    def update(self, delta_seconds: float) -> None:
        """Update sound positions, pitch, and volume for one frame."""
        for emitter in self._emitters.values():
            motion_state = _robot_motion_state(emitter.robot)
            forward_deceleration_mps2 = _forward_deceleration_mps2(
                previous_forward_speed_mps=emitter.previous_forward_speed_mps,
                current_forward_speed_mps=motion_state.forward_speed_mps,
                delta_seconds=delta_seconds,
            )
            emitter.previous_forward_speed_mps = motion_state.forward_speed_mps
            tire_state = tire_squeal_audio_state_for_robot(
                speed_mps=motion_state.speed_mps,
                lateral_slip_mps=motion_state.lateral_slip_mps,
                wheel_skid_loss=motion_state.wheel_skid_loss,
                forward_deceleration_mps2=forward_deceleration_mps2,
                command=emitter.command,
                eliminated=emitter.robot.eliminated,
                config=self._config,
                muted=self._muted,
            )
            formula_state = formula_engine_audio_state_for_robot(
                speed_mps=motion_state.speed_mps,
                command=emitter.command,
                previous_state=emitter.formula_engine_state,
                delta_seconds=delta_seconds,
                eliminated=emitter.robot.eliminated,
                config=self._config,
                muted=self._muted,
            )
            emitter.formula_engine_state = formula_state.runtime_state
            self._update_tire_smoothing(emitter=emitter, tire_state=tire_state, delta_seconds=delta_seconds)
            _apply_formula_engine_audio_state(emitter=emitter, state=formula_state)
            for index, sound in enumerate(emitter.tire_squeal_sounds):
                volume = _tire_squeal_layer_volume(
                    intensity=emitter.tire_squeal_intensity,
                    layer_index=index,
                    config=self._config,
                )
                _call_if_available(sound, "setPlayRate", 0.88 + emitter.tire_squeal_intensity * 0.24 + index * 0.035)
                _call_if_available(sound, "setVolume", volume)
        if self._music_sound is not None:
            _call_if_available(self._music_sound, "setVolume", self._music_volume())
        self._audio3d.update()

    def play_start_light_beep(self) -> None:
        """Load once and retrigger a single beep in sync with each red light."""
        if self._muted:
            return
        if self._start_light_sound is None:
            self._start_light_sound = self._loader.loadSfx(self._resource_path(START_LIGHT_AUDIO_FILENAME))
            _call_if_available(self._start_light_sound, "setLoop", False)
        _call_if_available(self._start_light_sound, "setVolume", clamp(self._config.master_volume, 0.0, 1.0))
        _call_if_available(self._start_light_sound, "play")

    def set_muted(self, muted: bool) -> None:
        """Turn all simulator audio on or off."""
        self._muted = muted
        if muted and self._start_light_sound is not None:
            _call_if_available(self._start_light_sound, "stop")

    def toggle_muted(self) -> bool:
        """Toggle global mute state and return the new state."""
        self.set_muted(not self._muted)
        return self._muted

    def button_text(self) -> str:
        """Choose the text shown on the audio mute button."""
        return MUTED_BUTTON_TEXT if self._muted else MUTE_BUTTON_TEXT

    def destroy(self) -> None:
        """Release temporary resource handles used by importlib.resources."""
        if self._start_light_sound is not None:
            _call_if_available(self._start_light_sound, "stop")
        self._resource_stack.close()

    def _apply_spatial_settings(self) -> None:
        self._audio3d.setDopplerFactor(self._config.doppler_factor)
        self._audio3d.setDistanceFactor(self._config.distance_factor)
        self._audio3d.setDropOffFactor(self._config.drop_off_factor)
        self._audio3d.setListenerVelocityAuto()

    def _resource_path(self, filename: str) -> str:
        resource = resources.files("racing").joinpath("assets", "audio", filename)
        return str(self._resource_stack.enter_context(resources.as_file(resource)))

    def _music_volume(self) -> float:
        if self._muted or not self._config.music_enabled:
            return 0.0
        return clamp(self._config.master_volume, 0.0, 1.0) * clamp(self._config.music_volume, 0.0, 1.0)

    def _load_spatial_loop(self, *, path: str, robot: RobotVehicle, min_distance: float, max_distance: float) -> Any:
        sound = self._audio3d.loadSfx(path)
        _call_if_available(sound, "setLoop", True)
        _call_if_available(sound, "setVolume", 0.0)
        self._audio3d.attachSoundToObject(sound, robot.chassis_np)
        self._audio3d.setSoundVelocityAuto(sound)
        self._audio3d.setSoundMinDistance(sound, min_distance)
        self._audio3d.setSoundMaxDistance(sound, max_distance)
        _call_if_available(sound, "play")
        return sound

    def _update_emitter_smoothing(
        self,
        *,
        emitter: EngineAudioEmitter,
        engine_state: EngineAudioState,
        tire_state: TireSquealAudioState,
        delta_seconds: float,
    ) -> None:
        if not emitter.initialized:
            emitter.engine_play_rate = engine_state.play_rate
            emitter.engine_volume = engine_state.volume
            emitter.tire_squeal_intensity = tire_state.intensity
            emitter.initialized = True
            return
        emitter.engine_play_rate = _smoothed_value(
            current=emitter.engine_play_rate,
            target=engine_state.play_rate,
            delta_seconds=delta_seconds,
            response_seconds=ENGINE_PLAY_RATE_RESPONSE_SECONDS,
        )
        emitter.engine_volume = _smoothed_value(
            current=emitter.engine_volume,
            target=engine_state.volume,
            delta_seconds=delta_seconds,
            response_seconds=ENGINE_VOLUME_RESPONSE_SECONDS,
        )
        tire_response_seconds = (
            TIRE_SQUEAL_ATTACK_SECONDS
            if tire_state.intensity > emitter.tire_squeal_intensity
            else TIRE_SQUEAL_RELEASE_SECONDS
        )
        emitter.tire_squeal_intensity = _smoothed_value(
            current=emitter.tire_squeal_intensity,
            target=tire_state.intensity,
            delta_seconds=delta_seconds,
            response_seconds=tire_response_seconds,
        )

    def _update_tire_smoothing(
        self,
        *,
        emitter: EngineAudioEmitter,
        tire_state: TireSquealAudioState,
        delta_seconds: float,
    ) -> None:
        if not emitter.initialized:
            emitter.tire_squeal_intensity = tire_state.intensity
            emitter.initialized = True
            return
        tire_response_seconds = (
            TIRE_SQUEAL_ATTACK_SECONDS
            if tire_state.intensity > emitter.tire_squeal_intensity
            else TIRE_SQUEAL_RELEASE_SECONDS
        )
        emitter.tire_squeal_intensity = _smoothed_value(
            current=emitter.tire_squeal_intensity,
            target=tire_state.intensity,
            delta_seconds=delta_seconds,
            response_seconds=tire_response_seconds,
        )


class NullRacingAudioRuntime:
    """No-op racing audio runtime used when audio is disabled or unavailable."""

    def __init__(self, *, muted: bool = True) -> None:
        self._muted = muted

    @property
    def enabled(self) -> bool:
        """Whether audio was created for the scene."""
        return False

    @property
    def muted(self) -> bool:
        """Whether all racing audio is currently muted."""
        return self._muted

    def register_vehicle(self, robot: RobotVehicle) -> None:
        """Register one robot as an engine sound source."""
        _ = robot

    def record_command(self, robot: RobotVehicle, command: RobotCommand) -> None:
        """Record the most recent drive command for a robot."""
        _ = robot, command

    def update(self, delta_seconds: float) -> None:
        """Update sound positions, pitch, and volume for one frame."""
        _ = delta_seconds

    def play_start_light_beep(self) -> None:
        """Keep the visual countdown available without an audio device."""

    def set_muted(self, muted: bool) -> None:
        """Turn all simulator audio on or off."""
        self._muted = muted

    def toggle_muted(self) -> bool:
        """Toggle global mute state and return the new state."""
        self._muted = not self._muted
        return self._muted

    def button_text(self) -> str:
        """Choose the text shown on the audio mute button."""
        return MUTED_BUTTON_TEXT if self._muted else MUTE_BUTTON_TEXT


def create_racing_audio_runtime(
    *,
    ursina: Any,
    config: RacingAudioConfig,
) -> RacingAudioRuntimeLike:
    """Create the best available racing audio runtime for a graphical scene."""
    if not config.enabled:
        return NullRacingAudioRuntime(muted=True)
    base = getattr(getattr(ursina, "application", None), "base", None)
    sfx_managers = getattr(base, "sfxManagerList", None)
    if not sfx_managers:
        return NullRacingAudioRuntime(muted=config.muted)
    loader = getattr(base, "loader", None)
    if loader is None:
        return NullRacingAudioRuntime(muted=config.muted)
    try:
        return RacingAudioRuntime(ursina=ursina, config=config)
    except (AttributeError, IndexError, ImportError):
        return NullRacingAudioRuntime(muted=config.muted)


def engine_audio_state_for_robot(
    *,
    speed_kmh: float,
    command: RobotCommand,
    eliminated: bool,
    config: RacingAudioConfig,
    muted: bool,
) -> EngineAudioState:
    """Return the engine loop pitch and volume for one robot."""
    abs_speed_kmh = abs(speed_kmh)
    speed_amount = clamp(abs_speed_kmh / ENGINE_REFERENCE_SPEED_KMH, 0.0, 1.0)
    drive_load, brake_load = _signed_throttle_loads(
        speed_mps=speed_kmh / 3.6,
        throttle=command.throttle,
    )

    play_rate = clamp(
        0.64 + speed_amount * 1.36 + drive_load * 0.32 - brake_load * 0.08,
        MIN_ENGINE_PLAY_RATE,
        MAX_ENGINE_PLAY_RATE,
    )
    if eliminated or muted:
        return EngineAudioState(play_rate=play_rate, volume=0.0)

    drive_amount = max(speed_amount, drive_load * 0.35)
    raw_volume = 0.18 + drive_amount * 0.64 + drive_load * 0.28
    damped_volume = raw_volume * (1.0 - brake_load * 0.28)
    volume = (
        clamp(config.master_volume, 0.0, 1.0) * clamp(config.engine_volume, 0.0, 1.0) * clamp(damped_volume, 0.0, 1.0)
    )
    return EngineAudioState(play_rate=play_rate, volume=volume)


def formula_engine_audio_state_for_robot(
    *,
    speed_mps: float,
    command: RobotCommand,
    previous_state: FormulaEngineRuntimeState,
    delta_seconds: float,
    eliminated: bool,
    config: RacingAudioConfig,
    muted: bool,
) -> FormulaEngineAudioState:
    """Return modern formula engine layer controls for one robot."""
    drive_load, _brake_load = _signed_throttle_loads(
        speed_mps=speed_mps,
        throttle=command.throttle,
    )
    abs_speed_mps = abs(speed_mps)
    gear, _ = _formula_engine_gear(
        speed_mps=abs_speed_mps,
        throttle_load=drive_load,
        previous_state=previous_state,
    )
    target_rpm = _formula_engine_target_rpm(speed_mps=abs_speed_mps, throttle_load=drive_load, gear=gear)
    rpm_response_seconds = (
        FORMULA_RPM_ATTACK_SECONDS if target_rpm > previous_state.rpm else FORMULA_RPM_RELEASE_SECONDS
    )
    rpm = _smoothed_value(
        current=previous_state.rpm,
        target=target_rpm,
        delta_seconds=delta_seconds,
        response_seconds=rpm_response_seconds,
    )
    target_load = drive_load
    load = _smoothed_value(
        current=previous_state.load,
        target=target_load,
        delta_seconds=delta_seconds,
        response_seconds=FORMULA_LOAD_RESPONSE_SECONDS,
    )
    runtime_state = FormulaEngineRuntimeState(
        gear=gear,
        rpm=rpm,
        load=load,
    )

    master_volume = (
        0.0 if eliminated or muted else clamp(config.master_volume, 0.0, 1.0) * clamp(config.engine_volume, 0.0, 1.0)
    )
    body_volume = master_volume * (0.16 + rpm * 0.31 + load * 0.18)
    layer_volumes = (clamp(body_volume, 0.0, 1.0),)
    layer_play_rates = (_formula_engine_play_rate(rpm),)
    return FormulaEngineAudioState(
        runtime_state=runtime_state,
        layer_play_rates=layer_play_rates,
        layer_volumes=layer_volumes,
    )


def _formula_engine_gear(
    *,
    speed_mps: float,
    throttle_load: float,
    previous_state: FormulaEngineRuntimeState,
) -> tuple[int, bool]:
    gear = min(max(previous_state.gear, 1), len(FORMULA_GEAR_TOP_SPEEDS_MPS))
    raw_rpm = _formula_engine_target_rpm(speed_mps=speed_mps, throttle_load=throttle_load, gear=gear)
    if raw_rpm > FORMULA_SHIFT_UP_RPM and throttle_load > FORMULA_SHIFT_THROTTLE_THRESHOLD:
        next_gear = min(gear + 1, len(FORMULA_GEAR_TOP_SPEEDS_MPS))
        return next_gear, next_gear != gear
    if previous_state.rpm < FORMULA_SHIFT_DOWN_RPM:
        return max(1, gear - 1), False
    return gear, False


def _formula_engine_target_rpm(*, speed_mps: float, throttle_load: float, gear: int) -> float:
    gear_index = min(max(gear, 1), len(FORMULA_GEAR_TOP_SPEEDS_MPS)) - 1
    gear_top_speed_mps = FORMULA_GEAR_TOP_SPEEDS_MPS[gear_index]
    speed_ratio = clamp(speed_mps / gear_top_speed_mps, 0.0, 1.0)
    rpm = 0.28 + speed_ratio * 0.68
    if speed_mps < 4.0:
        rpm = max(rpm, 0.42 + throttle_load * 0.35)
    return clamp(rpm, 0.25, 1.0)


def _formula_engine_play_rate(rpm: float) -> float:
    high_rpm_amount = clamp((rpm - FORMULA_HIGH_RPM_PITCH_START) / (1.0 - FORMULA_HIGH_RPM_PITCH_START), 0.0, 1.0)
    return 0.62 + rpm * 0.62 + high_rpm_amount * high_rpm_amount * 0.32


def _apply_formula_engine_audio_state(*, emitter: EngineAudioEmitter, state: FormulaEngineAudioState) -> None:
    for sound, play_rate, volume in zip(
        emitter.formula_engine_sounds,
        state.layer_play_rates,
        state.layer_volumes,
        strict=False,
    ):
        _call_if_available(sound, "setPlayRate", play_rate)
        _call_if_available(sound, "setVolume", volume)


def tire_squeal_audio_state_for_robot(
    *,
    speed_mps: float,
    lateral_slip_mps: float,
    wheel_skid_loss: float = 0.0,
    forward_deceleration_mps2: float = 0.0,
    command: RobotCommand,
    eliminated: bool,
    config: RacingAudioConfig,
    muted: bool,
) -> TireSquealAudioState:
    """Return tire-squeal intensity for one robot."""
    if eliminated or muted:
        return TireSquealAudioState(intensity=0.0)

    _drive_load, brake_load = _signed_throttle_loads(
        speed_mps=speed_mps,
        throttle=command.throttle,
    )
    steer_load = clamp(abs(command.steer), 0.0, 1.0)
    speed_amount = clamp(abs(speed_mps) / TIRE_SQUEAL_REFERENCE_SPEED_MPS, 0.0, 1.0)
    slip_amount = clamp(
        (abs(lateral_slip_mps) - TIRE_SQUEAL_SLIP_START_MPS) / (TIRE_SQUEAL_SLIP_FULL_MPS - TIRE_SQUEAL_SLIP_START_MPS),
        0.0,
        1.0,
    )
    skid_requirement_multiplier = 1.0 + (1.0 - brake_load) * (TIRE_SQUEAL_NO_BRAKE_SKID_REQUIREMENT_MULTIPLIER - 1.0)
    skid_for_audio = wheel_skid_loss / skid_requirement_multiplier
    skid_amount = clamp(
        (skid_for_audio - TIRE_SQUEAL_SKID_START) / TIRE_SQUEAL_SKID_FULL_RANGE,
        0.0,
        1.0,
    )
    decel_amount = clamp(
        (forward_deceleration_mps2 - TIRE_SQUEAL_DECEL_START_MPS2)
        / (TIRE_SQUEAL_DECEL_FULL_MPS2 - TIRE_SQUEAL_DECEL_START_MPS2),
        0.0,
        1.0,
    )

    brake_speed_amount = clamp((abs(speed_mps) - 1.5) / 10.0, 0.0, 1.0)
    hard_stop_amount = max(skid_amount, decel_amount)
    brake_squeal = (brake_load**0.68) * brake_speed_amount * (0.58 + hard_stop_amount * 0.42)
    skid_squeal = skid_amount * speed_amount * (0.34 + brake_load * 0.50 + steer_load * 0.22)
    slide_squeal = slip_amount * (brake_load * 0.46 + skid_amount * 0.54)
    steering_scrub = (
        clamp((steer_load - 0.58) / 0.42, 0.0, 1.0) * speed_amount * (brake_load * 0.08 + skid_amount * 0.26)
    )
    return TireSquealAudioState(intensity=clamp(max(brake_squeal, skid_squeal, slide_squeal, steering_scrub), 0.0, 1.0))


def _signed_throttle_loads(*, speed_mps: float, throttle: float) -> tuple[float, float]:
    """Split signed throttle into active drive and direction-change braking loads."""
    throttle_load = clamp(abs(throttle), 0.0, 1.0)
    direction_change_threshold_mps = 1.0 / 3.6
    opposing_motion = (speed_mps > direction_change_threshold_mps and throttle < 0.0) or (
        speed_mps < -direction_change_threshold_mps and throttle > 0.0
    )
    if opposing_motion:
        return 0.0, throttle_load
    return throttle_load, 0.0


def update_audio_mute_key(
    state: AudioKeyToggleState,
    *,
    mute_key_down: bool,
    audio_runtime: RacingAudioRuntimeLike,
) -> bool:
    """Toggle mute once per key press and return whether a toggle happened."""
    toggled = False
    if mute_key_down and not state.mute_key_was_down:
        audio_runtime.toggle_muted()
        toggled = True
    state.mute_key_was_down = mute_key_down
    return toggled


def _robot_speed_kmh(robot: RobotVehicle) -> float:
    if robot.eliminated:
        return 0.0
    return float(robot.vehicle.getCurrentSpeedKmHour())


def _robot_motion_state(robot: RobotVehicle) -> TireMotionState:
    if robot.eliminated:
        return TireMotionState(speed_mps=0.0, forward_speed_mps=0.0, lateral_slip_mps=0.0, wheel_skid_loss=0.0)
    body = robot.chassis_np.node()
    if not hasattr(body, "getLinearVelocity"):
        speed_mps = _robot_speed_kmh(robot) / 3.6
        return TireMotionState(
            speed_mps=speed_mps,
            forward_speed_mps=speed_mps,
            lateral_slip_mps=0.0,
            wheel_skid_loss=_wheel_skid_loss(robot),
        )
    velocity = body.getLinearVelocity()
    velocity_x = float(velocity[0])
    velocity_z = float(velocity[2])
    speed_mps = sqrt(velocity_x * velocity_x + velocity_z * velocity_z)
    heading_degrees = float(robot.chassis_np.getH())
    forward_x, forward_z = track_forward_vector(heading_degrees)
    left_x, left_z = track_left_vector(heading_degrees)
    forward_speed_mps = velocity_x * forward_x + velocity_z * forward_z
    lateral_slip_mps = velocity_x * left_x + velocity_z * left_z
    return TireMotionState(
        speed_mps=speed_mps,
        forward_speed_mps=forward_speed_mps,
        lateral_slip_mps=lateral_slip_mps,
        wheel_skid_loss=_wheel_skid_loss(robot),
    )


def _forward_deceleration_mps2(
    *,
    previous_forward_speed_mps: float | None,
    current_forward_speed_mps: float,
    delta_seconds: float,
) -> float:
    if previous_forward_speed_mps is None or delta_seconds <= 0.0:
        return 0.0
    return max(0.0, (previous_forward_speed_mps - current_forward_speed_mps) / delta_seconds)


def _wheel_skid_loss(robot: RobotVehicle) -> float:
    wheel_count = len(robot.wheel_nodes)
    if wheel_count <= 0:
        return 0.0
    skid_infos: list[float] = []
    for wheel_index in range(wheel_count):
        try:
            wheel = robot.vehicle.getWheel(wheel_index)
            skid_infos.append(float(wheel.getSkidInfo()))
        except (AttributeError, IndexError, TypeError, ValueError):
            continue
    if not skid_infos:
        return 0.0
    return clamp(1.0 - min(skid_infos), 0.0, 1.0)


def _tire_squeal_layer_volume(*, intensity: float, layer_index: int, config: RacingAudioConfig) -> float:
    thresholds = (0.0, 0.18, 0.48)
    gains = (0.95, 0.72, 0.52)
    threshold = thresholds[min(layer_index, len(thresholds) - 1)]
    gain = gains[min(layer_index, len(gains) - 1)]
    layer_amount = clamp((intensity - threshold) / max(0.001, 1.0 - threshold), 0.0, 1.0)
    return clamp(config.master_volume, 0.0, 1.0) * clamp(config.tire_squeal_volume, 0.0, 1.0) * gain * layer_amount


def _smoothed_value(*, current: float, target: float, delta_seconds: float, response_seconds: float) -> float:
    if delta_seconds <= 0.0 or response_seconds <= 0.0:
        return target
    amount = 1.0 - exp(-delta_seconds / response_seconds)
    return current + (target - current) * amount


def _call_if_available(target: Any, method_name: str, *args: object) -> None:
    method = getattr(target, method_name, None)
    if callable(method):
        method(*args)
