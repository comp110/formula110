"""Command-line parser and main function for the simulator."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from contextlib import nullcontext, redirect_stdout
from pathlib import Path
from typing import cast

from racing.game.app import create_app, create_head_to_head_viewer_app, create_heat_viewer_app
from racing.game.config import (
    DEFAULT_RACE_SECONDS,
    CameraView,
    GameConfig,
    HeadToHeadViewerConfig,
    HeatViewerConfig,
    RacingAudioConfig,
    parse_color_rgba,
    parse_window_size,
)
from racing.graphics.colors import (
    DEFAULT_CHALLENGER_TEAM_COLOR,
    DEFAULT_FORMULA_TEAM_COLOR,
    DEFAULT_INCUMBENT_TEAM_COLOR,
    ColorRGBA,
)
from racing.race.head_to_head import format_head_to_head_result, run_headless_head_to_head
from racing.race.heat import (
    DEFAULT_HEAT_COLORS,
    HEAT_ENTRANT_COUNTS,
    HeatEntrant,
    HeatResult,
    format_heat_result,
    run_headless_heat,
)
from racing.race.rules import HeadToHeadRaceRules, HeadToHeadScoring
from racing.race.runtime import DEFAULT_RACE_RANDOM_SEED
from racing.student.api import StudentControllerSubmission, load_student_submission
from racing.track.procedural import TRACK_ID_PROCEDURAL
from racing.track.world import TRACK_ID_MUGELLO_SHORT, track_layout_ids


def _add_audio_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-audio", action="store_true", help="disable graphical racing audio")
    parser.add_argument("--no-music", action="store_true", help="disable retro music while keeping engine audio")
    parser.add_argument("--muted", action="store_true", help="start graphical racing audio muted")


def _add_color_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--team-color",
        type=parse_color_rgba,
        default=DEFAULT_FORMULA_TEAM_COLOR,
        help="formula car paint color, as #RRGGBB or comma-separated 0.0-1.0 RGB(A)",
    )


def _add_track_arguments(parser: argparse.ArgumentParser, *, suppress_defaults: bool = False) -> None:
    parser.add_argument(
        "--track",
        choices=(*track_layout_ids(), TRACK_ID_PROCEDURAL),
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="track layout; defaults to mugello-short, or procedural when --track-seed is supplied",
    )
    parser.add_argument(
        "--track-seed",
        type=int,
        default=argparse.SUPPRESS if suppress_defaults else None,
        help="generate and reproduce a procedural track with this seed",
    )


def _track_selection_from_args(
    *,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
) -> tuple[str, int | None]:
    requested_track = cast(str | None, getattr(args, "track", None))
    track_seed = cast(int | None, getattr(args, "track_seed", None))
    track_id = TRACK_ID_PROCEDURAL if requested_track is None and track_seed is not None else requested_track
    if track_id is None:
        return TRACK_ID_MUGELLO_SHORT, None
    if track_id == TRACK_ID_PROCEDURAL and track_seed is None:
        parser.error("--track procedural requires --track-seed")
    if track_id != TRACK_ID_PROCEDURAL and track_seed is not None:
        parser.error("--track-seed can only be combined with --track procedural")
    return track_id, track_seed


def _audio_config_from_args(args: argparse.Namespace) -> RacingAudioConfig:
    return RacingAudioConfig(
        enabled=not bool(getattr(args, "no_audio", False)),
        muted=bool(getattr(args, "muted", False)),
        music_enabled=not bool(getattr(args, "no_music", False)),
    )


def _team_color_from_args(args: argparse.Namespace) -> ColorRGBA:
    return cast(ColorRGBA, getattr(args, "team_color", DEFAULT_FORMULA_TEAM_COLOR))


def _add_race_rule_arguments(parser: argparse.ArgumentParser, *, include_scoring: bool = True) -> None:
    if include_scoring:
        parser.add_argument("--win-margin-m", type=float, default=1.0, help="distance margin required for a win")
        parser.add_argument(
            "--scoring",
            choices=("team-sum", "best-copy"),
            default="team-sum",
            help="distance metric used to classify each race",
        )
    else:
        parser.set_defaults(scoring="team-sum", win_margin_m=1.0)
    parser.add_argument("--no-marshal", action="store_true", help="disable stuck-car marshal recovery")
    parser.add_argument("--marshal-stuck-seconds", type=float, default=1.5)
    parser.add_argument("--marshal-penalty-m", type=float, default=5.0)
    parser.add_argument("--marshal-cooldown-seconds", type=float, default=2.0)


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the parser used by the ``racing`` terminal command."""
    parser = argparse.ArgumentParser(description="Run the Racing simulator.")
    parser.add_argument(
        "--student-module",
        type=str,
        default=None,
        help="Python file path or importable module name that defines a control(sensors) function",
    )
    parser.add_argument(
        "--control-function",
        type=str,
        default="control",
        help="student module function to call each tick",
    )
    parser.add_argument(
        "--fixed-delta-seconds",
        type=float,
        default=1 / 60,
        help="fixed physics tick for playable mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RACE_RANDOM_SEED,
        help="seed for the deterministic random starting position",
    )
    _add_track_arguments(parser)
    parser.add_argument(
        "--record-human",
        type=Path,
        default=None,
        metavar="PATH",
        help="append one JSON object per manual-control tick to a JSONL file",
    )
    parser.add_argument(
        "--camera",
        choices=tuple(view.value for view in CameraView if view is not CameraView.SPLIT_FOLLOW),
        default=CameraView.DRONE.value,
        help="initial playable camera view; press v in-game to cycle",
    )
    parser.add_argument("--window-type", choices=("offscreen",), default=None)
    parser.add_argument("--size", type=parse_window_size, default=(1280, 720))
    _add_color_argument(parser)
    _add_audio_arguments(parser)

    subparsers = parser.add_subparsers(dest="command")
    h2h_parser = subparsers.add_parser("h2h", help="race student controllers and/or keyboard control")
    _add_audio_arguments(h2h_parser)
    h2h_parser.add_argument(
        "--challenger-module",
        type=str,
        default=None,
        help="Python student controller module for challenger",
    )
    h2h_parser.add_argument(
        "--incumbent-module",
        type=str,
        default=None,
        help="Python student controller module for incumbent",
    )
    h2h_parser.add_argument("--challenger-name", default=None, help="override the challenger display name")
    h2h_parser.add_argument("--incumbent-name", default=None, help="override the incumbent display name")
    h2h_parser.add_argument(
        "--challenger-fallback-name", default=None, help="challenger label when its controller has no RACING_NAME"
    )
    h2h_parser.add_argument(
        "--incumbent-fallback-name", default=None, help="incumbent label when its controller has no RACING_NAME"
    )
    h2h_parser.add_argument(
        "--challenger-keyboard",
        action="store_true",
        help="drive the challenger with keyboard/controller input",
    )
    h2h_parser.add_argument(
        "--incumbent-keyboard",
        action="store_true",
        help="drive the incumbent with keyboard/controller input",
    )
    h2h_parser.add_argument(
        "--copies-per-side",
        type=int,
        default=None,
        help="cars per side; defaults to one per side",
    )
    h2h_parser.add_argument(
        "--challenger-copies",
        type=int,
        default=None,
        help="challenger cars; overrides --copies-per-side for the challenger side",
    )
    h2h_parser.add_argument(
        "--incumbent-copies",
        type=int,
        default=None,
        help="incumbent cars; overrides --copies-per-side for the incumbent side",
    )
    h2h_parser.add_argument("--races", type=int, default=1, help="number of head-to-head races to run")
    h2h_parser.add_argument(
        "--round-seconds",
        type=float,
        default=DEFAULT_RACE_SECONDS,
        help="simulated seconds per race",
    )
    h2h_parser.add_argument(
        "--seed",
        type=int,
        default=argparse.SUPPRESS,
        help="seed for deterministic random starting positions",
    )
    _add_track_arguments(h2h_parser, suppress_defaults=True)
    _add_race_rule_arguments(h2h_parser)
    h2h_parser.add_argument("--watch", action="store_true", help="open the graphical race viewer")
    h2h_parser.add_argument(
        "--json",
        action="store_true",
        help="print versioned machine-readable results instead of the terminal table (headless only)",
    )
    h2h_parser.add_argument("--window-type", choices=("offscreen",), default=None)
    h2h_parser.add_argument("--fullscreen", action="store_true", help="start the watched race in fullscreen")
    h2h_parser.add_argument("--size", type=parse_window_size, default=(1280, 720))
    h2h_parser.add_argument(
        "--camera",
        choices=tuple(view.value for view in CameraView),
        default=CameraView.DRONE.value,
        help="initial viewer camera view; press v in-game to cycle",
    )
    h2h_parser.add_argument(
        "--challenger-team-color",
        type=parse_color_rgba,
        default=DEFAULT_CHALLENGER_TEAM_COLOR,
        help="challenger formula car paint color",
    )
    h2h_parser.add_argument(
        "--incumbent-team-color",
        type=parse_color_rgba,
        default=DEFAULT_INCUMBENT_TEAM_COLOR,
        help="incumbent formula car paint color",
    )

    heat_parser = subparsers.add_parser("heat", help="race four or eight student controllers on one shared grid")
    heat_parser.add_argument(
        "--module",
        action="append",
        required=True,
        metavar="MODULE",
        help="controller file or module; repeat exactly four or eight times in entrant order",
    )
    heat_parser.add_argument(
        "--name",
        action="append",
        metavar="NAME",
        help="override controller names; repeat once per module in entrant order",
    )
    heat_parser.add_argument(
        "--fallback-name",
        action="append",
        metavar="NAME",
        help="labels for controllers without RACING_NAME; repeat once per module in entrant order",
    )
    heat_parser.add_argument("--control-function", default=argparse.SUPPRESS)
    heat_parser.add_argument("--fixed-delta-seconds", type=float, default=argparse.SUPPRESS)
    heat_parser.add_argument("--races", type=int, default=1, help="number of heat races to run")
    heat_parser.add_argument("--round-seconds", type=float, default=DEFAULT_RACE_SECONDS)
    heat_parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    _add_track_arguments(heat_parser, suppress_defaults=True)
    _add_race_rule_arguments(heat_parser, include_scoring=False)
    _add_audio_arguments(heat_parser)
    heat_parser.add_argument("--watch", action="store_true", help="open the graphical heat viewer")
    heat_parser.add_argument("--json", action="store_true", help="print machine-readable results (headless only)")
    heat_parser.add_argument("--window-type", choices=("offscreen",), default=None)
    heat_parser.add_argument("--fullscreen", action="store_true", help="start the watched heat in fullscreen")
    heat_parser.add_argument("--size", type=parse_window_size, default=(1280, 720))
    heat_parser.add_argument(
        "--camera",
        choices=tuple(view.value for view in CameraView if view is not CameraView.SPLIT_FOLLOW),
        default=CameraView.THREE_QUARTER.value,
        help="shared heat camera; split_follow is only supported by h2h",
    )
    return parser


def _head_to_head_rules_from_args(args: argparse.Namespace) -> HeadToHeadRaceRules:
    return HeadToHeadRaceRules(
        scoring=cast(HeadToHeadScoring, args.scoring),
        win_margin_m=float(args.win_margin_m),
        marshal_enabled=not bool(args.no_marshal),
        marshal_stuck_seconds=float(args.marshal_stuck_seconds),
        marshal_penalty_m=float(args.marshal_penalty_m),
        marshal_cooldown_seconds=float(args.marshal_cooldown_seconds),
    )


def _load_submission_from_args(
    *,
    parser: argparse.ArgumentParser,
    student_module: str,
    function_name: str,
    role: str,
) -> StudentControllerSubmission:
    try:
        return load_student_submission(student_module, function_name=function_name)
    except (AttributeError, FileNotFoundError, ImportError, TypeError, ValueError) as error:
        parser.error(f"{role}: {error}")
    raise AssertionError("parser.error should exit")


def _student_submission_name(
    submission: StudentControllerSubmission | None,
    role: str,
    override: str | None = None,
    *,
    fallback: str | None = None,
) -> str:
    return override or (submission.display_name if submission is not None else None) or fallback or role


def _student_submission_color(submission: StudentControllerSubmission | None, fallback: ColorRGBA) -> ColorRGBA:
    if submission is None or submission.car_color is None:
        return fallback
    return submission.car_color


def _role_source_count(*, student_module: str | None, keyboard: bool) -> int:
    return int(student_module is not None) + int(keyboard)


def _resolve_h2h_copy_counts(
    *,
    copies_per_side: int | None,
    challenger_copies: int | None,
    incumbent_copies: int | None,
    challenger_keyboard: bool,
    incumbent_keyboard: bool,
) -> tuple[int, int]:
    if copies_per_side is not None and copies_per_side < 1:
        raise ValueError("--copies-per-side must be at least 1")
    if challenger_copies is not None and challenger_copies < 1:
        raise ValueError("--challenger-copies must be at least 1")
    if incumbent_copies is not None and incumbent_copies < 1:
        raise ValueError("--incumbent-copies must be at least 1")

    default_copies = 1 if copies_per_side is None else copies_per_side
    resolved_challenger_copies = default_copies if challenger_copies is None else challenger_copies
    resolved_incumbent_copies = default_copies if incumbent_copies is None else incumbent_copies
    if challenger_keyboard and resolved_challenger_copies != 1:
        raise ValueError("keyboard-controlled challenger must use exactly 1 copy")
    if incumbent_keyboard and resolved_incumbent_copies != 1:
        raise ValueError("keyboard-controlled incumbent must use exactly 1 copy")
    return resolved_challenger_copies, resolved_incumbent_copies


def _run_heat_from_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    track_id: str,
    track_seed: int | None,
) -> HeatResult | None:
    modules = cast(list[str], args.module)
    if len(modules) not in HEAT_ENTRANT_COUNTS:
        parser.error("heat requires exactly four or eight --module arguments")
    names = cast(list[str] | None, args.name)
    fallback_names = cast(list[str] | None, args.fallback_name)
    for flag, values in (("--name", names), ("--fallback-name", fallback_names)):
        if values is not None and len(values) != len(modules):
            parser.error(f"{flag} must be repeated {len(modules)} times in entrant order")
    if bool(args.watch) and bool(args.json):
        parser.error("--json is only available for headless heats")
    rules = _head_to_head_rules_from_args(args)
    entrants: list[HeatEntrant] = []
    for index, module in enumerate(modules):
        role = f"entrant {index + 1}"
        submission = _load_submission_from_args(
            parser=parser,
            student_module=module,
            function_name=str(args.control_function),
            role=role,
        )
        entrants.append(
            HeatEntrant(
                name=_student_submission_name(
                    submission,
                    role,
                    None if names is None else names[index],
                    fallback=None if fallback_names is None else fallback_names[index],
                ),
                controller=submission.controller,
                team_color=_student_submission_color(submission, DEFAULT_HEAT_COLORS[index]),
            )
        )
    if bool(args.watch):
        create_heat_viewer_app(
            HeatViewerConfig(
                entrants=tuple(entrants),
                size=cast(tuple[int, int], args.size),
                fullscreen=bool(args.fullscreen),
                camera_view=CameraView(str(args.camera)),
                race_count=int(args.races),
                round_seconds=float(args.round_seconds),
                fixed_delta_seconds=float(args.fixed_delta_seconds),
                random_seed=int(args.seed),
                track_id=track_id,
                track_seed=track_seed,
                rules=rules,
                window_type=cast(str | None, args.window_type),
                audio=_audio_config_from_args(args),
            )
        ).run()
        return
    return run_headless_heat(
        entrants=tuple(entrants),
        race_count=int(args.races),
        round_seconds=float(args.round_seconds),
        fixed_delta_seconds=float(args.fixed_delta_seconds),
        random_seed=int(args.seed),
        track_id=track_id,
        track_seed=track_seed,
        rules=rules,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the command-line entry point."""
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    human_recording_path = cast(Path | None, args.record_human)
    track_id, track_seed = _track_selection_from_args(parser=parser, args=args)

    if getattr(args, "command", None) == "heat":
        if human_recording_path is not None:
            parser.error("--record-human is only available in single-car manual mode")
        with redirect_stdout(sys.stderr) if bool(args.json) else nullcontext():
            result = _run_heat_from_args(parser, args, track_id=track_id, track_seed=track_seed)
        if result is not None:
            if bool(args.json):
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
            else:
                print(format_heat_result(result))
        return

    if getattr(args, "command", None) == "h2h":
        if human_recording_path is not None:
            parser.error("--record-human is only available in single-car manual mode")
        challenger_module = cast(str | None, args.challenger_module)
        incumbent_module = cast(str | None, args.incumbent_module)
        challenger_keyboard = bool(args.challenger_keyboard)
        incumbent_keyboard = bool(args.incumbent_keyboard)
        rules = _head_to_head_rules_from_args(args)
        try:
            challenger_copies, incumbent_copies = _resolve_h2h_copy_counts(
                copies_per_side=cast(int | None, args.copies_per_side),
                challenger_copies=cast(int | None, args.challenger_copies),
                incumbent_copies=cast(int | None, args.incumbent_copies),
                challenger_keyboard=challenger_keyboard,
                incumbent_keyboard=incumbent_keyboard,
            )
        except ValueError as error:
            parser.error(str(error))
        if _role_source_count(student_module=challenger_module, keyboard=challenger_keyboard) != 1:
            parser.error("challenger requires exactly one of --challenger-module or --challenger-keyboard")
        if _role_source_count(student_module=incumbent_module, keyboard=incumbent_keyboard) != 1:
            parser.error("incumbent requires exactly one of --incumbent-module or --incumbent-keyboard")

        if (challenger_keyboard or incumbent_keyboard) and not bool(args.watch):
            parser.error("keyboard head-to-head requires --watch")
        if bool(args.watch) and bool(args.json):
            parser.error("--json is only available for headless head-to-head races")
        if bool(args.watch):
            challenger_submission = (
                None
                if challenger_keyboard
                else _load_submission_from_args(
                    parser=parser,
                    student_module=cast(str, challenger_module),
                    function_name=str(args.control_function),
                    role="challenger",
                )
            )
            incumbent_submission = (
                None
                if incumbent_keyboard
                else _load_submission_from_args(
                    parser=parser,
                    student_module=cast(str, incumbent_module),
                    function_name=str(args.control_function),
                    role="incumbent",
                )
            )
            create_head_to_head_viewer_app(
                HeadToHeadViewerConfig(
                    size=cast(tuple[int, int], args.size),
                    fullscreen=bool(args.fullscreen),
                    camera_view=CameraView(str(args.camera)),
                    challenger_name=_student_submission_name(
                        challenger_submission,
                        "keyboard" if challenger_keyboard else "challenger",
                        args.challenger_name,
                        fallback=args.challenger_fallback_name,
                    ),
                    incumbent_name=_student_submission_name(
                        incumbent_submission,
                        "keyboard" if incumbent_keyboard else "incumbent",
                        args.incumbent_name,
                        fallback=args.incumbent_fallback_name,
                    ),
                    challenger_controller=None if challenger_submission is None else challenger_submission.controller,
                    incumbent_controller=None if incumbent_submission is None else incumbent_submission.controller,
                    challenger_keyboard=challenger_keyboard,
                    incumbent_keyboard=incumbent_keyboard,
                    challenger_copies=challenger_copies,
                    incumbent_copies=incumbent_copies,
                    race_count=int(args.races),
                    round_seconds=float(args.round_seconds),
                    random_seed=int(args.seed),
                    track_id=track_id,
                    track_seed=track_seed,
                    win_margin_m=rules.win_margin_m,
                    rules=rules,
                    window_type=cast(str | None, args.window_type),
                    challenger_team_color=_student_submission_color(
                        challenger_submission, cast(ColorRGBA, args.challenger_team_color)
                    ),
                    incumbent_team_color=_student_submission_color(
                        incumbent_submission, cast(ColorRGBA, args.incumbent_team_color)
                    ),
                    audio=_audio_config_from_args(args),
                )
            ).run()
            return

        if challenger_module is None or incumbent_module is None:
            parser.error("headless h2h requires two student modules")
        challenger_submission = _load_submission_from_args(
            parser=parser,
            student_module=challenger_module,
            function_name=str(args.control_function),
            role="challenger",
        )
        incumbent_submission = _load_submission_from_args(
            parser=parser,
            student_module=incumbent_module,
            function_name=str(args.control_function),
            role="incumbent",
        )
        result = run_headless_head_to_head(
            challenger_controller=challenger_submission.controller,
            incumbent_controller=incumbent_submission.controller,
            challenger_name=_student_submission_name(
                challenger_submission, "challenger", args.challenger_name, fallback=args.challenger_fallback_name
            ),
            incumbent_name=_student_submission_name(
                incumbent_submission, "incumbent", args.incumbent_name, fallback=args.incumbent_fallback_name
            ),
            race_count=int(args.races),
            round_seconds=float(args.round_seconds),
            random_seed=int(args.seed),
            track_id=track_id,
            track_seed=track_seed,
            rules=rules,
            challenger_copies=challenger_copies,
            incumbent_copies=incumbent_copies,
        )
        if bool(args.json):
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True, allow_nan=False))
        else:
            print(format_head_to_head_result(result))
        return

    student_module = cast(str | None, args.student_module)
    if student_module is not None and human_recording_path is not None:
        parser.error("--record-human cannot be combined with --student-module")
    student_submission: StudentControllerSubmission | None = None
    if student_module is not None:
        student_submission = _load_submission_from_args(
            parser=parser,
            student_module=student_module,
            function_name=str(args.control_function),
            role="student",
        )

    playable_app = create_app(
        GameConfig(
            size=cast(tuple[int, int], args.size),
            camera_view=CameraView(str(args.camera)),
            student_controller=None if student_submission is None else student_submission.controller,
            fixed_delta_seconds=float(args.fixed_delta_seconds),
            random_seed=int(args.seed),
            track_id=track_id,
            track_seed=track_seed,
            window_type=cast(str | None, args.window_type),
            human_recording_path=human_recording_path,
            team_color=_student_submission_color(student_submission, _team_color_from_args(args)),
            audio=_audio_config_from_args(args),
        )
    )
    if human_recording_path is not None:
        print(f"Recording human gameplay to {human_recording_path.resolve()}")
    playable_app.run()
