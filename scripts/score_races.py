#!/usr/bin/env python3
"""Rank headless two-to-ten-car lap races by lead changes, overtakes, and close finishes."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import run_submissions
from run_submissions import DEFAULT_EXPORT_DIR, PROJECT_ROOT, resolve_controller, submission_id

from racing.game import cli
from racing.race.heat import DEFAULT_HEAT_COLORS, HEAT_ENTRANT_COUNTS, HeatEntrant, HeatRaceSnapshot, run_headless_heat
from racing.race.interest import InterestWeights, RaceInterestTracker, score_interest
from racing.race.progress import resolve_track
from racing.race.rules import HeadToHeadRaceRules
from racing.student.api import load_student_submission
from racing.track.world import TRACK_ID_MUGELLO_SHORT, track_layout_ids


@dataclass(frozen=True)
class RaceSpec:
    submissions: tuple[str, ...]
    names: tuple[str, ...] = ()
    label: str = "field"
    export_dir: str = str(DEFAULT_EXPORT_DIR)
    seed: int = 110
    track: str = TRACK_ID_MUGELLO_SHORT
    track_seed: int | None = None
    round_laps: int = 5
    finish_timeout_seconds: float = 10.0
    rules: HeadToHeadRaceRules = field(default_factory=HeadToHeadRaceRules)
    fixed_delta_seconds: float = 1 / 60
    control_function: str = "control"

    def validate(self) -> None:
        if len(self.submissions) not in HEAT_ENTRANT_COUNTS or len(set(self.submissions)) != len(self.submissions):
            raise ValueError("provide two to ten distinct submission IDs")
        if self.names and (len(self.names) != len(self.submissions) or any(not name.strip() for name in self.names)):
            raise ValueError("repeat --name once per car, with nonempty names")
        if self.round_laps < 1:
            raise ValueError("laps must be positive")
        if not math.isfinite(self.finish_timeout_seconds) or self.finish_timeout_seconds < 0:
            raise ValueError("finish timeout must be finite and nonnegative")
        if not math.isfinite(self.fixed_delta_seconds) or self.fixed_delta_seconds <= 0:
            raise ValueError("fixed delta must be finite and positive")
        resolve_track(self.track, self.track_seed)
        for identifier in self.submissions:
            resolve_controller(Path(self.export_dir), identifier)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RaceSpec:
        values: dict[str, Any] = dict(data)
        values.update(
            submissions=tuple(data["submissions"]),
            names=tuple(data["names"]),
            rules=HeadToHeadRaceRules(**data["rules"]),
        )
        return cls(**values)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def nonnegative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return number


def positive_float(value: str) -> float:
    number = nonnegative_float(value)
    if number == 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("submissions", nargs="*", type=submission_id, metavar="ID")
    parser.add_argument("--name", action="append", help="repeat once per car in grid order")
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--commands", metavar="PATH|-", help="read run_submissions.py commands; - reads stdin")
    parser.add_argument("--track", choices=(*track_layout_ids(), "procedural"))
    parser.add_argument(
        "--track-seed",
        "--start-track-seed",
        type=int,
        help="first procedural layout seed, distinct from the starting-grid seed",
    )
    parser.add_argument(
        "--track-count",
        type=positive_int,
        default=1,
        help="consecutive procedural track seeds per field (default: 1; requires a track seed)",
    )
    parser.add_argument("--seed", "--start-seed", type=int, help="first starting seed (default: 110 or command's seed)")
    parser.add_argument(
        "--count", type=positive_int, default=1, help="consecutive starting seeds per track and field (default: 1)"
    )
    parser.add_argument(
        "--round-laps", "--laps", type=positive_int, help="laps per race (default: 5 or command's laps)"
    )
    parser.add_argument("--damage", action=argparse.BooleanOptionalAction, default=None, help="damage on by default")
    parser.add_argument("--finish-timeout-seconds", type=nonnegative_float, help="wait after P1 (default: 10 seconds)")
    parser.add_argument("--timeout-seconds", type=positive_float, default=300, help="wall-clock limit per race")
    parser.add_argument(
        "--weights", nargs=3, type=nonnegative_float, default=(30, 30, 40), metavar=("LEAD", "PASS", "FINISH")
    )
    parser.add_argument("--finish-gap-scale-seconds", type=positive_float, default=2.0)
    parser.add_argument(
        "--top", type=positive_int, default=3, help="number of highest-scoring replay commands to print"
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "artifacts" / "race-interest.jsonl", help="append-only JSONL log"
    )
    parser.add_argument(
        "--explain-score", action="store_true", help="show the formula and synthetic sanity checks, then exit"
    )
    return parser


def specs_from_commands(source: str) -> list[RaceSpec]:
    """Parse command text as data. Never execute shell syntax or command prefixes."""
    specs: list[RaceSpec] = []
    label = "field"
    for line in source.replace("\\\n", " ").splitlines():
        if line.startswith("# ") and " — " in line:
            label = line[2:].split(" — ", 1)[0]
        tokens = shlex.split(line, comments=True)
        if not tokens:
            continue
        script = next((index for index, token in enumerate(tokens) if Path(token).name == "run_submissions.py"), None)
        if script is None:
            raise ValueError("expected a run_submissions.py command on each non-comment line")
        runner_args, extra = run_submissions.build_parser().parse_known_args(tokens[script + 1 :])
        if extra[:1] == ["--"]:
            extra = extra[1:]
        heat_args = ["heat", "--seed", str(runner_args.seed)]
        for identifier in runner_args.submissions:
            heat_args.extend(["--module", f"unused-{identifier}"])
        race = cli.build_argument_parser().parse_args([*heat_args, *extra])
        if race.races != 1:
            raise ValueError("command --races must be 1; use score_races.py --count for independent starting seeds")
        if any(token.split("=", 1)[0] == "--round-seconds" for token in extra):
            raise ValueError("interest scoring requires lap races; replace --round-seconds with --round-laps")
        specs.append(
            RaceSpec(
                submissions=tuple(runner_args.submissions),
                names=tuple(race.name or race.fallback_name or ()),
                label=label,
                export_dir=str(runner_args.export_dir.expanduser().resolve()),
                seed=race.seed,
                track=race.track or ("procedural" if race.track_seed is not None else TRACK_ID_MUGELLO_SHORT),
                track_seed=race.track_seed,
                round_laps=race.round_laps or 5,
                finish_timeout_seconds=race.finish_timeout_seconds
                if any(token.split("=", 1)[0] == "--finish-timeout-seconds" for token in extra)
                else 10.0,
                rules=HeadToHeadRaceRules(
                    damage_enabled=not race.no_damage,
                    marshal_enabled=not race.no_marshal,
                    marshal_stuck_seconds=race.marshal_stuck_seconds,
                    marshal_penalty_m=race.marshal_penalty_m,
                    marshal_cooldown_seconds=race.marshal_cooldown_seconds,
                ),
                fixed_delta_seconds=race.fixed_delta_seconds,
                control_function=race.control_function,
            )
        )
    if not specs:
        raise ValueError("no race commands found")
    return specs


def apply_overrides(spec: RaceSpec, args: argparse.Namespace) -> RaceSpec:
    updates: dict[str, Any] = {}
    for key in ("seed", "round_laps", "finish_timeout_seconds"):
        if (value := getattr(args, key)) is not None:
            updates[key] = value
    if args.export_dir is not None:
        updates["export_dir"] = str(args.export_dir.expanduser().resolve())
    if args.name is not None:
        updates["names"] = tuple(args.name)
    if args.track is not None or args.track_seed is not None:
        updates.update(track=args.track or "procedural", track_seed=args.track_seed)
    if args.damage is not None:
        updates["rules"] = replace(spec.rules, damage_enabled=args.damage)
    return replace(spec, **updates)


def trial_specs(spec: RaceSpec, *, seed_count: int, track_count: int) -> Iterator[RaceSpec]:
    """Try each track with the same starting-seed range, preserving field order."""
    for track_offset in range(track_count):
        track_seed = None if spec.track_seed is None else spec.track_seed + track_offset
        for seed in range(spec.seed, spec.seed + seed_count):
            yield replace(spec, seed=seed, track_seed=track_seed)


def replay_command(
    spec: RaceSpec,
    *,
    camera: str = "cinematic",
    fullscreen: bool = True,
    audio: bool = True,
    music: bool = True,
    muted: bool = False,
    title: str | None = None,
) -> str:
    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT_ROOT),
        "python",
        str(PROJECT_ROOT / "scripts" / "run_submissions.py"),
    ]
    command.extend(spec.submissions)
    if len(spec.submissions) == 2:
        command.append("--heat")
    for name in spec.names:
        command.extend(["--name", name])
    command.extend(
        [
            "--export-dir",
            spec.export_dir,
            "--track",
            spec.track,
            "--seed",
            str(spec.seed),
            "--round-laps",
            str(spec.round_laps),
            "--finish-timeout-seconds",
            str(spec.finish_timeout_seconds),
            "--fixed-delta-seconds",
            str(spec.fixed_delta_seconds),
            "--control-function",
            spec.control_function,
            "--marshal-stuck-seconds",
            str(spec.rules.marshal_stuck_seconds),
            "--marshal-penalty-m",
            str(spec.rules.marshal_penalty_m),
            "--marshal-cooldown-seconds",
            str(spec.rules.marshal_cooldown_seconds),
        ]
    )
    if spec.track_seed is not None:
        command.extend(["--track-seed", str(spec.track_seed)])
    if not spec.rules.damage_enabled:
        command.append("--no-damage")
    if not spec.rules.marshal_enabled:
        command.append("--no-marshal")
    command.extend(["--camera", camera])
    if fullscreen:
        command.append("--fullscreen")
    if not audio:
        command.append("--no-audio")
    if not music:
        command.append("--no-music")
    if muted:
        command.append("--muted")
    if title:
        command.extend(["--title", title])
    return shlex.join(command)


def run_race(spec: RaceSpec, weights: InterestWeights) -> dict[str, Any]:
    entrants: list[HeatEntrant] = []
    for index, identifier in enumerate(spec.submissions):
        submission = load_student_submission(
            resolve_controller(Path(spec.export_dir), identifier),
            function_name=spec.control_function,
            suppress_prints=True,
        )
        entrants.append(
            HeatEntrant(
                name=spec.names[index] if spec.names else submission.display_name or f"#{identifier}",
                controller=submission.controller,
                team_color=submission.car_color or DEFAULT_HEAT_COLORS[index],
            )
        )
    tracker = RaceInterestTracker()
    duration_seconds = 0.0

    def observe(snapshot: HeatRaceSnapshot) -> None:
        nonlocal duration_seconds
        duration_seconds = snapshot.elapsed_seconds
        tracker.update(snapshot)

    result = run_headless_heat(
        entrants=tuple(entrants),
        random_seed=spec.seed,
        round_laps=spec.round_laps,
        finish_timeout_seconds=spec.finish_timeout_seconds,
        track_id=spec.track,
        track_seed=spec.track_seed,
        rules=spec.rules,
        fixed_delta_seconds=spec.fixed_delta_seconds,
        observer=observe,
    )
    score = tracker.score(result.races[0].standings, round_laps=spec.round_laps, weights=weights)
    return {
        "duration_seconds": duration_seconds,
        "interest": score.to_dict(),
        "result": result.to_dict(),
        "events": [asdict(event) for event in tracker.events],
        "event_car_names": {f"heat-{index}:0": entrant.name for index, entrant in enumerate(entrants)},
        "event_filters": {
            "pass_margin_m": tracker.pass_margin_m,
            "confirmation_seconds": tracker.confirmation_seconds,
            "recovery_seconds": tracker.recovery_seconds,
        },
    }


def run_worker(spec: RaceSpec, weights: InterestWeights, timeout: float) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--worker"],
        input=json.dumps({"spec": asdict(spec), "weights": asdict(weights)}),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"race exited {completed.returncode}: {completed.stderr[-3000:]}")
    try:
        payload = json.loads(completed.stdout)
        if not isinstance(payload, dict) or "interest" not in payload or "result" not in payload:
            raise ValueError("missing result or interest score")
        return cast(dict[str, Any], payload)
    except ValueError as error:
        raise RuntimeError(f"invalid race output: {completed.stdout[-2000:]}") from error


def explain_score(weights: InterestWeights) -> None:
    total = weights.lead + weights.overtakes + weights.finish
    print(
        f"Weights: lead {100 * weights.lead / total:g}%, overtakes {100 * weights.overtakes / total:g}%, "
        f"finish {100 * weights.finish / total:g}%."
    )
    print("L = lead changes; O = overtakes; N = cars; laps = race length.")
    print("Lead component = L / (L + laps); overtake component = O / (O + laps*(N-1)).")
    print(
        f"Finish component = exp(-mean(P2-P1, P3-P1) / {weights.finish_gap_scale_seconds:g}s); "
        "zero if fewer than 3 finishers."
    )
    print("Score = 100 * weighted mean of components. Lead passes intentionally earn both event terms.")
    print("Synthetic checks (8 cars, 5 laps; hypothetical, not simulated):")
    for label, leads, passes, times in (
        ("Procession, spread finish", 0, 0, (100.0, 105.0, 110.0)),
        ("Procession, close finish", 0, 0, (100.0, 100.2, 100.5)),
        ("Battles, spread finish", 5, 35, (100.0, 105.0, 110.0)),
        ("Battles, close finish", 5, 35, (100.0, 100.2, 100.5)),
        ("Same battles, only 2 finishers", 5, 35, (100.0, 100.2)),
    ):
        score = score_interest(
            lead_changes=leads, overtakes=passes, round_laps=5, entrant_count=8, finish_times=times, weights=weights
        )
        print(f"  {score.score:6.2f}  {label}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        weights = InterestWeights(*args.weights, finish_gap_scale_seconds=args.finish_gap_scale_seconds)
        if args.explain_score:
            explain_score(weights)
            return 0
        if args.commands is not None:
            if args.submissions:
                raise ValueError("choose submission IDs or --commands, not both")
            source = sys.stdin.read() if args.commands == "-" else Path(args.commands).expanduser().read_text()
            specs = specs_from_commands(source)
        else:
            specs = [RaceSpec(submissions=tuple(args.submissions))]
        specs = [apply_overrides(spec, args) for spec in specs]
        for spec in specs:
            if args.track_count > 1 and (spec.track != "procedural" or spec.track_seed is None):
                raise ValueError(
                    "--track-count greater than 1 requires --track-seed (or a procedural seed in each command)"
                )
            spec.validate()
    except (ValueError, OSError) as error:
        parser.error(str(error))

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed: list[dict[str, Any]] = []
    failures = 0
    run_id = datetime.now(UTC).isoformat()
    print(
        f"Scoring {len(specs)} field(s) x {args.track_count} track(s) x {args.count} starting seed(s) "
        f"= {len(specs) * args.track_count * args.count} races. Results: {output_path}",
        flush=True,
    )
    with output_path.open("a", encoding="utf-8") as output:
        try:
            for original in specs:
                for spec in trial_specs(original, seed_count=args.count, track_count=args.track_count):
                    print(
                        f"{spec.label}: seed {spec.seed}, {spec.track}"
                        f"{'' if spec.track_seed is None else f' (track seed {spec.track_seed})'}, "
                        f"{spec.round_laps} laps, damage {'on' if spec.rules.damage_enabled else 'off'} ...",
                        flush=True,
                    )
                    record: dict[str, Any] = {
                        "schema_version": 1,
                        "run_id": run_id,
                        "spec": asdict(spec),
                        "weights": asdict(weights),
                        "replay_command": replay_command(spec),
                    }
                    started = time.monotonic()
                    try:
                        record.update(run_worker(spec, weights, args.timeout_seconds), status="completed")
                        completed.append(record)
                        score = record["interest"]
                        gap = score["gap_p3_seconds"]
                        gap_text = "no P3 finisher" if gap is None else f"P3 +{gap:.3f}s"
                        print(
                            f"  {score['score']:.2f}/100 | {score['lead_changes']} lead changes | "
                            f"{score['overtakes']} overtakes | {gap_text} | "
                            f"{score['finishers']}/{len(spec.submissions)} finished",
                            flush=True,
                        )
                    except subprocess.TimeoutExpired:
                        failures += 1
                        record.update(status="timeout", timeout_seconds=args.timeout_seconds)
                        print("  Timed out; no score assigned.", flush=True)
                    except RuntimeError as error:
                        failures += 1
                        record.update(status="error", error=str(error))
                        print(f"  {error}", file=sys.stderr, flush=True)
                    record["wall_seconds"] = time.monotonic() - started
                    output.write(json.dumps(record, allow_nan=False) + "\n")
                    output.flush()
        except KeyboardInterrupt:
            print("\nStopped. Completed races are saved in the JSONL log.", flush=True)
            return 130

    ranked = sorted(
        completed,
        key=lambda record: (-record["interest"]["score"], record["spec"]["seed"], record["spec"]["track_seed"] or 0),
    )
    print(f"\nCompleted {len(completed)}; errors/timeouts {failures}. Best races:", flush=True)
    for rank, record in enumerate(ranked[: args.top], start=1):
        score = record["interest"]
        spec = RaceSpec.from_dict(record["spec"])
        track_label = spec.track if spec.track_seed is None else f"{spec.track} (track seed {spec.track_seed})"
        print(
            f"{rank}. {spec.label}, seed {spec.seed}, {track_label}: {score['score']:.2f}/100 "
            f"(lead {score['lead_points']:.2f} + pass {score['overtake_points']:.2f} "
            f"+ finish {score['finish_points']:.2f})"
        )
        print(record["replay_command"], flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--worker"]:
        request = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            response = run_race(RaceSpec.from_dict(request["spec"]), InterestWeights(**request["weights"]))
        print(json.dumps(response, allow_nan=False))
    else:
        raise SystemExit(main())
