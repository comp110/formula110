#!/usr/bin/env python3
"""Select exclusive leaderboard groups, find interesting races, and build a replay show plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shlex
import subprocess
import sys
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from leaderboard_heats import display_name, read_leaderboards, select_boards
from run_submissions import DEFAULT_EXPORT_DIR, PROJECT_ROOT, resolve_controller
from score_races import (
    RaceSpec,
    nonnegative_float,
    positive_float,
    positive_int,
    replay_command,
    run_worker,
    trial_specs,
)

from racing.game.config import CameraView
from racing.race.heat import DEFAULT_HEAT_COLORS, HEAT_ENTRANT_COUNTS
from racing.race.interest import InterestWeights
from racing.race.rules import HeadToHeadRaceRules
from racing.student.api import load_student_submission
from racing.track.world import TRACK_ID_MUGELLO_SHORT


@dataclass(frozen=True)
class Category:
    id: str
    query: str
    title: str
    order: str
    laps: int
    track: str = "procedural"


# Show order stays low to high category; allocation follows the selected policy.
CATEGORIES = (
    Category("sips-tea", "Sips Tea", "Sips Tea", "asc", 6),
    Category("gs-going-crazy", "Gs Going Crazy", "G's Going Crazy", "desc", 6),
    Category("hits-different", "Hits Different", "Hits Different", "desc", 6),
    Category("gas-locked-in", "Gas Locked In", "Gas Locked In", "asc", 8),
    Category("clock-it", "Clock It", "Clock It", "asc", 10, TRACK_ID_MUGELLO_SHORT),
)


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def read_review_scores(path: Path, export_dir: Path, *, column: str = "total_score") -> dict[str, float | None]:
    """Read a numeric review CSV column without treating blank values as zero."""
    scores: dict[str, float | None] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {"submission_id", "total_score", column}
        if not required.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(f"{path}: expected unique {', '.join(sorted(required))} CSV columns")
        for row in reader:
            identifier = (row.get("submission_id") or "").strip()
            if not identifier.isascii() or not identifier.isdecimal():
                raise ValueError(f"{path}:{reader.line_num}: invalid submission_id {identifier!r}")
            if identifier in scores:
                raise ValueError(f"{path}:{reader.line_num}: duplicate submission_id {identifier}")
            if not (export_dir / f"submission_{identifier}").is_dir():
                raise ValueError(f"{path}:{reader.line_num}: submission {identifier} is not in this export")
            raw = (row.get(column) or "").strip()
            if not raw:
                scores[identifier] = None
                continue
            try:
                score = float(raw)
            except ValueError as error:
                raise ValueError(f"{path}:{reader.line_num}: invalid {column} {raw!r}") from error
            if not math.isfinite(score) or score < 0:
                raise ValueError(f"{path}:{reader.line_num}: {column} must be finite and nonnegative")
            scores[identifier] = score
    return scores


def filter_juiced_scores(
    review_scores: dict[str, float | None],
    threshold: float,
    *,
    lap_scores: dict[str, float | None] | None = None,
    min_laps: float | None = None,
) -> dict[str, float]:
    """Both configured inclusive minimums must pass; unknown laps never pass a lap filter."""
    held_out: dict[str, float] = {}
    for identifier, score in review_scores.items():
        if score is None or score < threshold:
            continue
        if min_laps is not None:
            laps = (lap_scores or {}).get(identifier)
            if laps is None or laps < min_laps:
                continue
        held_out[identifier] = score
    return held_out


def select_groups(
    export_dir: Path,
    selection_priority: str = "highest",
    *,
    review_scores: dict[str, float | None] | None = None,
    juiced_threshold: float = 1.0,
    review_laps: dict[str, float | None] | None = None,
    juiced_min_laps: float | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Allocate ten unclaimed cars per board in the requested category order."""
    if selection_priority not in ("highest", "lowest"):
        raise ValueError("selection priority must be highest or lowest")
    boards, warnings = read_leaderboards(export_dir)
    category_boards = {category.id: select_boards(boards, [category.query])[0] for category in CATEGORIES}
    held_out: set[str] = set()
    if review_scores is not None:
        reviewed_boards = [*category_boards.values(), *select_boards(boards, ["All Spawns, No Crumbs"])]
        eligible = {leader.submission_id for board in reviewed_boards for leader in board.leaders}
        unreviewed = sorted(identifier for identifier in eligible if review_scores.get(identifier) is None)
        if unreviewed:
            raise ValueError(
                "review CSV has missing/blank total_score for eligible submissions: "
                + ", ".join(unreviewed)
                + "; regenerate or complete the review CSV for this export"
            )
        held_out = set(
            filter_juiced_scores(review_scores, juiced_threshold, lap_scores=review_laps, min_laps=juiced_min_laps)
        )
    claimed: dict[str, str] = {}
    groups: dict[str, dict[str, Any]] = {}
    allocation = list(enumerate(CATEGORIES, start=1))
    if selection_priority == "highest":
        allocation.reverse()
    for allocation_rank, (priority, category) in enumerate(allocation, start=1):
        board = category_boards[category.id]
        if board.order != category.order:
            raise ValueError(f"{board.name}: expected {category.order} ranking, got {board.order}")
        competitors: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        skipped_holdout: list[dict[str, Any]] = []
        for original_rank, leader in enumerate(board.ranked(), start=1):
            identifier = leader.submission_id
            if identifier in held_out:
                skipped_holdout.append({"submission_id": identifier, "leaderboard_rank": original_rank})
                continue
            if identifier in claimed:
                skipped.append(
                    {
                        "submission_id": identifier,
                        "leaderboard_rank": original_rank,
                        "assigned_category": claimed[identifier],
                    }
                )
                continue
            competitors.append(
                {
                    "submission_id": identifier,
                    "display_name": leader.name,
                    "category_id": category.id,
                    "group_rank": len(competitors) + 1,
                    "leaderboard_rank": original_rank,
                    "leaderboard_score": leader.value,
                    "controller_path": str(resolve_controller(export_dir, identifier)),
                }
            )
            claimed[identifier] = category.id
            if len(competitors) == 10:
                break
        if len(competitors) != 10:
            raise ValueError(
                f"{board.name}: only {len(competitors)} unclaimed eligible cars; "
                "need 10 after earlier category allocations and juiced holdouts"
            )
        groups[category.id] = {
            "id": category.id,
            "title": category.title,
            "priority": priority,
            "allocation_rank": allocation_rank,
            "leaderboard": {"name": board.name, "order": board.order, "eligible_count": len(board.leaders)},
            "round_laps": category.laps,
            "track": category.track,
            "competitors": competitors,
            "skipped_already_assigned": skipped,
            "skipped_juiced": skipped_holdout,
            # Retain the original field for consumers of highest-first plans.
            "skipped_higher_priority": skipped if selection_priority == "highest" else [],
            "top_races": [],
            "status": "planned",
        }
    return [groups[category.id] for category in CATEGORIES], warnings


def build_juiced_groups(
    export_dir: Path,
    held_out_scores: dict[str, float],
    *,
    track: str,
    laps: int,
    lap_scores: dict[str, float | None] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Include every held-out car, splitting large fields into balanced heats."""
    if not held_out_scores:
        return [], []
    if len(held_out_scores) == 1:
        return [], ["Only one juiced entrant: held out of all categories, but a separate heat requires at least two."]
    payload = yaml.safe_load((export_dir / "submission_metadata.yml").read_text())
    competitors: list[dict[str, Any]] = []
    for identifier in sorted(held_out_scores, key=int):
        record = payload.get(f"submission_{identifier}")
        if not isinstance(record, dict):
            raise ValueError(f"submission {identifier}: missing metadata for juiced entrant")
        competitors.append(
            {
                "submission_id": identifier,
                "display_name": display_name(cast(dict[str, object], record), identifier),
                "category_id": "juiced",
                "leaderboard_rank": None,
                "leaderboard_score": None,
                "review_total_score": held_out_scores[identifier],
                **({"all_spawns_no_crumbs_laps": lap_scores.get(identifier)} if lap_scores is not None else {}),
                "controller_path": str(resolve_controller(export_dir, identifier)),
            }
        )
    heat_count = (len(competitors) + 9) // 10
    groups: list[dict[str, Any]] = []
    for index in range(heat_count):
        field = [{**c, "group_rank": rank} for rank, c in enumerate(competitors[index::heat_count], start=1)]
        groups.append(
            {
                "id": "juiced" if heat_count == 1 else f"juiced-{index + 1}",
                "title": "Juiced" if heat_count == 1 else f"Juiced {index + 1}",
                "round_laps": laps,
                "track": track,
                "competitors": field,
                "top_races": [],
                "status": "planned",
            }
        )
    return groups, []


def inspect_controller(path: str, timeout: float) -> dict[str, Any]:
    """Read actual controller metadata in a bounded, isolated interpreter."""
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--inspect-controller"],
        input=json.dumps({"path": path}),
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode:
        raise ValueError(f"could not load {path}: {completed.stderr[-2000:]}")
    return json.loads(completed.stdout)


def color_metadata(competitor: dict[str, Any], index: int) -> dict[str, Any]:
    """Use the same explicit-color/grid-fallback rule as both scorer and viewer."""
    declared = competitor.get("declared_color_rgba")
    color = list(declared if declared is not None else DEFAULT_HEAT_COLORS[index])
    return {
        **competitor,
        "grid_position": index + 1,
        "runtime_car_id": f"heat-{index}:0",
        "team_color_rgba": color,
        "team_color_hex": "#" + "".join(f"{round(channel * 255):02X}" for channel in color[:3]),
        "color_source": "RACING_COLOR" if declared is not None else "grid_palette",
    }


def enrich_competitors(groups: list[dict[str, Any]], export_dir: Path, jobs: int, timeout: float) -> None:
    payload = yaml.safe_load((export_dir / "submission_metadata.yml").read_text())
    competitors = [competitor for group in groups for competitor in group["competitors"]]
    print(f"Inspecting {len(competitors)} selected controllers and their team colors ...", flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(inspect_controller, c["controller_path"], timeout) for c in competitors]
        for competitor, future in zip(competitors, futures, strict=True):
            competitor.update(future.result())
            record = payload[f"submission_{competitor['submission_id']}"]
            submitters = record.get(":submitters", record.get("submitters", []))
            competitor["submitter_names"] = [s.get(":name", s.get("name", "")) for s in submitters]
            competitor["submitted_at"] = str(record.get(":created_at", record.get("created_at", "")))
    for group in groups:
        group["competitors"] = [color_metadata(c, i) for i, c in enumerate(group["competitors"])]


def source_fingerprint(export_dir: Path, groups: list[dict[str, Any]]) -> dict[str, Any]:
    sources: dict[str, str] = {}
    for group in groups:
        for competitor in group["competitors"]:
            root = export_dir / f"submission_{competitor['submission_id']}"
            files = {
                str(path.relative_to(root)): file_digest(path)
                for path in sorted(root.rglob("*"))
                if path.is_file()
                and not any(part in ("__pycache__", ".git", ".venv") for part in path.parts)
                and path.name != ".DS_Store"
            }
            source_hash = digest(files)
            sources[competitor["submission_id"]] = source_hash
            competitor["source_sha256"] = source_hash
    engine_files = [
        *sorted((PROJECT_ROOT / "src" / "racing").rglob("*.py")),
        PROJECT_ROOT / "scripts" / "score_races.py",
        PROJECT_ROOT / "scripts" / "run_submissions.py",
        Path(__file__).resolve(),
    ]
    return {
        "export_dir": str(export_dir),
        "metadata_sha256": file_digest(export_dir / "submission_metadata.yml"),
        "submission_sha256": sources,
        "engine_sha256": digest({str(path.relative_to(PROJECT_ROOT)): file_digest(path) for path in engine_files}),
    }


def stage_specs(stage: dict[str, Any], settings: dict[str, Any]) -> list[RaceSpec]:
    procedural = stage["track"] == "procedural"
    base = RaceSpec(
        submissions=tuple(c["submission_id"] for c in stage["competitors"]),
        names=tuple(c["display_name"] for c in stage["competitors"]),
        label=stage["title"],
        export_dir=settings["export_dir"],
        seed=settings["start_seed"],
        track=stage["track"],
        track_seed=settings["start_track_seed"] if procedural else None,
        round_laps=stage["round_laps"],
        finish_timeout_seconds=settings["finish_timeout_seconds"],
        rules=HeadToHeadRaceRules(**settings["rules"]),
    )
    fixed_count = settings["bahrain_seed_count"] if stage["track"] == "bahrain" else settings["fixed_seed_count"]
    specs = list(
        trial_specs(
            base,
            seed_count=settings["starts_per_track"] if procedural else fixed_count,
            track_count=settings["track_count"] if procedural else 1,
        )
    )
    return [
        replace(
            spec, round_laps=stage.get("lap_calibration", {}).get(str(spec.track_seed), {}).get("laps", spec.round_laps)
        )
        for spec in specs
    ]


def trial_id(stage_id: str, spec: RaceSpec) -> str:
    return f"{stage_id}-{digest(asdict(spec))[:16]}"


def launch_details(spec: RaceSpec, playback: dict[str, Any]) -> dict[str, Any]:
    command = replay_command(spec, **playback, title=spec.label)
    return {"cwd": str(PROJECT_ROOT), "argv": shlex.split(command), "shell_command": command}


def classification(stage: dict[str, Any], record: dict[str, Any]) -> list[dict[str, Any]]:
    rows = record["result"]["races"][0]["standings"]
    if sorted(row["entrant_index"] for row in rows) != list(range(len(stage["competitors"]))):
        raise ValueError("race result does not classify each competitor exactly once")
    return [
        {
            **row,
            **{
                key: stage["competitors"][row["entrant_index"]][key]
                for key in (
                    "submission_id",
                    "display_name",
                    "group_rank",
                    "leaderboard_rank",
                    "leaderboard_score",
                    "team_color_hex",
                    "team_color_rgba",
                    "category_id",
                )
            },
        }
        for row in rows
    ]


def rank_stage(stage: dict[str, Any], records: dict[str, dict[str, Any]], settings: dict[str, Any]) -> None:
    specs = stage_specs(stage, settings)
    stage_records = [records[trial_id(stage["id"], spec)] for spec in specs if trial_id(stage["id"], spec) in records]
    completed = [record for record in stage_records if record["status"] == "completed"]
    completed.sort(key=lambda r: (-r["quality"]["score"], r["spec"]["track_seed"] or 0, r["spec"]["seed"]))
    stage["search"] = {
        "planned": len(specs),
        "completed": len(completed),
        "failed": len(stage_records) - len(completed),
        "pending": len(specs) - len(stage_records),
        "track_count": settings["track_count"] if stage["track"] == "procedural" else 1,
        "starting_seeds_per_track": len(specs) // (settings["track_count"] if stage["track"] == "procedural" else 1),
    }
    stage["status"] = (
        "searching"
        if stage["search"]["pending"]
        else ("complete" if not stage["search"]["failed"] else "complete_with_errors")
    )
    top_races: list[dict[str, Any]] = []
    stage["top_races"] = top_races
    for rank, record in enumerate(completed[:3], start=1):
        rows = classification(stage, record)
        podium = sorted(
            (row for row in rows if not row["dnf"] and row["finish_position"] in (1, 2, 3)),
            key=lambda row: row["finish_position"],
        )
        top_races.append(
            {
                "interest_rank": rank,
                "trial_id": record["trial_id"],
                "spec": record["spec"],
                "interest": record["interest"],
                "quality": record["quality"],
                "classification": rows,
                "podium": podium,
                "launch": launch_details(RaceSpec.from_dict(record["spec"]), settings["playback"]),
            }
        )


def build_bahrain_groups(export_dir: Path, held_out: set[str], car_count: int = 10) -> list[dict[str, Any]]:
    """Select independent All Spawns fields, without category-allocation exclusions."""
    if car_count not in HEAT_ENTRANT_COUNTS:
        raise ValueError("Bahrain fields require two to twenty cars")
    boards, _ = read_leaderboards(export_dir)
    board = select_boards(boards, ["All Spawns, No Crumbs"])[0]
    if board.order != "desc":
        raise ValueError("All Spawns, No Crumbs must rank most laps first (descending)")
    groups = []
    for identifier, title, exclude_juiced in (
        ("bahrain-no-juiced", f"Bahrain - All Spawns, No Crumbs Top {car_count} (No Juiced)", True),
        ("bahrain-open", f"Bahrain - All Spawns, No Crumbs Top {car_count} (Open, Including Juiced)", False),
    ):
        competitors = []
        for rank, leader in enumerate(board.ranked(), start=1):
            if exclude_juiced and leader.submission_id in held_out:
                continue
            competitors.append(
                {
                    "submission_id": leader.submission_id,
                    "display_name": leader.name,
                    "category_id": identifier,
                    "group_rank": len(competitors) + 1,
                    "leaderboard_rank": rank,
                    "leaderboard_score": leader.value,
                    "is_juiced": leader.submission_id in held_out,
                    "controller_path": str(resolve_controller(export_dir, leader.submission_id)),
                }
            )
            if len(competitors) == car_count:
                break
        if len(competitors) != car_count:
            raise ValueError(
                f"{title}: need {car_count} eligible All Spawns, No Crumbs entrants; found {len(competitors)}"
            )
        groups.append(
            {
                "id": identifier,
                "title": title,
                "round_laps": 3,
                "track": "bahrain",
                "qualification": "Highest All Spawns, No Crumbs laps, independent of regular category allocation",
                "leaderboard": {"name": board.name, "order": board.order, "eligible_count": len(board.leaders)},
                "exclude_juiced": exclude_juiced,
                "competitors": competitors,
                "top_races": [],
                "status": "planned",
            }
        )
    return groups


def plan_stages(plan: dict[str, Any]) -> list[dict[str, Any]]:
    return [*plan["groups"], *plan.get("juiced_groups", []), *plan.get("bahrain_groups", [])]


def save_plan(path: Path, plan: dict[str, Any]) -> None:
    plan["updated_at"] = utc_now()
    stages = plan_stages(plan)
    plan["show_order"] = [stage["id"] for stage in stages]
    show_runner: list[dict[str, Any]] = []
    plan["show_runner"] = show_runner
    for stage in stages:
        if stage["top_races"]:
            best = stage["top_races"][0]
            show_runner.append(
                {
                    "stage_id": stage["id"],
                    "title": stage["title"],
                    "selected_trial_id": best["trial_id"],
                    "alternative_trial_ids": [race["trial_id"] for race in stage["top_races"][1:]],
                    "steps": [
                        {"type": "introduce_competitors", "competitors": stage["competitors"]},
                        {"type": "launch_race", **best["launch"]},
                        {"type": "show_results", "podium": best["podium"], "classification": best["classification"]},
                    ],
                }
            )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(plan, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_trial(
    stage_id: str,
    spec: RaceSpec,
    weights: InterestWeights,
    timeout: float,
    fingerprint: str,
    playback: dict[str, Any],
    duration_settings: dict[str, Any],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "schema_version": 1,
        "fingerprint": fingerprint,
        "stage_id": stage_id,
        "trial_id": trial_id(stage_id, spec),
        "spec": asdict(spec),
        "weights": asdict(weights),
        "started_at": utc_now(),
        "replay_command": replay_command(spec, **playback, title=spec.label),
    }
    started = time.monotonic()
    try:
        record.update(run_worker(spec, weights, timeout), status="completed")
        record["quality"] = race_quality(record, duration_settings)
    except subprocess.TimeoutExpired:
        record.update(status="timeout", timeout_seconds=timeout)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError) as error:
        record.update(status="error", error=str(error))
    record["wall_seconds"] = time.monotonic() - started
    return record


def race_quality(record: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Blend action with proximity to the target; time is simulated heat duration."""
    duration = float(record["duration_seconds"])
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("race duration must be finite and positive")
    target = settings["target_seconds"]
    tolerance = settings["duration_tolerance_seconds"]
    finishers = sum(not row["dnf"] for row in record["result"]["races"][0]["standings"])
    proximity = math.exp(-0.5 * ((duration - target) / tolerance) ** 2) if finishers else 0.0
    share = settings["duration_weight"]
    return {
        "score": (1 - share) * record["interest"]["score"] + share * 100 * proximity,
        "interest_score": record["interest"]["score"],
        "duration_score": 100 * proximity,
        "duration_points": share * 100 * proximity,
        "duration_seconds": duration,
        "target_seconds": target,
        "in_duration_window": bool(finishers) and abs(duration - target) <= tolerance + 1e-9,
        "finishers": finishers,
    }


def calibration_step(
    stage: dict[str, Any], base: RaceSpec, records: dict[str, dict[str, Any]], settings: dict[str, Any]
) -> tuple[dict[str, Any], RaceSpec | None]:
    """Replay bounded adaptive pilots deterministically, so interrupted work resumes."""
    target = settings["target_seconds"]
    tolerance = settings["duration_tolerance_seconds"]
    maximum = settings["max_calibration_laps"]
    candidate = min(base.round_laps, maximum)
    tried: dict[int, float | None] = {}
    attempts: list[str] = []
    for _ in range(settings["calibration_attempts"]):
        spec = replace(base, round_laps=candidate)
        identifier = trial_id(stage["id"], spec)
        record = records.get(identifier)
        if record is None:
            return {"status": "pending", "attempts": attempts, "laps": base.round_laps}, spec
        attempts.append(identifier)
        duration = (
            record["quality"]["duration_seconds"]
            if record["status"] == "completed" and record["quality"]["finishers"]
            else None
        )
        tried[candidate] = duration
        if duration is not None and abs(duration - target) <= tolerance + 1e-9:
            return {"status": "in_window", "attempts": attempts, "laps": candidate, "duration_seconds": duration}, None
        remaining = set(range(1, maximum + 1)) - tried.keys()
        if not remaining:
            break
        valid = {laps: seconds for laps, seconds in tried.items() if seconds is not None}
        if valid:
            closest = min(valid, key=lambda laps: (abs(valid[laps] - target), laps))
            estimate = closest * target / valid[closest]
            # Two pilots can estimate startup/finish overhead as well as seconds per lap.
            below = [laps for laps, seconds in valid.items() if seconds < target]
            above = [laps for laps, seconds in valid.items() if seconds > target]
            if below and above:
                low = max(below, key=lambda laps: valid[laps])
                high = min(above, key=lambda laps: valid[laps])
                estimate = low + (high - low) * (target - valid[low]) / (valid[high] - valid[low])
            candidate = min(remaining, key=lambda laps: (abs(laps - estimate), laps))
        else:
            candidate = min(remaining, key=lambda laps: (abs(laps - max(1, candidate // 2)), laps))
    valid = {laps: seconds for laps, seconds in tried.items() if seconds is not None}
    selected = min(valid, key=lambda laps: (abs(valid[laps] - target), laps)) if valid else base.round_laps
    return {
        "status": "outside_window" if valid else "unavailable",
        "attempts": attempts,
        "laps": selected,
        "duration_seconds": valid.get(selected),
    }, None


def calibrate_stages(
    stages: list[dict[str, Any]],
    plan: dict[str, Any],
    output: Path,
    records: dict[str, dict[str, Any]],
    jobs: int,
    timeout: float,
    retry_failed: bool,
) -> None:
    """Choose a lap count per field/layout before searching its full starting-seed grid."""
    settings = plan["settings"]
    units = [
        (stage, spec)
        for stage in stages
        if stage.get("calibrate_laps")
        for spec in stage_specs(stage, settings)
        if spec.seed == settings["start_seed"]
    ]
    if not units:
        return
    plan["status"] = "calibrating"
    # Retry failed pilots only once per invocation, avoiding endless retry cycles.
    if retry_failed:
        by_id = {stage["id"]: stage for stage, _ in units}
        failed = [
            (by_id[record["stage_id"]], RaceSpec.from_dict(record["spec"]))
            for record in list(records.values())
            if record.get("phase") == "calibration" and record["status"] != "completed" and record["stage_id"] in by_id
        ]
        if failed:
            search_stages(stages, plan, output, records, jobs, timeout, True, trials=failed, phase="calibration")
    while True:
        pending = []
        for stage, base in units:
            calibration, candidate = calibration_step(stage, base, records, settings)
            stage.setdefault("lap_calibration", {})[str(base.track_seed)] = calibration
            if candidate is not None:
                pending.append((stage, candidate))
        save_plan(output, plan)
        if not pending:
            break
        search_stages(stages, plan, output, records, jobs, timeout, False, trials=pending, phase="calibration")
    for stage, base in units:
        calibration = stage["lap_calibration"][str(base.track_seed)]
        print(
            f"Calibration {stage['title']}, track {base.track_seed}: {calibration['laps']} laps, "
            f"{calibration['duration_seconds']} seconds ({calibration['status']})",
            flush=True,
        )
        if calibration["status"] != "in_window":
            warning = (
                f"{stage['title']}, track {base.track_seed}: calibration {calibration['status']}; "
                f"using {calibration['laps']} laps"
            )
            if warning not in plan["warnings"]:
                plan["warnings"].append(warning)
    plan["status"] = "running"
    save_plan(output, plan)


def read_records(path: Path, fingerprint: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            try:
                record = json.loads(line)
            except ValueError as error:
                raise ValueError(f"{path}:{number}: incomplete/corrupt trial record") from error
            if record["fingerprint"] != fingerprint:
                raise ValueError("trial log belongs to a different export, engine, or search configuration")
            records[record["trial_id"]] = record
    return records


def search_stages(
    stages: list[dict[str, Any]],
    plan: dict[str, Any],
    output: Path,
    records: dict[str, dict[str, Any]],
    jobs: int,
    timeout: float,
    retry_failed: bool,
    *,
    trials: list[tuple[dict[str, Any], RaceSpec]] | None = None,
    phase: str = "search",
) -> None:
    settings = plan["settings"]
    weights = InterestWeights(**settings["weights"])
    requested = (
        trials if trials is not None else [(stage, spec) for stage in stages for spec in stage_specs(stage, settings)]
    )
    pending = [
        (stage, spec)
        for stage, spec in requested
        if trial_id(stage["id"], spec) not in records
        or (retry_failed and records[trial_id(stage["id"], spec)]["status"] != "completed")
    ]
    for stage in stages:
        rank_stage(stage, records, settings)
    save_plan(output, plan)
    print(f"{phase.capitalize()}: {len(pending)} pending races with {jobs} workers ...", flush=True)
    with Path(plan["trial_log"]).open("a", encoding="utf-8") as log, ThreadPoolExecutor(max_workers=jobs) as pool:
        work = iter(pending)
        active: dict[Any, dict[str, Any]] = {}

        def submit_next() -> None:
            item = next(work, None)
            if item is not None:
                stage, spec = item
                active[
                    pool.submit(
                        run_trial,
                        stage["id"],
                        spec,
                        weights,
                        timeout,
                        plan["fingerprint"],
                        settings["playback"],
                        settings,
                    )
                ] = stage

        for _ in range(jobs):
            submit_next()
        done_count = 0
        while active:
            finished, _ = wait(active, timeout=0.5, return_when=FIRST_COMPLETED)
            for future in finished:
                stage = active.pop(future)
                record = future.result()
                record["phase"] = phase
                # Validate classifications before publishing a successful trial.
                if record["status"] == "completed":
                    try:
                        classification(stage, record)
                    except (ValueError, KeyError, TypeError) as error:
                        record.update(status="error", error=f"invalid race classification: {error}")
                log.write(json.dumps(record, allow_nan=False) + "\n")
                log.flush()
                records[record["trial_id"]] = record
                rank_stage(stage, records, settings)
                save_plan(output, plan)
                done_count += 1
                score = (
                    f"score {record['quality']['score']:.2f}, {record['duration_seconds']:.1f}s"
                    if record["status"] == "completed"
                    else record["status"]
                )
                print(
                    f"[{done_count}/{len(pending)}] {stage['title']}: track {record['spec']['track_seed']}, "
                    f"start {record['spec']['seed']}: {score}",
                    flush=True,
                )
                submit_next()


def print_results(plan: dict[str, Any]) -> None:
    for stage in plan_stages(plan):
        print(f"\n{stage['title']}:", flush=True)
        for race in stage["top_races"]:
            podium = ", ".join(
                f"P{row['finish_position']} {row['display_name']} #{row['submission_id']}" for row in race["podium"]
            )
            print(
                f"  #{race['interest_rank']}: {race['quality']['score']:.2f}/100; "
                f"{race['spec']['round_laps']} laps, {race['quality']['duration_seconds']:.1f}s; "
                f"track={race['spec']['track']}:{race['spec']['track_seed']}, start={race['spec']['seed']}; {podium}"
            )
            print(f"  # {stage['title']} - replay {race['interest_rank']}")
            print(f"  {race['launch']['shell_command']}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "race-night" / "plan.json")
    parser.add_argument("--review-csv", type=Path, help="review_submissions.py CSV used to select juiced holdouts")
    parser.add_argument(
        "--juiced-threshold",
        type=nonnegative_float,
        default=1.0,
        help="hold out total_score greater than or equal to this value (default: 1)",
    )
    parser.add_argument(
        "--juiced-min-laps",
        type=nonnegative_float,
        help="also require all_spawns_no_crumbs_laps >= this value in the review CSV (default: no lap filter)",
    )
    parser.add_argument(
        "--juiced-track",
        choices=("procedural", "mugello-short", "bahrain"),
        default="procedural",
        help="track for the separate juiced heat(s); laps are set with --laps juiced=N",
    )
    parser.add_argument(
        "--selection-priority",
        choices=("highest", "lowest"),
        default="highest",
        help="which category claims overlapping submissions first (default: highest)",
    )
    parser.add_argument("--track-count", type=positive_int, default=10)
    parser.add_argument("--starts-per-track", type=positive_int, default=10)
    parser.add_argument(
        "--fixed-seed-count",
        type=positive_int,
        help="starts for Clock It and fixed non-Bahrain Juiced tracks (default: track-count * starts-per-track)",
    )
    parser.add_argument(
        "--bahrain-seed-count", type=positive_int, default=100, help="seeds per Bahrain field (default: 100)"
    )
    parser.add_argument(
        "--bahrain-cars",
        type=int,
        choices=HEAT_ENTRANT_COUNTS,
        default=10,
        help="cars per Bahrain field; use 20 for a full grid (default: 10)",
    )
    parser.add_argument(
        "--target-seconds", type=positive_float, default=90.0, help="target simulated heat duration (default: 90)"
    )
    parser.add_argument(
        "--duration-tolerance-seconds",
        type=positive_float,
        default=10.0,
        help="calibration window half-width (default: 10)",
    )
    parser.add_argument(
        "--duration-weight",
        type=nonnegative_float,
        default=0.2,
        help="fraction of ranking score from duration proximity, 0 to 1 (default: 0.2)",
    )
    parser.add_argument(
        "--calibrate-laps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="calibrate non-Bahrain laps per field/layout before searching (default: enabled)",
    )
    parser.add_argument(
        "--calibration-attempts",
        type=positive_int,
        default=6,
        help="maximum pilot lap counts per field/layout (default: 6)",
    )
    parser.add_argument(
        "--max-calibration-laps",
        type=positive_int,
        default=30,
        help="largest lap count calibration may try (default: 30)",
    )
    parser.add_argument("--start-track-seed", type=int, default=0)
    parser.add_argument("--start-seed", type=int, default=110)
    parser.add_argument("--damage", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--marshal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--marshal-stuck-seconds", type=positive_float, default=1.0)
    parser.add_argument("--marshal-penalty-m", type=nonnegative_float, default=5.0)
    parser.add_argument("--marshal-cooldown-seconds", type=nonnegative_float, default=2.0)
    parser.add_argument("--finish-timeout-seconds", type=nonnegative_float, default=10.0)
    parser.add_argument(
        "--laps",
        action="append",
        default=[],
        metavar="CATEGORY=N",
        help="fix laps and skip calibration for a category, e.g. clock-it=12 or juiced=6; Bahrain stays at 3",
    )
    parser.add_argument(
        "--camera",
        choices=[view.value for view in CameraView if view is not CameraView.SPLIT_FOLLOW],
        default="cinematic",
        help="camera in generated replay commands",
    )
    parser.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--music", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--muted", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout-seconds", type=positive_float, default=300.0)
    parser.add_argument("--jobs", type=positive_int, default=4)
    parser.add_argument(
        "--weights", type=float, nargs=3, default=(30.0, 30.0, 40.0), metavar=("LEAD", "PASS", "FINISH")
    )
    parser.add_argument("--finish-gap-scale-seconds", type=positive_float, default=2.0)
    parser.add_argument("--plan-only", action="store_true", help="select and inspect groups without simulating races")
    parser.add_argument(
        "--resume", action="store_true", help="reuse saved trials after verifying source/settings hashes"
    )
    parser.add_argument("--retry-failed", action="store_true", help="with --resume, retry errors and timeouts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    log_path = output.with_suffix(".trials.jsonl")
    export_dir = args.export_dir.expanduser().resolve()
    review_csv = args.review_csv.expanduser().resolve() if args.review_csv else None
    try:
        if args.duration_weight > 1:
            raise ValueError("--duration-weight must be between 0 and 1")
        if args.duration_tolerance_seconds >= args.target_seconds:
            raise ValueError("--duration-tolerance-seconds must be less than --target-seconds")
        if args.retry_failed and not args.resume:
            raise ValueError("--retry-failed requires --resume")
        if (output.exists() or log_path.exists()) and not args.resume:
            raise ValueError(f"{output} or its log already exists; use --resume or a new --output")
        if args.resume and not output.exists():
            raise ValueError("--resume requires an existing plan JSON")
        if review_csv is None and (args.juiced_threshold != 1.0 or args.juiced_min_laps is not None):
            raise ValueError("--juiced-threshold and --juiced-min-laps require --review-csv")
        weights = InterestWeights(*args.weights, finish_gap_scale_seconds=args.finish_gap_scale_seconds)
        review_scores = read_review_scores(review_csv, export_dir) if review_csv else None
        review_laps = (
            read_review_scores(review_csv, export_dir, column="all_spawns_no_crumbs_laps")
            if review_csv and args.juiced_min_laps is not None
            else None
        )
        groups, warnings = select_groups(
            export_dir,
            args.selection_priority,
            review_scores=review_scores,
            juiced_threshold=args.juiced_threshold,
            review_laps=review_laps,
            juiced_min_laps=args.juiced_min_laps,
        )
        held_out_scores = filter_juiced_scores(
            review_scores or {}, args.juiced_threshold, lap_scores=review_laps, min_laps=args.juiced_min_laps
        )
        bahrain_groups = build_bahrain_groups(export_dir, set(held_out_scores), args.bahrain_cars)
        lap_counts = {category.id: category.laps for category in CATEGORIES} | {
            "bahrain-no-juiced": 3,
            "bahrain-open": 3,
            "juiced": 6,
        }
        fixed_laps: set[str] = set()
        for override in args.laps:
            category_id, separator, laps = override.partition("=")
            if not separator or category_id not in lap_counts or not laps.isdecimal() or int(laps) < 1:
                raise ValueError(
                    f"invalid --laps {override!r}; use CATEGORY=N with positive N; categories: {', '.join(lap_counts)}"
                )
            lap_counts[category_id] = int(laps)
            fixed_laps.add(category_id)
            if category_id.startswith("bahrain-") and int(laps) != 3:
                raise ValueError("Bahrain fields run exactly 3 laps")
        for group in groups:
            group["round_laps"] = lap_counts[group["id"]]
        juiced_groups, juiced_warnings = build_juiced_groups(
            export_dir, held_out_scores, track=args.juiced_track, laps=lap_counts["juiced"], lap_scores=review_laps
        )
        warnings.extend(juiced_warnings)
        initial_stages = [*groups, *juiced_groups, *bahrain_groups]
        for stage in initial_stages:
            category_id = "juiced" if stage["id"].startswith("juiced") else stage["id"]
            stage["calibrate_laps"] = (
                args.calibrate_laps and stage["track"] != "bahrain" and category_id not in fixed_laps
            )
        rules = HeadToHeadRaceRules(
            damage_enabled=args.damage,
            marshal_enabled=args.marshal,
            marshal_stuck_seconds=args.marshal_stuck_seconds,
            marshal_penalty_m=args.marshal_penalty_m,
            marshal_cooldown_seconds=args.marshal_cooldown_seconds,
        )
        settings: dict[str, Any] = {
            "export_dir": str(export_dir),
            "selection_priority": args.selection_priority,
            "review_csv": str(review_csv) if review_csv else None,
            "juiced_threshold": args.juiced_threshold,
            "juiced_min_laps": args.juiced_min_laps,
            "juiced_track": args.juiced_track,
            "track_count": args.track_count,
            "starts_per_track": args.starts_per_track,
            "fixed_seed_count": args.fixed_seed_count or args.track_count * args.starts_per_track,
            "bahrain_seed_count": args.bahrain_seed_count,
            "bahrain_cars": args.bahrain_cars,
            "target_seconds": args.target_seconds,
            "duration_tolerance_seconds": args.duration_tolerance_seconds,
            "duration_weight": args.duration_weight,
            "calibrate_laps": args.calibrate_laps,
            "calibration_attempts": args.calibration_attempts,
            "max_calibration_laps": args.max_calibration_laps,
            "fixed_lap_categories": sorted(fixed_laps),
            "start_seed": args.start_seed,
            "start_track_seed": args.start_track_seed,
            "rules": asdict(rules),
            "finish_timeout_seconds": args.finish_timeout_seconds,
            "weights": asdict(weights),
            "laps": lap_counts,
            "playback": {key: getattr(args, key) for key in ("camera", "fullscreen", "audio", "music", "muted")},
        }
        source = source_fingerprint(export_dir, initial_stages)
        if review_csv:
            source["review_csv_sha256"] = file_digest(review_csv)
        # Playback-only changes can regenerate the show plan without rerunning physics.
        fingerprint = digest({"source": source, "settings": {k: v for k, v in settings.items() if k != "playback"}})
        previous = json.loads(output.read_text()) if args.resume else None
        if previous is not None and previous["fingerprint"] != fingerprint:
            raise ValueError("export, controller/engine sources, or settings changed; use a new --output")
        enrich_competitors(initial_stages, export_dir, args.jobs, min(args.timeout_seconds, 30.0))
        records = read_records(log_path, fingerprint) if args.resume else {}
    except (ValueError, OSError, csv.Error, subprocess.TimeoutExpired) as error:
        parser.error(str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    plan: dict[str, Any] = {
        "schema_version": 2,
        "status": "planned" if args.plan_only else "running",
        "created_at": previous["created_at"] if previous else utc_now(),
        "source": source,
        "fingerprint": fingerprint,
        "settings": settings,
        "warnings": warnings,
        "trial_log": str(log_path),
        "groups": groups,
        "juiced_groups": juiced_groups,
        "bahrain_groups": bahrain_groups,
        "holdout": {
            "enabled": review_csv is not None,
            "review_csv": str(review_csv) if review_csv else None,
            "score_column": "total_score",
            "comparison": ">=",
            "threshold": args.juiced_threshold,
            "lap_column": "all_spawns_no_crumbs_laps",
            "lap_comparison": ">=",
            "min_laps": args.juiced_min_laps,
            "criteria_operator": "and",
            "excluded_from_bahrain_no_juiced": True,
            "eligible_for_bahrain_open": True,
            "submissions": [
                {
                    "submission_id": identifier,
                    "total_score": held_out_scores[identifier],
                    **({"all_spawns_no_crumbs_laps": review_laps.get(identifier)} if review_laps is not None else {}),
                }
                for identifier in sorted(held_out_scores, key=int)
            ],
        },
        "bahrain_qualification": (
            f"Independent All Spawns, No Crumbs top {args.bahrain_cars}: excluding Juiced, then open to everyone"
        ),
        "selection_policy": (
            f"{args.selection_priority.capitalize()} category first; "
            "backfill each to 10 after juiced holdouts; regular groups are exclusive, Bahrain fields are independent"
        ),
        "allocation_order": [g["id"] for g in sorted(groups, key=lambda g: g["allocation_rank"])],
        "color_policy": "RACING_COLOR if declared, otherwise that stage's grid palette (matches replay)",
    }
    for stage in initial_stages:
        rank_stage(stage, records, settings)
        print(
            f"{stage['title']}: "
            + ", ".join(
                f"{c['display_name']} #{c['submission_id']}"
                + (
                    f" (review {c['review_total_score']:g})"
                    if "review_total_score" in c
                    else f" (LB {c['leaderboard_rank']})"
                )
                for c in stage["competitors"]
            ),
            flush=True,
        )
    if review_csv:
        criteria = f"total_score >= {args.juiced_threshold:g}"
        if args.juiced_min_laps is not None:
            criteria += f" and all_spawns_no_crumbs_laps >= {args.juiced_min_laps:g}"
        print(
            f"Juiced holdout: {len(held_out_scores)} submissions "
            f"with {criteria}; excluded from regular categories and the no-Juiced Bahrain field.",
            flush=True,
        )
    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr, flush=True)
    total = sum(len(stage_specs(stage, settings)) for stage in initial_stages)
    print(
        f"Planned {total} seed trials including both Bahrain fields, plus bounded lap calibration. JSON: {output}",
        flush=True,
    )
    save_plan(output, plan)
    if args.plan_only:
        return 0
    try:
        calibrate_stages(initial_stages, plan, output, records, args.jobs, args.timeout_seconds, args.retry_failed)
        search_stages(initial_stages, plan, output, records, args.jobs, args.timeout_seconds, args.retry_failed)
    except KeyboardInterrupt:
        plan["status"] = "interrupted"
        save_plan(output, plan)
        print(f"\nInterrupted. Completed trials are saved; rerun with --resume. JSON: {output}", flush=True)
        return 130
    except (ValueError, OSError) as error:
        plan.update(status="blocked", error=str(error))
        save_plan(output, plan)
        print(str(error), file=sys.stderr)
        print_results(plan)
        return 2
    failures = sum(stage["search"]["failed"] for stage in plan_stages(plan))
    plan["status"] = "complete_with_errors" if failures else "complete"
    save_plan(output, plan)
    print_results(plan)
    print(f"\nShow plan: {output}\nDetailed trials: {log_path}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    if sys.argv[1:] == ["--inspect-controller"]:
        request = json.load(sys.stdin)
        with redirect_stdout(sys.stderr):
            submission = load_student_submission(request["path"], suppress_prints=True)
        print(
            json.dumps(
                {"racing_name": submission.display_name, "declared_color_rgba": submission.car_color}, allow_nan=False
            )
        )
    else:
        raise SystemExit(main())
