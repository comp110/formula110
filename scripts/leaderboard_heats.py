#!/usr/bin/env python3
"""Print race commands for the leading cars in each exported leaderboard category."""

from __future__ import annotations

import argparse
import math
import re
import shlex
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import yaml
from run_submissions import DEFAULT_EXPORT_DIR, PROJECT_ROOT, resolve_controller


@dataclass(frozen=True)
class Leader:
    submission_id: str
    name: str
    value: float


@dataclass
class Leaderboard:
    name: str
    order: str
    leaders: list[Leader] = field(default_factory=list[Leader])

    def ranked(self) -> list[Leader]:
        """Break equal scores by numeric submission ID for reproducible selection."""
        return sorted(
            self.leaders,
            key=lambda leader: (
                leader.value if self.order == "asc" else -leader.value,
                int(leader.submission_id),
            ),
        )


def display_name(submission: dict[str, object], identifier: str) -> str:
    """Join each submitter's first/last initials in the metadata's team order."""
    submitters = submission.get(":submitters", submission.get("submitters"))
    if not isinstance(submitters, list) or not submitters:
        return f"#{identifier}"
    initials: list[str] = []
    for raw_submitter in cast(list[object], submitters):
        if not isinstance(raw_submitter, dict):
            return f"#{identifier}"
        submitter = cast(dict[str, object], raw_submitter)
        name = submitter.get(":name", submitter.get("name"))
        if not isinstance(name, str):
            return f"#{identifier}"
        parts = ["".join(char for char in part if char.isalpha()) for part in name.split()]
        parts = [part for part in parts if part]
        if not parts:
            return f"#{identifier}"
        initials.append((parts[0][0] + parts[-1][0] if len(parts) > 1 else parts[0][:2]).upper())
    return "+".join(initials)


def read_leaderboards(export_dir: Path) -> tuple[list[Leaderboard], list[str]]:
    """Use only current results; history can describe code absent from the export."""
    metadata = export_dir / "submission_metadata.yml"
    try:
        payload: object = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {metadata}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata}: expected a mapping of submission_ID records")
    boards: dict[str, Leaderboard] = {}
    warnings: list[str] = []
    for key, raw_submission in cast(dict[object, object], payload).items():
        if not isinstance(key, str) or re.fullmatch(r"submission_[0-9]+", key) is None:
            continue
        identifier = key.removeprefix("submission_")
        if not isinstance(raw_submission, dict):
            raise ValueError(f"{key}: expected a submission record")
        submission = cast(dict[str, object], raw_submission)
        raw_results = submission.get(":results", submission.get("results"))
        if raw_results is None:
            continue
        if not isinstance(raw_results, dict):
            raise ValueError(f"{key}: expected a results mapping")
        raw_entries = cast(dict[str, object], raw_results).get("leaderboard")
        if raw_entries is None:
            continue
        if not isinstance(raw_entries, list):
            raise ValueError(f"{key}: expected a leaderboard list")
        entries = cast(list[object], raw_entries)
        if not entries:
            continue
        try:
            resolve_controller(export_dir, identifier)
        except ValueError as error:
            warnings.append(str(error))
            name = None
        else:
            name = display_name(submission, identifier)
        seen: set[str] = set()
        for raw_entry in entries:
            if not isinstance(raw_entry, dict):
                raise ValueError(f"{key}: expected a leaderboard entry mapping")
            entry = cast(dict[str, object], raw_entry)
            metric = entry.get("name")
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError(f"{key}: leaderboard entry needs a name")
            if metric in seen:
                raise ValueError(f"{key}: duplicate leaderboard entry {metric!r}")
            seen.add(metric)
            order = entry.get("order", "desc")
            if order not in ("asc", "desc"):
                raise ValueError(f"{key}: {metric!r} has invalid order {order!r}")
            board = boards.setdefault(metric, Leaderboard(metric, str(order)))
            if board.order != order:
                raise ValueError(f"{metric!r}: conflicting sort orders in current submissions")
            value = entry.get("value")
            if name is None or value is None or isinstance(value, bool):
                continue
            if not isinstance(value, (int, float, str)):
                continue
            try:
                score = float(value)
            except (ValueError, OverflowError):
                continue
            if math.isfinite(score):
                board.leaders.append(Leader(identifier, name, score))
    return list(boards.values()), warnings


def normalized_name(value: str) -> str:
    """Allow partial names, slugs, and punctuation such as 'g's going crazy'."""
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def select_boards(boards: list[Leaderboard], queries: Sequence[str]) -> list[Leaderboard]:
    selected: list[Leaderboard] = []
    for query in queries:
        normalized = normalized_name(query)
        exact = [board for board in boards if normalized_name(board.name) == normalized]
        matches = exact or [board for board in boards if normalized and normalized in normalized_name(board.name)]
        if len(matches) != 1:
            detail = "ambiguous" if matches else "unknown"
            available = ", ".join(board.name for board in (matches or boards))
            raise ValueError(f"{detail} metric {query!r}; choose from: {available}")
        if matches[0] not in selected:
            selected.append(matches[0])
    return selected if queries else boards


def heat_command(leaders: Sequence[Leader], export_dir: Path, race_args: Sequence[str]) -> str:
    """Quote every argument, including car labels, for a POSIX shell."""
    if Path.cwd().resolve() == PROJECT_ROOT:
        command = ["uv", "run", "python", "scripts/run_submissions.py"]
    else:
        command = [
            "uv",
            "run",
            "--project",
            str(PROJECT_ROOT),
            "python",
            str(PROJECT_ROOT / "scripts/run_submissions.py"),
        ]
    command.extend(leader.submission_id for leader in leaders)
    if export_dir != DEFAULT_EXPORT_DIR:
        command.extend(["--export-dir", str(export_dir)])
    for leader in leaders:
        command.extend(["--name", leader.name])
    command.extend(race_args)
    return shlex.join(command)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        epilog="Race options go after --, e.g. -- --fullscreen --seed 110 --races 3. No races are launched.",
    )
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR, help="extracted assignment directory")
    parser.add_argument(
        "--metric", action="append", default=[], help="category name or unique substring; repeat to select"
    )
    parser.add_argument("--cars", type=int, choices=(4, 8), default=8, help="cars per heat (default: 8)")
    parser.add_argument("--list", action="store_true", help="list categories, sort directions, and eligible car counts")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    separator = arguments.index("--") if "--" in arguments else len(arguments)
    args = parser.parse_args(arguments[:separator])
    race_args = arguments[separator + 1 :]
    export_dir = cast(Path, args.export_dir).expanduser().resolve()
    try:
        boards, warnings = read_leaderboards(export_dir)
        if not boards:
            raise ValueError("no current leaderboard entries found in submission_metadata.yml")
        boards = select_boards(boards, cast(list[str], args.metric))
    except ValueError as error:
        parser.error(str(error))
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    for board in boards:
        direction = "lower" if board.order == "asc" else "higher"
        title = " ".join(board.name.split())
        print(f"# {title} — {direction} is better; {len(board.leaders)} eligible cars")
        if args.list:
            continue
        leaders = board.ranked()[: args.cars]
        for rank, leader in enumerate(leaders, start=1):
            print(f"# {rank}. {leader.name} (submission {leader.submission_id}): {leader.value:g}")
        if len(leaders) < args.cars:
            print(f"# Skipped: need {args.cars} distinct eligible cars; found {len(leaders)}.")
        else:
            print(heat_command(leaders, export_dir, race_args))
        print()


if __name__ == "__main__":
    main()
