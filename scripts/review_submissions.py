#!/usr/bin/env python3
"""Rank manifest-selected controllers by course-convention review signals.

Never imports or runs submitted code. Scores describe observable syntax, not
the probability that a submission used AI. Includes exported race metrics.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import tokenize
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TextIO

import yaml

SUBMISSION_MANIFEST = "formula110-submission.json"
EXERCISE_MANIFEST = "formula110-exercise-submission.json"
SCORE_NOTE = "Review priority from syntax/convention signals; not an AI probability or proof of authorship."
RACE_METRICS = {
    "all_spawns_no_crumbs_laps": "All Spawns, No Crumbs (Laps)",
    "clean_lap_seconds": "Clock It (s)",
}
EXPERIENCE_LEVELS = (
    "None",
    "A little experience (e.g., brief exposure, self-guided learning)",
    "Some experience (e.g., a high school course, personal projects)",
    "Substantial experience (e.g., multiple courses or significant project experience)",
)
EXPERIENCE_COLUMNS = ("experience_level", "experience_rank", "experience_match")


@dataclass(frozen=True)
class Rule:
    weight: int
    max_occurrences: int
    description: str


RULES = {
    "precise_float": Rule(2, 5, "Float literal with at least --float-digits significant mantissa digits"),
    "nonlocal": Rule(5, 2, "nonlocal declaration (enclosing-function state)"),
    "callable": Rule(3, 2, "callable() or a Callable type"),
    "persistent_state": Rule(5, 4, "Potential state via function/object attributes, defaults, closures, or caches"),
    "unannotated_local": Rule(
        0, 10, "Local assigned without an annotation anywhere in its function scope (disabled by default)"
    ),
    "versioned_name": Rule(2, 1, "RACING_NAME contains a version marker such as v2, version 3, or 2.1"),
    "tuple": Rule(2, 4, "Tuple type, tuple() construction, or tuple value (not unpacking/type argument lists)"),
    "for_loop": Rule(3, 4, "for/async for loop or comprehension generator"),
    "zip": Rule(4, 2, "zip() call"),
    "class": Rule(6, 2, "Class definition"),
}

VERSION_PATTERN = re.compile(
    r"(?<![a-z])(?:v(?:ersion)?|ver|rev(?:ision)?|mk)[\s._-]*\d+(?:\.\d+)*\b|\b\d+\.\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
MUTATORS = {
    "append",
    "extend",
    "insert",
    "pop",
    "remove",
    "clear",
    "update",
    "setdefault",
    "add",
    "discard",
    "sort",
    "reverse",
}
FUNCTIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass(frozen=True)
class Finding:
    rule: str
    line: int
    column: int
    message: str
    snippet: str


@dataclass
class SubmissionResult:
    submission_id: str
    submission_dir: str
    status: str = "error"
    manifest: str | None = None
    controller_module: str | None = None
    controller_path: str | None = None
    car_name: str | None = None
    score: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    points: dict[str, int] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    error: str | None = None
    race_scores: dict[str, float] = field(default_factory=dict)
    experience_rank: int | None = None
    experience_match: str = ""


@dataclass
class ExperienceSurvey:
    responses: dict[int, int | None] = field(default_factory=dict)
    by_student_id: dict[str, set[int]] = field(default_factory=dict)
    by_email: dict[str, set[int]] = field(default_factory=dict)

    def match(self, submitter: dict[str, object]) -> int | None:
        """Use exact identity keys; disagreeing or ambiguous matches stay unknown."""
        student_id = str(submitter.get(":sid", submitter.get("sid")) or "").strip()
        email = str(submitter.get(":email", submitter.get("email")) or "").strip().casefold()
        by_id = self.by_student_id.get(student_id, set())
        by_email = self.by_email.get(email, set())
        candidates = by_id & by_email if by_id and by_email else by_id or by_email
        if len(candidates) != 1:
            return None
        return self.responses[next(iter(candidates))]


def read_experience_survey(path: Path) -> ExperienceSurvey:
    """Read only the roster identities and Question 1.1 self-reported experience."""
    survey = ExperienceSurvey()
    levels = {
        label.casefold(): rank
        for rank, full_label in enumerate(EXPERIENCE_LEVELS)
        for label in (full_label, full_label.split(" (")[0])
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if (
            "Question 1.1 Response" not in fields
            or not {"Student ID", "Email"}.intersection(fields)
            or len(fields) != len(set(fields))
        ):
            raise ValueError(f"{path}: expected unique Question 1.1 Response and Student ID and/or Email CSV columns")
        for row in reader:
            response = " ".join((row.get("Question 1.1 Response") or "").split()).casefold()
            if response and response not in levels:
                raise ValueError(f"{path}:{reader.line_num}: unrecognized Question 1.1 Response")
            survey.responses[reader.line_num] = levels[response] if response else None
            student_id = (row.get("Student ID") or "").strip()
            email = (row.get("Email") or "").strip().casefold()
            if student_id:
                survey.by_student_id.setdefault(student_id, set()).add(reader.line_num)
            if email:
                survey.by_email.setdefault(email, set()).add(reader.line_num)
    return survey


def collate_experience(results: list[SubmissionResult], metadata: dict[str, object], survey: ExperienceSurvey) -> None:
    """Take the maximum known response across all current submitters in a team."""
    for result in results:
        record = metadata.get(f"submission_{result.submission_id}")
        submitters = record.get(":submitters", record.get("submitters")) if isinstance(record, dict) else None
        ranks = []
        if isinstance(submitters, list):
            for submitter in submitters:
                rank = survey.match(submitter) if isinstance(submitter, dict) else None
                if rank is not None:
                    ranks.append(rank)
        result.experience_rank = max(ranks) if ranks else None
        result.experience_match = (
            "unmatched" if not ranks else "complete" if len(ranks) == len(submitters) else "partial"
        )


def target_names(node: ast.AST) -> list[ast.Name]:
    """Unpacking binds names; writing an attribute or subscript does not."""
    if isinstance(node, ast.Name):
        return [node]
    if isinstance(node, (ast.Tuple, ast.List)):
        return [name for item in node.elts for name in target_names(item)]
    if isinstance(node, ast.Starred):
        return target_names(node.value)
    return []


class Bindings(ast.NodeVisitor):
    """Collect declarations in one scope without leaking nested-scope locals."""

    def __init__(self, node: ast.AST) -> None:
        self.assigned: dict[str, ast.AST] = {}
        self.annotated: set[str] = set()
        self.parameters: set[str] = set()
        self.external: set[str] = set()
        self.other: set[str] = set()
        if isinstance(node, FUNCTIONS):
            args = node.args
            self.parameters = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
            self.parameters.update(arg.arg for arg in (args.vararg, args.kwarg) if arg is not None)
        if isinstance(node, ast.Lambda):
            self.visit(node.body)
        else:
            for statement in node.body:
                self.visit(statement)

    @property
    def locals(self) -> set[str]:
        return (self.assigned.keys() | self.annotated | self.parameters | self.other) - self.external

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.assigned.setdefault(node.id, node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.annotated.update(name.id for name in target_names(node.target))
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.external.update(node.names)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.external.update(node.names)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.assigned.setdefault(node.name, node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        self.other.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self.other.update(alias.asname or alias.name for alias in node.names)

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.other.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.other.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        pass

    def visit_ListComp(self, node: ast.AST) -> None:
        # Comprehension targets have their own scope. The loop rule covers them.
        pass

    visit_SetComp = visit_ListComp
    visit_DictComp = visit_ListComp
    visit_GeneratorExp = visit_ListComp


class ControllerAnalysis:
    def __init__(self, source: str, filename: str, float_digits: int = 8) -> None:
        self.source = source
        self.lines = source.splitlines()
        self.tree = ast.parse(source, filename=filename)
        self.nodes = list(ast.walk(self.tree))
        self.parents = {child: parent for parent in self.nodes for child in ast.iter_child_nodes(parent)}
        self.bindings = {node: Bindings(node) for node in self.nodes if isinstance(node, FUNCTIONS)}
        self.function_names = {
            node.name for node in self.nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.aliases: dict[str, str] = {}
        for node in self.nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.aliases[alias.asname or alias.name.split(".")[0]] = (
                        alias.name if alias.asname else alias.name.split(".")[0]
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    self.aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        self.float_digits = float_digits
        self.findings: list[Finding] = []
        self.seen: set[tuple[str, int, int, str]] = set()
        self.car_name: str | None = None

    def add(self, rule: str, node: ast.AST, message: str) -> None:
        line = getattr(node, "lineno", 1)
        column = getattr(node, "col_offset", 0) + 1
        key = (rule, line, column, message)
        if key not in self.seen:
            self.seen.add(key)
            self.findings.append(Finding(rule, line, column, message, self.lines[line - 1].strip()))

    def qualified_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            return f"{self.qualified_name(node.value)}.{node.attr}"
        return ""

    def enclosing_function(self, node: ast.AST) -> ast.AST | None:
        parent = self.parents.get(node)
        while parent is not None:
            if isinstance(parent, FUNCTIONS):
                return parent
            if isinstance(parent, ast.ClassDef):
                return None
            parent = self.parents.get(parent)
        return None

    def state_owner(self, node: ast.AST, at: ast.AST) -> str | None:
        """Recognize attribute state and mutated closure bindings, excluding globals."""
        root = node
        while isinstance(root, (ast.Attribute, ast.Subscript)):
            root = root.value
        if not isinstance(root, ast.Name):
            return None
        if root.id in self.function_names:
            return f"function attribute state on {root.id}"
        function = self.enclosing_function(at)
        if function is None:
            return None
        if root.id in {"self", "cls"}:
            return f"instance/class state on {root.id}"
        scope = self.bindings[function]
        if root.id in scope.locals or root.id in scope.external:
            return None
        outer = self.enclosing_function(function)
        while outer is not None:
            if root.id in self.bindings[outer].locals:
                return f"mutation of enclosing-function binding {root.id}"
            outer = self.enclosing_function(outer)
        return None

    def check_state(self, node: ast.AST) -> None:
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Delete)):
            targets = node.targets if isinstance(node, (ast.Assign, ast.Delete)) else [node.target]
            for target in targets:
                for child in ast.walk(target):
                    if isinstance(child, (ast.Attribute, ast.Subscript)) and isinstance(
                        child.ctx, (ast.Store, ast.Del)
                    ):
                        owner = self.state_owner(child, node)
                        if owner:
                            self.add("persistent_state", child, owner)
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute) and node.func.attr in MUTATORS:
                owner = self.state_owner(node.func.value, node)
                if owner:
                    self.add("persistent_state", node, owner)
            if (
                self.qualified_name(node.func) in {"setattr", "builtins.setattr", "delattr", "builtins.delattr"}
                and node.args
            ):
                owner = self.state_owner(node.args[0], node)
                if owner:
                    self.add("persistent_state", node, owner)
        if isinstance(node, FUNCTIONS):
            for default in [*node.args.defaults, *node.args.kw_defaults]:
                if isinstance(default, (ast.List, ast.Dict, ast.Set, *COMPREHENSIONS[:3])) or (
                    isinstance(default, ast.Call) and self.qualified_name(default.func) in {"list", "dict", "set"}
                ):
                    self.add("persistent_state", default, "mutable default argument persists across calls")
            for decorator in getattr(node, "decorator_list", []):
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                if self.qualified_name(target) in {"functools.cache", "functools.lru_cache"}:
                    self.add("persistent_state", decorator, "cached function retains results across calls")

    def check_name(self, node: ast.AST) -> None:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            return
        # RACING_NAME is module metadata. Do not mistake local strings for it.
        if self.parents.get(node) is not self.tree:
            return
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if not any(name.id == "RACING_NAME" for target in targets for name in target_names(target)):
            return
        value = node.value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            self.car_name = value.value
            if VERSION_PATTERN.search(value.value):
                self.add("versioned_name", node, f"version marker in RACING_NAME: {value.value!r}")

    def check_float(self, node: ast.Constant) -> None:
        if not isinstance(node.value, float):
            return
        literal = ast.get_source_segment(self.source, node) or ""
        mantissa = literal.lower().replace("_", "").split("e")[0]
        # Ignore magnitude and zero padding: 1e-12 and 0.1000000000 are round.
        digits = mantissa.replace(".", "").lstrip("0").rstrip("0")
        if len(digits) >= self.float_digits:
            self.add("precise_float", node, f"{literal} has {len(digits)} significant mantissa digits")

    def run(self) -> tuple[str | None, list[Finding]]:
        for node in self.nodes:
            self.check_name(node)
            self.check_state(node)
            if isinstance(node, ast.Constant):
                self.check_float(node)
            elif isinstance(node, ast.Nonlocal):
                self.add("nonlocal", node, f"nonlocal {', '.join(node.names)} retains enclosing-function state")
            elif isinstance(node, ast.ClassDef):
                self.add("class", node, f"class {node.name}")
            elif isinstance(node, (ast.For, ast.AsyncFor)):
                self.add("for_loop", node, "for loop")
            elif isinstance(node, ast.comprehension):
                self.add("for_loop", node.target, "comprehension generator")
            elif isinstance(node, ast.Call):
                name = self.qualified_name(node.func)
                if name in {"zip", "builtins.zip"}:
                    self.add("zip", node, "zip() call")
                elif name in {"callable", "builtins.callable"}:
                    self.add("callable", node, "callable() check")
                elif name in {"tuple", "builtins.tuple"}:
                    self.add("tuple", node, "tuple() construction")
            elif isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
                name = self.qualified_name(node)
                if name in {"typing.Callable", "collections.abc.Callable"}:
                    self.add("callable", node, "Callable type")
                elif name in {"typing.Tuple", "tuple", "builtins.tuple"}:
                    parent = self.parents.get(node)
                    if not (isinstance(parent, ast.Call) and parent.func is node):
                        self.add("tuple", node, "tuple type/reference")
            elif isinstance(node, ast.Tuple) and isinstance(node.ctx, ast.Load):
                parent = self.parents.get(node)
                if not (isinstance(parent, ast.Subscript) and parent.slice is node):
                    self.add("tuple", node, "tuple value")
        for function, bindings in self.bindings.items():
            for name, node in bindings.assigned.items():
                if name in bindings.locals and name not in bindings.annotated | bindings.parameters:
                    label = getattr(function, "name", "<lambda>")
                    self.add("unannotated_local", node, f"{label}(): {name} has no local type annotation")
        self.findings.sort(key=lambda finding: (finding.line, finding.column, finding.rule, finding.message))
        return self.car_name, self.findings


def analyze_source(
    source: str, filename: str = "<controller>", *, float_digits: int = 8
) -> tuple[str | None, list[Finding]]:
    return ControllerAnalysis(source, filename, float_digits).run()


def resolve_controller(root: Path, result: SubmissionResult) -> Path:
    """Match run_submissions' manifest precedence without importing the simulator."""
    manifest = root / SUBMISSION_MANIFEST
    if not manifest.is_file():
        manifest = root / EXERCISE_MANIFEST
    if not manifest.is_file():
        raise ValueError(f"expected {SUBMISSION_MANIFEST} or {EXERCISE_MANIFEST}")
    result.manifest = str(manifest)
    if not manifest.resolve().is_relative_to(root):
        raise ValueError("manifest resolves outside the submission directory")
    payload = json.loads(manifest.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("expected a manifest object with schema_version 1")
    if manifest.name == EXERCISE_MANIFEST or (
        "controller_module" not in payload and payload.get("exercise") == "formula110-progression"
    ):
        levels = payload.get("levels")
        module = levels.get("3") if isinstance(levels, dict) else None
    else:
        module = payload.get("controller_module")
    if (
        not isinstance(module, str)
        or not module.startswith("controllers.")
        or not all(part.isidentifier() for part in module.split("."))
    ):
        raise ValueError("manifest must select a dotted controllers.* module")
    result.controller_module = module
    relative = Path(*module.split(".")).with_suffix(".py")
    for source_root in (root, root / "src"):
        candidate = (source_root / relative).resolve()
        if candidate.is_relative_to(root) and candidate.is_file():
            return candidate
    raise ValueError(f"{module} selected by {manifest.name}, but {relative} is missing or outside the submission")


def scan_submission(root: Path, weights: dict[str, int], *, float_digits: int = 8) -> SubmissionResult:
    result = SubmissionResult(root.name.removeprefix("submission_"), str(root))
    try:
        controller = resolve_controller(root.resolve(), result)
        result.controller_path = str(controller)
        with tokenize.open(controller) as stream:
            source = stream.read()
        result.car_name, result.findings = analyze_source(source, str(controller), float_digits=float_digits)
        counts = Counter(finding.rule for finding in result.findings)
        result.counts = {name: counts[name] for name in RULES}
        result.points = {name: min(counts[name], rule.max_occurrences) * weights[name] for name, rule in RULES.items()}
        result.score = sum(result.points.values())
        result.status = "ok"
    except (OSError, ValueError, SyntaxError, RecursionError) as error:
        result.error = f"{type(error).__name__}: {error}"
    return result


def scan_directory(directory: Path, weights: dict[str, int], *, float_digits: int = 8) -> list[SubmissionResult]:
    if not directory.is_dir():
        raise ValueError(f"submission directory does not exist: {directory}")
    roots = sorted(path for path in directory.iterdir() if path.is_dir() and path.name.startswith("submission_"))
    if not roots:
        raise ValueError(f"no submission_* directories found in {directory}")
    results = []
    for root in roots:
        if root.is_symlink():
            results.append(
                SubmissionResult(
                    root.name.removeprefix("submission_"), str(root), error="symlink submission directory skipped"
                )
            )
        else:
            results.append(scan_submission(root, weights, float_digits=float_digits))
    return sorted(results, key=lambda result: (result.score is None, -(result.score or 0), result.submission_id))


def read_submission_metadata(directory: Path) -> dict[str, object]:
    metadata = directory / "submission_metadata.yml"
    if not metadata.exists():
        print(f"No {metadata.name}; race metrics and submitter identities are unavailable.", file=sys.stderr)
        return {}
    try:
        payload = yaml.safe_load(metadata.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise ValueError(f"could not read {metadata}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{metadata}: expected a mapping of submission_ID records")
    return payload


def read_race_scores(directory: Path, *, metadata: dict[str, object] | None = None) -> dict[str, dict[str, float]]:
    """Join only current leaderboard values by ID; never substitute historical results."""
    payload = read_submission_metadata(directory) if metadata is None else metadata
    scores: dict[str, dict[str, float]] = {}
    columns = {title: column for column, title in RACE_METRICS.items()}
    for key, submission in payload.items():
        if not isinstance(key, str) or not key.startswith("submission_"):
            continue
        if not isinstance(submission, dict):
            raise ValueError(f"{key}: expected a submission record")
        results = submission.get(":results", submission.get("results"))
        if results is None:
            continue
        if not isinstance(results, dict):
            raise ValueError(f"{key}: expected a results mapping")
        entries = results.get("leaderboard")
        if entries is None:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"{key}: expected a leaderboard list")
        values: dict[str, float] = {}
        seen: set[str] = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"{key}: expected a leaderboard entry mapping")
            title = entry.get("name")
            if not isinstance(title, str) or title not in columns:
                continue
            column = columns[title]
            if column in seen:
                raise ValueError(f"{key}: duplicate leaderboard entry {title!r}")
            seen.add(column)
            value = entry.get("value")
            if isinstance(value, bool) or not isinstance(value, (int, float, str)):
                continue
            try:
                score = float(value)
            except (ValueError, OverflowError):
                continue
            if math.isfinite(score):
                values[column] = score
        scores[key.removeprefix("submission_")] = values
    return scores


def write_csv(stream: TextIO, results: list[SubmissionResult]) -> None:
    writer = csv.writer(stream)
    writer.writerow(["submission_id", "total_score", *RACE_METRICS, *EXPERIENCE_COLUMNS])
    for result in results:
        identifier = result.submission_id
        if identifier.lstrip().startswith(("=", "+", "-", "@")):
            identifier = "'" + identifier
        writer.writerow(
            [
                identifier,
                result.score,
                *(result.race_scores.get(column) for column in RACE_METRICS),
                EXPERIENCE_LEVELS[result.experience_rank] if result.experience_rank is not None else None,
                result.experience_rank,
                result.experience_match,
            ]
        )


def nonnegative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("submissions_dir", type=Path, help="extracted export containing submission_* directories")
    parser.add_argument("--csv", type=Path, help="CSV output path (default: stdout)")
    parser.add_argument(
        "--experience-csv",
        type=Path,
        help="Question 1.1 survey CSV (default: submission_metadata.csv beside the export directory, if present)",
    )
    parser.add_argument("--float-digits", type=int, default=8, help="minimum significant mantissa digits (default: 8)")
    parser.add_argument(
        "--weight",
        action="append",
        default=[],
        metavar="RULE=POINTS",
        help="override a rule's per-occurrence weight; 0 disables scoring (repeatable)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.float_digits < 2:
        parser.error("--float-digits must be at least 2")
    weights = {name: rule.weight for name, rule in RULES.items()}
    for override in args.weight:
        name, separator, value = override.partition("=")
        if not separator or name not in RULES:
            parser.error(f"--weight requires RULE=POINTS; rules: {', '.join(RULES)}")
        try:
            weights[name] = nonnegative_int(value)
        except (ValueError, argparse.ArgumentTypeError):
            parser.error(f"invalid nonnegative weight: {override}")
    directory = args.submissions_dir.expanduser().resolve()
    try:
        results = scan_directory(directory, weights, float_digits=args.float_digits)
        metadata = read_submission_metadata(directory)
        race_scores = read_race_scores(directory, metadata=metadata)
        for result in results:
            result.race_scores = race_scores.get(result.submission_id, {})
        experience_csv = (
            args.experience_csv.expanduser() if args.experience_csv else directory.parent / "submission_metadata.csv"
        )
        if args.experience_csv is not None or experience_csv.is_file():
            collate_experience(results, metadata, read_experience_survey(experience_csv))
            coverage = Counter(result.experience_match for result in results)
            print(
                f"Experience matches: {coverage['complete']} complete, {coverage['partial']} partial, "
                f"{coverage['unmatched']} unmatched. Survey: {experience_csv}",
                file=sys.stderr,
            )
        if args.csv:
            output = args.csv.expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("w", encoding="utf-8", newline="") as stream:
                write_csv(stream, results)
        else:
            write_csv(sys.stdout, results)
    except (OSError, ValueError, csv.Error) as error:
        parser.error(str(error))
    scanned = sum(result.status == "ok" for result in results)
    print(f"Scanned {scanned}/{len(results)} submissions; {len(results) - scanned} have blank scores.", file=sys.stderr)
    print(SCORE_NOTE, file=sys.stderr)
    for result in results:
        if result.error:
            print(f"UNSCANNED {result.submission_id}: {result.error}", file=sys.stderr)
    if args.csv:
        print(f"CSV: {args.csv}", file=sys.stderr)
    # Partial exports remain useful, with errors explicitly recorded in each report.
    return 0 if scanned else 1


if __name__ == "__main__":
    raise SystemExit(main())
