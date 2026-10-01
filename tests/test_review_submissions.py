from __future__ import annotations

import csv
import importlib.util
import io
import json
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

import pytest
import yaml


@pytest.fixture(scope="module")
def reviewer() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "review_submissions.py"
    spec = importlib.util.spec_from_file_location("review_submissions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_submission(
    directory: Path,
    identifier: str,
    source: str,
    *,
    module: str = "controllers.chosen",
    legacy: bool = False,
    src_layout: bool = False,
) -> Path:
    root = directory / f"submission_{identifier}"
    controller = (root / "src" if src_layout else root).joinpath(*module.split(".")).with_suffix(".py")
    controller.parent.mkdir(parents=True)
    controller.write_text(source, encoding="utf-8")
    manifest = "formula110-exercise-submission.json" if legacy else "formula110-submission.json"
    payload = (
        {"schema_version": 1, "exercise": "formula110-progression", "levels": {"3": module}}
        if legacy
        else {"schema_version": 1, "controller_module": module}
    )
    (root / manifest).write_text(json.dumps(payload), encoding="utf-8")
    return root


def counts(reviewer: ModuleType, source: str) -> Counter[str]:
    _, findings = reviewer.analyze_source(source)
    return Counter(finding.rule for finding in findings)


def test_course_style_and_global_state_are_not_flagged(reviewer: ModuleType) -> None:
    source = """
RACING_NAME: str = "Level 3"
last: float = 0.0
history: list[float] = []
def control(sensors: object) -> float:
    global last
    steer: float
    throttle: float = 1.0
    i: int = 0
    while i < 3:
        i += 1
    steer = last
    last = throttle
    history.append(steer)
    return steer
"""
    assert counts(reviewer, source) == {}


def test_only_unannotated_local_bindings_are_counted_once_per_scope(reviewer: ModuleType) -> None:
    source = """
def control(sensors):
    steer = 0.0
    steer += 1.0
    throttle: float
    throttle = 1.0
    sensors = None
    left, right = [1, 2]
    def helper():
        steer: float = 0.0
        other = 1
    return steer
"""
    assert counts(reviewer, source) == {"unannotated_local": 4}


def test_annotation_in_nested_function_does_not_cover_outer_local(reviewer: ModuleType) -> None:
    source = "def control():\n    x = 1\n    def helper():\n        x: int = 2\n"
    assert counts(reviewer, source) == {"unannotated_local": 1}


@pytest.mark.parametrize(
    ("literal", "expected"),
    [
        ("0.123456789", 1),
        ("1.2345678e-12", 1),
        ("123_456.789", 1),
        ("-0.123456789", 1),
        ("0.00000000012345678", 1),
        ("0.1000000000000", 0),
        ("1e-12", 0),
        ("100000000.0", 0),
        ("0.1234567", 0),
        ("123456789", 0),
    ],
)
def test_float_precision_uses_significant_mantissa_digits(reviewer: ModuleType, literal: str, expected: int) -> None:
    assert counts(reviewer, f"GAIN: float = {literal}")["precise_float"] == expected


def test_comments_and_strings_are_not_code(reviewer: ModuleType) -> None:
    source = """
# for, nonlocal, callable(), zip(), tuple(), class, 0.123456789
description = "nonlocal x; zip(x); class Controller: 0.123456789"
"""
    assert counts(reviewer, source) == {}


@pytest.mark.parametrize(
    ("name", "expected"), [("Car v2", 1), ("Car version 3", 1), ("Car 2.1", 1), ("Level 3", 0), ("F1 Racer", 0)]
)
def test_versioned_car_names(reviewer: ModuleType, name: str, expected: int) -> None:
    assert counts(reviewer, f"RACING_NAME: str = {name!r}")["versioned_name"] == expected


def test_tuple_types_values_and_calls_not_other_generic_parameters_or_unpacking(reviewer: ModuleType) -> None:
    source = """
from typing import Tuple as Pair
mapping: dict[str, float] = {}
values: Pair[float, float] = (1.0, 2.0)
other: tuple[int, int] = tuple([1, 2])
left, right = [1, 2]
"""
    assert counts(reviewer, source) == {"tuple": 4}


def test_callable_zip_aliases_and_comprehensions(reviewer: ModuleType) -> None:
    source = """
from collections.abc import Callable as Fn
from builtins import zip as pair
import typing as t
fn: Fn[[int], int]
other: t.Callable[..., float]
callable(fn)
for x in pair([1], [2]):
    pass
values = [x for x in range(3) if x > 1]
"""
    assert counts(reviewer, source) == {"callable": 3, "zip": 1, "for_loop": 2}


@pytest.mark.parametrize(
    "source",
    [
        "def control(sensors, state={}):\n    return state\n",
        "def control(sensors, *, state=list()):\n    return state\n",
        "def control():\n    control.last = 1\n",
        "def control():\n    setattr(control, 'last', 1)\n",
        "def control():\n    control.history.append(1)\n",
        "class Driver:\n    def __call__(self):\n        self.last = 1\n",
        "def create_controller():\n    history: list[int] = []\n"
        "    def control():\n        history.append(1)\n    return control\n",
        "def create_controller():\n    state: dict[str, int] = {}\n"
        "    def control():\n        state['last'] = 1\n    return control\n",
        "from functools import lru_cache as memo\n@memo(maxsize=1)\ndef helper():\n    return 1\n",
    ],
)
def test_persistent_state_patterns(reviewer: ModuleType, source: str) -> None:
    assert counts(reviewer, source)["persistent_state"] == 1


def test_nonlocal_is_separate_from_unannotated_local_and_other_state(reviewer: ModuleType) -> None:
    source = """
def create_controller():
    last: float = 0.0
    def control():
        nonlocal last
        last += 1.0
        return last
    return control
"""
    assert counts(reviewer, source) == {"nonlocal": 1}


def test_class_and_method_locals(reviewer: ModuleType) -> None:
    source = "class Driver:\n    last = 0\n    def control(self):\n        steer = 1.0\n        self.last = steer\n"
    assert counts(reviewer, source) == {"class": 1, "unannotated_local": 1, "persistent_state": 1}


def test_only_selected_file_is_scanned_without_executing_it(reviewer: ModuleType, tmp_path: Path) -> None:
    marker = tmp_path / "must_not_exist"
    source = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    root = make_submission(tmp_path, "101", source)
    (root / "controllers" / "unselected.py").write_text("class Advanced:\n    pass\n", encoding="utf-8")
    (root / "formula110-exercise-submission.json").write_text(
        json.dumps({"schema_version": 1, "levels": {"3": "controllers.unselected"}}), encoding="utf-8"
    )
    results = reviewer.scan_directory(tmp_path, {name: rule.weight for name, rule in reviewer.RULES.items()})
    assert results[0].score == 0
    assert results[0].controller_module == "controllers.chosen"
    assert not marker.exists()


@pytest.mark.parametrize("renamed_manifest", [False, True])
def test_legacy_level_three_src_layout(reviewer: ModuleType, tmp_path: Path, renamed_manifest: bool) -> None:
    root = make_submission(tmp_path, "202", "class Driver: pass\n", legacy=True, src_layout=True)
    if renamed_manifest:
        (root / "formula110-exercise-submission.json").rename(root / "formula110-submission.json")
    results = reviewer.scan_directory(tmp_path, {name: rule.weight for name, rule in reviewer.RULES.items()})
    assert results[0].score == 6
    assert results[0].controller_path.endswith("/src/controllers/chosen.py")


def test_csv_lists_every_submission_descending_with_capped_weights(
    reviewer: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_submission(tmp_path, "101", "def control():\n" + "".join(f"    x{i} = {i}\n" for i in range(30)))
    make_submission(tmp_path, "202", "class Driver: pass\n")
    make_submission(tmp_path, "303", "GAIN: float = 1.0\n")
    assert reviewer.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert list(csv.reader(io.StringIO(captured.out))) == [
        [
            "submission_id",
            "total_score",
            "all_spawns_no_crumbs_laps",
            "clean_lap_seconds",
            *reviewer.EXPERIENCE_COLUMNS,
        ],
        ["202", "6", "", "", "", "", ""],
        ["101", "0", "", "", "", "", ""],
        ["303", "0", "", "", "", "", ""],
    ]
    assert "Scanned 3/3" in captured.err

    output = tmp_path / "reports" / "scores.csv"
    assert (
        reviewer.main([str(tmp_path), "--csv", str(output), "--weight", "class=20", "--weight", "unannotated_local=1"])
        == 0
    )
    assert capsys.readouterr().out == ""
    assert list(csv.reader(io.StringIO(output.read_text()))) == [
        [
            "submission_id",
            "total_score",
            "all_spawns_no_crumbs_laps",
            "clean_lap_seconds",
            *reviewer.EXPERIENCE_COLUMNS,
        ],
        ["202", "20", "", "", "", "", ""],
        ["101", "10", "", "", "", "", ""],
        ["303", "0", "", "", "", "", ""],
    ]


def test_errors_stay_in_csv_without_a_score_and_do_not_stop_scan(
    reviewer: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_submission(tmp_path, "101", "def invalid(:\n")
    root = make_submission(tmp_path, "202", "")
    (root / "formula110-submission.json").write_text("{", encoding="utf-8")
    make_submission(tmp_path, "303", "x: int = 1\n")
    (tmp_path / "submission_404").mkdir()
    assert reviewer.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert list(csv.reader(io.StringIO(captured.out))) == [
        [
            "submission_id",
            "total_score",
            "all_spawns_no_crumbs_laps",
            "clean_lap_seconds",
            *reviewer.EXPERIENCE_COLUMNS,
        ],
        ["303", "0", "", "", "", "", ""],
        ["101", "", "", "", "", "", ""],
        ["202", "", "", "", "", "", ""],
        ["404", "", "", "", "", "", ""],
    ]
    assert "SyntaxError" in captured.err
    assert "JSONDecodeError" in captured.err
    assert "UNSCANNED 404" in captured.err


def test_outside_controller_symlink_is_not_read(reviewer: ModuleType, tmp_path: Path) -> None:
    root = make_submission(tmp_path, "101", "")
    external = tmp_path / "external.py"
    external.write_text("class External: pass\n", encoding="utf-8")
    chosen = root / "controllers" / "chosen.py"
    chosen.unlink()
    chosen.symlink_to(external)
    result = reviewer.scan_submission(root, {name: rule.weight for name, rule in reviewer.RULES.items()})
    assert result.score is None
    assert "outside" in result.error


@pytest.mark.parametrize("extra", [["--weight", "unknown=1"], ["--weight", "class=-1"], ["--float-digits", "1"]])
def test_invalid_scoring_options(reviewer: ModuleType, tmp_path: Path, extra: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        reviewer.main([str(tmp_path), *extra])
    assert error.value.code == 2


def test_csv_joins_current_race_metrics_by_id_and_keeps_review_order(
    reviewer: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    make_submission(tmp_path, "101", "class Driver: pass\n")
    make_submission(tmp_path, "202", "GAIN: float = 1.0\n")
    make_submission(tmp_path, "303", "def invalid(:\n")
    metadata = {
        "submission_202": {
            "results": {"leaderboard": [{"name": "All Spawns, No Crumbs (Laps)", "value": 4.5}]},
            "history": [{"results": {"leaderboard": [{"name": "Clock It (s)", "value": 1.0}]}}],
        },
        "submission_101": {
            ":results": {
                "leaderboard": [
                    {"name": "Clock It (s)", "value": "12.34", "order": "asc"},
                    {"name": "All Spawns, No Crumbs (Laps)", "value": 2.5678},
                ]
            },
            ":history": [{":results": {"leaderboard": [{"name": "Clock It (s)", "value": 99.0}]}}],
        },
        "submission_303": {":results": {"leaderboard": [{"name": "All Spawns, No Crumbs (Laps)", "value": 0.0}]}},
        "submission_404": {":results": {"leaderboard": [{"name": "All Spawns, No Crumbs (Laps)", "value": 9.0}]}},
    }
    (tmp_path / "submission_metadata.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    assert reviewer.main([str(tmp_path)]) == 0
    captured = capsys.readouterr()
    assert list(csv.reader(io.StringIO(captured.out))) == [
        [
            "submission_id",
            "total_score",
            "all_spawns_no_crumbs_laps",
            "clean_lap_seconds",
            *reviewer.EXPERIENCE_COLUMNS,
        ],
        ["101", "6", "2.5678", "12.34", "", "", ""],
        ["202", "0", "4.5", "", "", "", ""],
        ["303", "", "0.0", "", "", "", ""],
    ]


def test_absent_invalid_and_nonfinite_metrics_stay_blank(reviewer: ModuleType, tmp_path: Path) -> None:
    metadata = {
        "submission_101": {
            ":results": None,
            ":history": [{":results": {"leaderboard": [{"name": "Clock It (s)", "value": 1.0}]}}],
        },
        "submission_202": {":results": {"leaderboard": [{"name": "Clock It (s)", "value": True}]}},
        "submission_303": {":results": {"leaderboard": [{"name": "Clock It (s)", "value": float("nan")}]}},
        "submission_404": {":results": {"leaderboard": [{"name": "Clock It (s)", "value": "N/A"}]}},
        "submission_505": {":results": {"leaderboard": [{"name": "Clock It (s)", "value": "inf"}]}},
        "submission_606": {":results": {"leaderboard": [{"name": "Clock It (s)", "value": "-"}]}},
    }
    (tmp_path / "submission_metadata.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    assert reviewer.read_race_scores(tmp_path) == {"202": {}, "303": {}, "404": {}, "505": {}, "606": {}}


@pytest.mark.parametrize(
    "metadata",
    [
        "[invalid",
        "- not a mapping\n",
        "submission_101:\n  :results:\n    leaderboard: invalid\n",
        "submission_101:\n  :results:\n    leaderboard:\n"
        "    - {name: 'Clock It (s)', value: 1.0}\n"
        "    - {name: 'Clock It (s)', value: 2.0}\n",
    ],
)
def test_malformed_metadata_does_not_overwrite_existing_csv(
    reviewer: ModuleType, tmp_path: Path, metadata: str
) -> None:
    make_submission(tmp_path, "101", "")
    (tmp_path / "submission_metadata.yml").write_text(metadata, encoding="utf-8")
    output = tmp_path / "scores.csv"
    output.write_text("previous report\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        reviewer.main([str(tmp_path), "--csv", str(output)])
    assert error.value.code == 2
    assert output.read_text() == "previous report\n"


def write_survey(path: Path, responses: list[tuple[str, str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Student ID", "Email", "Question 1.1 Response", "Submission ID", "Question 1.1 Score"])
        # These are survey submission IDs and grading points, not join keys or experience ranks.
        writer.writerows((*row, "404", "0.0") for row in responses)


@pytest.mark.parametrize("explicit_path", [False, True])
def test_experience_joins_submitters_takes_team_max_and_reports_missing_matches(
    reviewer: ModuleType, tmp_path: Path, capsys: pytest.CaptureFixture[str], explicit_path: bool
) -> None:
    export = tmp_path / "export"
    for identifier in ("101", "202", "303", "404", "505"):
        make_submission(export, identifier, "class Driver: pass\n" if identifier == "101" else "")
    metadata = {
        "submission_101": {
            ":submitters": [{":sid": "001"}, {":sid": "002", ":email": " TWO@example.com "}],
            ":results": {"leaderboard": [{"name": "Clock It (s)", "value": 12.5}]},
        },
        "submission_202": {"submitters": [{"sid": "001"}]},
        "submission_303": {":submitters": [{":sid": "outdated-id", ":email": "three@EXAMPLE.com"}]},
        "submission_404": {":submitters": [{":sid": "unmatched"}]},
        "submission_505": {":submitters": [{":sid": "002"}, {":sid": "004"}]},
    }
    (export / "submission_metadata.yml").write_text(yaml.safe_dump(metadata), encoding="utf-8")
    survey = tmp_path / ("chosen-survey.csv" if explicit_path else "submission_metadata.csv")
    write_survey(
        survey,
        [
            ("001", "one@example.com", "None"),
            ("002", "two@example.com", reviewer.EXPERIENCE_LEVELS[2]),
            ("003", "three@example.com", reviewer.EXPERIENCE_LEVELS[3]),
            ("004", "four@example.com", ""),
        ],
    )
    args = [str(export), *(["--experience-csv", str(survey)] if explicit_path else [])]
    assert reviewer.main(args) == 0
    captured = capsys.readouterr()
    rows = list(csv.DictReader(io.StringIO(captured.out)))
    assert [row["submission_id"] for row in rows] == ["101", "202", "303", "404", "505"]
    assert [row["total_score"] for row in rows] == ["6", "0", "0", "0", "0"]
    assert rows[0]["clean_lap_seconds"] == "12.5"
    assert [row["experience_rank"] for row in rows] == ["2", "0", "3", "", "2"]
    assert [row["experience_level"] for row in rows] == [
        reviewer.EXPERIENCE_LEVELS[2],
        "None",
        reviewer.EXPERIENCE_LEVELS[3],
        "",
        reviewer.EXPERIENCE_LEVELS[2],
    ]
    assert [row["experience_match"] for row in rows] == ["complete", "complete", "complete", "unmatched", "partial"]
    assert "Experience matches: 3 complete, 1 partial, 1 unmatched" in captured.err


def test_identity_conflicts_and_ambiguous_duplicates_remain_unmatched(reviewer: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "survey.csv"
    write_survey(
        path,
        [
            ("001", "one@example.com", "None"),
            ("002", "two@example.com", reviewer.EXPERIENCE_LEVELS[3]),
            ("003", "shared@example.com", reviewer.EXPERIENCE_LEVELS[1]),
            ("004", "shared@example.com", reviewer.EXPERIENCE_LEVELS[2]),
        ],
    )
    survey = reviewer.read_experience_survey(path)
    assert survey.match({"sid": "001", "email": "two@example.com"}) is None
    assert survey.match({"email": "shared@example.com"}) is None
    assert survey.match({"sid": "003", "email": "shared@example.com"}) == 1
    assert survey.match({"sid": "004", "email": "shared@example.com"}) == 2


def test_survey_none_is_distinct_from_blank_and_omitted_responses(reviewer: ModuleType, tmp_path: Path) -> None:
    path = tmp_path / "survey.csv"
    path.write_text(
        "Student ID,Question 1.1 Response\n001,None\n002,\n003\n004,  a LITTLE experience  \n",
        encoding="utf-8",
    )
    survey = reviewer.read_experience_survey(path)
    assert survey.match({"sid": "001"}) == 0
    assert survey.match({"sid": "002"}) is None
    assert survey.match({"sid": "003"}) is None
    assert survey.match({"sid": "004"}) == 1
    assert survey.match({"sid": "1"}) is None  # Preserve leading zeros; don't conflate IDs.


def test_unscannable_controller_can_still_have_experience(reviewer: ModuleType, tmp_path: Path) -> None:
    root = make_submission(tmp_path, "101", "def invalid(:\n")
    result = reviewer.scan_submission(root, {name: rule.weight for name, rule in reviewer.RULES.items()})
    path = tmp_path / "survey.csv"
    write_survey(path, [("001", "one@example.com", "None")])
    reviewer.collate_experience(
        [result], {"submission_101": {":submitters": [{":sid": "001"}]}}, reviewer.read_experience_survey(path)
    )
    assert result.score is None and result.experience_rank == 0 and result.experience_match == "complete"


@pytest.mark.parametrize(
    "body",
    [
        "Student ID,Email\n001,one@example.com\n",
        "Submission ID,Question 1.1 Response\n101,None\n",
        "Student ID,Question 1.1 Response,Question 1.1 Response\n001,None,None\n",
        "Student ID,Question 1.1 Response\n001,unrecognized response\n",
    ],
)
def test_bad_survey_does_not_overwrite_report(reviewer: ModuleType, tmp_path: Path, body: str) -> None:
    make_submission(tmp_path, "101", "")
    survey = tmp_path / "survey.csv"
    survey.write_text(body, encoding="utf-8")
    output = tmp_path / "scores.csv"
    output.write_text("previous report\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        reviewer.main([str(tmp_path), "--experience-csv", str(survey), "--csv", str(output)])
    assert error.value.code == 2 and output.read_text() == "previous report\n"
