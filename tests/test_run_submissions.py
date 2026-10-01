from __future__ import annotations

import importlib.util
import json
import shlex
import sys
from pathlib import Path
from types import ModuleType

import pytest

from racing.game import cli
from racing.game.config import CameraView, HeadToHeadViewerConfig, HeatViewerConfig
from racing.race.runtime import DEFAULT_RACE_RANDOM_SEED
from racing.student.api import RobotSensors, load_student_controller


@pytest.fixture
def runner() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_submissions.py"
    spec = importlib.util.spec_from_file_location("run_submissions", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_submission(
    export_dir: Path,
    identifier: str,
    *,
    module: str = "controllers.level_3",
    legacy: bool = False,
    src_layout: bool = False,
    source: str = "raise RuntimeError('controller must not be imported during resolution')\n",
) -> Path:
    root = export_dir / f"submission_{identifier}"
    source_root = root / "src" if src_layout else root
    controller = source_root.joinpath(*module.split(".")).with_suffix(".py")
    controller.parent.mkdir(parents=True)
    controller.write_text(source, encoding="utf-8")
    manifest = "formula110-exercise-submission.json" if legacy else "formula110-submission.json"
    payload = (
        {"schema_version": 1, "levels": {"3": module}}
        if legacy
        else {
            "schema_version": 1,
            "controller_module": module,
        }
    )
    (root / manifest).write_text(json.dumps(payload), encoding="utf-8")
    return controller.resolve()


@pytest.fixture
def exported_pair(tmp_path: Path) -> tuple[Path, Path, Path]:
    export_dir = tmp_path / "export with spaces"
    challenger = make_submission(export_dir, "101")
    incumbent = make_submission(export_dir, "202")
    return export_dir, challenger, incumbent


def test_resolves_manifest_selected_custom_controller_before_legacy_level_three(
    runner: ModuleType,
    tmp_path: Path,
) -> None:
    chosen = make_submission(tmp_path, "101", module="controllers.custom_driver")
    (chosen.parent / "level_3.py").write_text("raise AssertionError('wrong controller')\n", encoding="utf-8")
    (chosen.parent.parent / "formula110-exercise-submission.json").write_text(
        json.dumps({"schema_version": 1, "levels": {"3": "controllers.level_3"}}),
        encoding="utf-8",
    )

    assert runner.resolve_controller(tmp_path, "101") == chosen


def test_resolves_legacy_level_three_in_src_layout(runner: ModuleType, tmp_path: Path) -> None:
    chosen = make_submission(tmp_path, "101", legacy=True, src_layout=True)

    assert runner.resolve_controller(tmp_path, "101") == chosen


@pytest.mark.parametrize(
    ("manifest_text", "message"),
    [
        (None, "expected formula110-submission.json or formula110-exercise-submission.json"),
        ("{", "could not read"),
        ('{"schema_version": 2, "controller_module": "controllers.level_3"}', "expected schema_version 1"),
        ('{"schema_version": 1, "controller_module": "../outside"}', r"expected a controllers\.\* module"),
        (
            '{"schema_version": 1, "controller_module": "controllers.missing"}',
            r"controllers\.missing selected by formula110-submission\.json.*is missing",
        ),
    ],
)
def test_bad_exports_explain_manifest_or_controller_problem(
    runner: ModuleType,
    tmp_path: Path,
    manifest_text: str | None,
    message: str,
) -> None:
    root = tmp_path / "submission_101"
    root.mkdir()
    if manifest_text is not None:
        (root / "formula110-submission.json").write_text(manifest_text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        runner.resolve_controller(tmp_path, "101")


@pytest.mark.parametrize(
    ("identifier", "message"),
    [("999", "submission 999 not found"), ("../101", "expected a numeric submission ID")],
)
def test_bad_submission_ids_report_cli_errors(
    runner: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    identifier: str,
    message: str,
) -> None:
    with pytest.raises(SystemExit) as error:
        runner.main([identifier, "202", "--export-dir", str(tmp_path)])

    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(("named", "fullscreen"), [(True, False), (False, False), (True, True)])
def test_watched_viewer_receives_team_names_and_window_settings(
    runner: ModuleType,
    exported_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    named: bool,
    fullscreen: bool,
) -> None:
    export_dir, challenger, incumbent = exported_pair
    configs: list[HeadToHeadViewerConfig] = []
    for path, name in [(challenger, "Blue Team"), (incumbent, "Red Team")]:
        path.write_text(
            "from racing import RobotCommand\n"
            + (f"RACING_NAME = {name!r}\n" if named else "")
            + "def control(sensors):\n    return RobotCommand()\n",
            encoding="utf-8",
        )

    class FakeApp:
        def run(self) -> None:
            pass

    def fake_viewer(config: HeadToHeadViewerConfig) -> FakeApp:
        configs.append(config)
        return FakeApp()

    monkeypatch.setattr(cli, "create_head_to_head_viewer_app", fake_viewer)

    runner.main(["submission_101", "202", "--export-dir", str(export_dir), *(["--fullscreen"] if fullscreen else [])])

    assert len(configs) == 1
    config = configs[0]
    assert config.challenger_name == ("Blue Team" if named else "#101")
    assert config.incumbent_name == ("Red Team" if named else "#202")
    assert config.camera_view == CameraView.CINEMATIC
    assert config.starting_grid
    assert config.grid_names == ("#101", "#202")
    assert config.random_seed == DEFAULT_RACE_RANDOM_SEED
    assert config.fullscreen is fullscreen


def test_headless_explicit_seed_and_h2h_options_are_forwarded(
    runner: ModuleType,
    exported_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir, _, _ = exported_pair
    dispatched: list[list[str]] = []
    monkeypatch.setattr(cli, "main", dispatched.append)

    runner.main(
        [
            "101",
            "202",
            "--export-dir",
            str(export_dir),
            "--headless",
            "--",
            "--seed",
            "7656",
            "--races",
            "3",
            "--round-seconds",
            "0.5",
            "--camera",
            "follow",
            "--no-damage",
            "--json",
        ]
    )

    args = cli.build_argument_parser().parse_args(dispatched[0])
    assert args.watch is False
    assert args.seed == 7656
    assert args.races == 3
    assert args.round_seconds == 0.5
    assert args.camera == "follow"
    assert args.no_damage is True
    assert args.json is True
    assert "Seed: 7656 (reuse with --seed 7656)" in capsys.readouterr().err


def test_random_seed_is_chosen_once_printed_and_dispatched(
    runner: ModuleType,
    exported_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir, _, _ = exported_pair
    dispatched: list[list[str]] = []
    random_calls: list[int] = []

    def choose_seed(upper_bound: int) -> int:
        random_calls.append(upper_bound)
        return 123456

    monkeypatch.setattr(runner.secrets, "randbelow", choose_seed)
    monkeypatch.setattr(cli, "main", dispatched.append)

    runner.main(["101", "202", "--export-dir", str(export_dir), "--seed", "random"])

    args = cli.build_argument_parser().parse_args(dispatched[0])
    assert len(random_calls) == 1
    assert args.seed == 123456
    assert "Seed: 123456 (reuse with --seed 123456)" in capsys.readouterr().err


def test_dry_run_prints_runnable_command_without_importing_controllers_or_dispatching(
    runner: ModuleType,
    exported_pair: tuple[Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    export_dir, challenger, incumbent = exported_pair

    def unexpected_dispatch(_args: list[str]) -> None:
        pytest.fail("dry-run dispatched the race")

    monkeypatch.setattr(cli, "main", unexpected_dispatch)

    runner.main(["101", "202", "--export-dir", str(export_dir), "--dry-run", "--seed", "42"])

    output = capsys.readouterr()
    command = shlex.split(output.out)
    assert command[:3] == [sys.executable, "-m", "racing"]
    args = cli.build_argument_parser().parse_args(command[3:])
    assert args.challenger_module == str(challenger)
    assert args.incumbent_module == str(incumbent)
    assert args.suppress_student_prints is True
    assert args.seed == 42
    assert args.watch is True


def test_submissions_with_identical_controller_names_load_independently(runner: ModuleType, tmp_path: Path) -> None:
    for identifier, throttle in [("101", 0.25), ("202", 0.75)]:
        make_submission(
            tmp_path,
            identifier,
            source=(
                "from racing import RobotCommand\n"
                "def control(sensors):\n"
                f"    return RobotCommand(throttle={throttle})\n"
            ),
        )

    challenger = load_student_controller(runner.resolve_controller(tmp_path, "101"))
    incumbent = load_student_controller(runner.resolve_controller(tmp_path, "202"))

    assert challenger(RobotSensors()).throttle == 0.25
    assert incumbent(RobotSensors()).throttle == 0.75


@pytest.mark.parametrize("headless", [False, True])
@pytest.mark.parametrize("entrant_count", [3, 4, 5, 6, 7, 8, 9, 10, 20])
def test_heat_ids_dispatch_with_ordered_submission_labels_and_shared_options(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    headless: bool,
    entrant_count: int,
) -> None:
    identifiers = [str(index * 101) for index in range(1, entrant_count + 1)]
    controllers = [make_submission(tmp_path, identifier) for identifier in identifiers]
    dispatched: list[list[str]] = []

    def fixed_random_seed(_bound: int) -> int:
        return 424242

    monkeypatch.setattr(cli, "main", dispatched.append)
    monkeypatch.setattr(runner.secrets, "randbelow", fixed_random_seed)

    runner.main(
        [
            *identifiers,
            "--export-dir",
            str(tmp_path),
            "--seed",
            "random",
            "--track-seed",
            "110",
            "--no-damage",
            "--races",
            "3",
            "--round-seconds",
            "0.5",
            *(["--headless", "--json"] if headless else ["--fullscreen"]),
        ]
    )

    assert len(dispatched) == 1
    args = cli.build_argument_parser().parse_args(dispatched[0])
    assert args.command == "heat"
    assert args.suppress_student_prints is True
    assert args.module == [str(path) for path in controllers]
    assert args.fallback_name == [f"#{identifier}" for identifier in identifiers]
    assert args.name is None
    assert args.watch is not headless
    assert args.camera == "cinematic"
    assert args.starting_grid is not headless
    assert args.fullscreen is not headless
    assert args.json is headless
    assert args.seed == 424242
    assert args.track_seed == 110
    assert args.no_damage is True
    assert args.races == 3
    assert args.round_seconds == 0.5
    output = capsys.readouterr()
    assert "Seed: 424242 (reuse with --seed 424242)" in output.err
    assert f"Entrant {entrant_count} #{identifiers[-1]}:" in output.err


@pytest.mark.parametrize("entrant_count", [1, 21])
def test_wrapper_requires_two_to_twenty_ids(
    runner: ModuleType,
    entrant_count: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identifiers = [str(index * 101) for index in range(1, entrant_count + 1)]
    with pytest.raises(SystemExit) as error:
        runner.main(identifiers)

    assert error.value.code == 2
    assert "provide two to twenty submission IDs" in capsys.readouterr().err


@pytest.mark.parametrize("entrant_count", [3, 4, 5, 6, 7, 8, 9, 10, 20])
def test_heat_rejects_duplicate_submission_ids(
    runner: ModuleType,
    capsys: pytest.CaptureFixture[str],
    entrant_count: int,
) -> None:
    identifiers = [str(index * 101) for index in range(1, entrant_count)]
    with pytest.raises(SystemExit) as error:
        runner.main([*identifiers, "submission_101"])

    assert error.value.code == 2
    assert f"{entrant_count} distinct submission IDs" in capsys.readouterr().err


@pytest.mark.parametrize("entrant_count", [3, 4, 5, 6, 7, 8, 9, 10, 20])
def test_heat_dry_run_resolves_modules_without_importing_them(
    runner: ModuleType,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    entrant_count: int,
) -> None:
    identifiers = [str(index * 101) for index in range(1, entrant_count + 1)]
    for identifier in identifiers:
        make_submission(tmp_path, identifier)

    def unexpected_dispatch(_args: list[str]) -> None:
        pytest.fail("dry-run dispatched the heat")

    monkeypatch.setattr(cli, "main", unexpected_dispatch)
    runner.main([*identifiers, "--export-dir", str(tmp_path), "--dry-run", "--seed", "42"])

    command = shlex.split(capsys.readouterr().out)
    assert command[:4] == [sys.executable, "-m", "racing", "heat"]
    args = cli.build_argument_parser().parse_args(command[3:])
    assert args.fallback_name == [f"#{identifier}" for identifier in identifiers]
    assert len(args.module) == entrant_count
    assert args.seed == 42


@pytest.mark.parametrize("allow_prints", [False, True])
def test_runner_suppresses_student_output_but_keeps_race_output(
    runner: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    allow_prints: bool,
) -> None:
    source = (
        "from racing import RobotCommand\n"
        "print('student import')\n"
        "def control(sensors):\n"
        "    print('student tick')\n"
        "    return RobotCommand(throttle=0.25)\n"
    )
    paths = [make_submission(tmp_path, identifier, source=source) for identifier in ("101", "202")]

    class FakeApp:
        def run(self) -> None:
            print("race output")

    def fake_viewer(config: HeadToHeadViewerConfig) -> FakeApp:
        for controller in (config.challenger_controller, config.incumbent_controller):
            assert controller is not None
            assert controller(RobotSensors()).throttle == 0.25
        return FakeApp()

    monkeypatch.setattr(cli, "create_head_to_head_viewer_app", fake_viewer)
    runner.main(["101", "202", "--export-dir", str(tmp_path), *(["--allow-student-prints"] if allow_prints else [])])

    output = capsys.readouterr()
    assert output.out.count("student import") == (2 if allow_prints else 0)
    assert output.out.count("student tick") == (2 if allow_prints else 0)
    assert "race output" in output.out
    assert "Seed:" in output.err
    assert all(path.read_text(encoding="utf-8") == source for path in paths)


@pytest.mark.parametrize("heat", [False, True])
def test_partners_title_and_car_names_reach_the_starting_grid(
    runner: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, heat: bool,
) -> None:
    for identifier in ("101", "202"):
        make_submission(
            tmp_path, identifier,
            source="from racing import RobotCommand\nRACING_NAME = 'Zoom'\n"
            "RACING_COLOR = '#ff8000'\ndef control(sensors):\n    return RobotCommand()\n",
        )
    (tmp_path / "submission_metadata.yml").write_text(
        "submission_101:\n  :submitters:\n    - :name: Ada Lovelace\n    - :name: Grace Brewster Hopper\n"
        "submission_202:\n  submitters:\n    - name: Jean-Luc Picard\n", encoding="utf-8",
    )
    configs: list[HeadToHeadViewerConfig | HeatViewerConfig] = []

    class FakeApp:
        def run(self) -> None:
            pass

    def viewer(config: HeadToHeadViewerConfig | HeatViewerConfig) -> FakeApp:
        configs.append(config)
        return FakeApp()

    monkeypatch.setattr(cli, "create_heat_viewer_app" if heat else "create_head_to_head_viewer_app", viewer)
    runner.main([
        "101", "202", "--export-dir", str(tmp_path), "--title", "The Grand Final",
        *(["--heat"] if heat else []),
    ])
    config = configs[0]
    assert config.title == "The Grand Final"
    assert config.grid_names == ("Ada L. & Grace H.", "Jean-Luc P.")
    assert config.starting_grid
    if isinstance(config, HeatViewerConfig):
        assert config.entrants[0].name == "Zoom"
        assert config.entrants[0].team_color == (1.0, 128 / 255, 0.0, 1.0)
    else:
        assert config.challenger_name == "Zoom"
        assert config.challenger_team_color == (1.0, 128 / 255, 0.0, 1.0)


def test_participant_metadata_fallbacks_and_single_names(runner: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "submission_metadata.yml").write_text(
        "submission_101:\n  :submitters:\n    - :name: '  Élodie   Durand '\n    - :name: Prince\n"
        "submission_202:\n  :submitters: []\n", encoding="utf-8",
    )
    assert runner.participant_names(tmp_path, ["101", "202", "303"]) == ["Élodie D. & Prince", "#202", "#303"]


def test_grid_names_validate_count_before_loading_controllers() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["heat", "--module", "missing", "--module", "missing", "--grid-name", "Ada L."])
    assert error.value.code == 2
