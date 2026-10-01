from __future__ import annotations

import sys
from pathlib import Path

import pytest

from racing.race.head_to_head import controller_for_copy
from racing.student.api import (
    CameraSensors,
    ContactSensors,
    LidarSensors,
    OdometrySensors,
    RobotCommand,
    RobotSensors,
    clamp_command,
    default_student_controller,
    load_student_controller,
    load_student_submission,
)


def test_clamp_command_bounds_normalized_actuators() -> None:
    command = clamp_command(RobotCommand(throttle=3.0, steer=-2.0))

    assert command == RobotCommand(throttle=1.0, steer=-1.0)


def test_robot_command_has_only_signed_throttle_and_steering_controls() -> None:
    command = RobotCommand(throttle=-0.75, steer=0.25)

    assert command.throttle == -0.75
    assert command.steer == 0.25
    assert not hasattr(command, "brake")


def test_clamp_command_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        clamp_command(RobotCommand(throttle=float("nan")))


def test_lidar_rejects_mismatched_angles_and_distances() -> None:
    with pytest.raises(ValueError, match="same length"):
        LidarSensors(angles_degrees=(0.0, 30.0), distances_m=(1.0,))


def test_default_student_controller_steers_toward_track_heading() -> None:
    sensors = RobotSensors(
        camera=CameraSensors(center_offset_m=0.5, heading_error_degrees=30.0),
        odometry=OdometrySensors(speed_mps=0.0),
    )

    command = default_student_controller(sensors)

    assert command.throttle > 0.0
    assert command.steer > 0.0


def test_default_student_controller_backs_out_of_contact() -> None:
    sensors = RobotSensors(
        contact=ContactSensors(wall=0.2),
        lidar=LidarSensors(distances_m=(5.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.5)),
    )

    command = default_student_controller(sensors)

    assert command.throttle < 0.0


def test_bundled_starter_controller_loads_from_top_level_controllers_package() -> None:
    controller = load_student_controller("controllers.crash_fast")

    command = controller(RobotSensors())

    assert isinstance(command, RobotCommand)


def test_crash_fast_controller_declares_mclaren_orange_car_color() -> None:
    submission = load_student_submission("controllers.crash_fast")

    assert submission.display_name == "Crash Fast"
    assert submission.car_color == (1.0, 0x87 / 255, 0.0, 1.0)


def test_load_student_controller_from_file(tmp_path: Path) -> None:
    module_path = tmp_path / "student_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "def drive(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.25, steer=0.5)\n",
        encoding="utf-8",
    )

    controller = load_student_controller(module_path, function_name="drive")

    assert controller(RobotSensors()) == RobotCommand(throttle=0.25, steer=0.5)


@pytest.mark.parametrize("import_by_name", [False, True])
def test_suppress_prints_skips_arguments_and_preserves_controller_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    import_by_name: bool,
) -> None:
    module_path = tmp_path / "quiet_student_driver.py"
    source = (
        "from racing import RobotCommand\n"
        "print('import', 1 / 0)\n"
        "def create_controller():\n"
        "    print('factory', 1 / 0)\n"
        "    def control(sensors):\n"
        "        if sensors.tick == 0:\n"
        "            print(\n"
        "                'debug', 1 / 0,\n"
        "            )\n"
        "        return RobotCommand(throttle=0.25, steer=0.5)\n"
        "    return control\n"
    )
    module_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    controller = load_student_controller(module_path.stem if import_by_name else module_path, suppress_prints=True)

    assert controller(RobotSensors()) == RobotCommand(throttle=0.25, steer=0.5)
    assert controller_for_copy(controller)(RobotSensors()) == RobotCommand(throttle=0.25, steer=0.5)
    assert capsys.readouterr().out == ""
    assert module_path.read_text(encoding="utf-8") == source


def test_suppression_ignores_cached_bytecode_and_does_not_affect_normal_loading(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module_path = tmp_path / "noisy_student_driver.py"
    module_path.write_text(
        "from racing import RobotCommand\n"
        "print('import')\n"
        "def control(sensors):\n"
        "    print('tick')\n"
        "    return RobotCommand()\n",
        encoding="utf-8",
    )
    for suppress_prints in (False, True, False):
        controller = load_student_controller(module_path, suppress_prints=suppress_prints)
        assert controller(RobotSensors()) == RobotCommand()
        print("race output")
        expected = "race output\n" if suppress_prints else "import\ntick\nrace output\n"
        assert capsys.readouterr().out == expected


def test_controller_factory_creates_independent_state_for_each_car(tmp_path: Path) -> None:
    module_path = tmp_path / "stateful_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "class Controller:\n"
        "    def __init__(self) -> None:\n"
        "        self.calls = 0\n"
        "\n"
        "    def __call__(self, sensors: RobotSensors) -> RobotCommand:\n"
        "        self.calls += 1\n"
        "        return RobotCommand(throttle=float(self.calls) / 10.0)\n"
        "\n"
        "def create_controller() -> Controller:\n"
        "    return Controller()\n",
        encoding="utf-8",
    )

    prototype = load_student_controller(module_path)
    first_car = controller_for_copy(prototype)
    second_car = controller_for_copy(prototype)

    assert first_car(RobotSensors()).throttle == 0.1
    assert first_car(RobotSensors()).throttle == 0.2
    assert second_car(RobotSensors()).throttle == 0.1


def test_controller_factory_must_return_callable(tmp_path: Path) -> None:
    module_path = tmp_path / "invalid_factory.py"
    module_path.write_text("def create_controller():\n    return 123\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must return a callable"):
        load_student_controller(module_path)


def test_load_student_submission_returns_none_for_missing_display_name(tmp_path: Path) -> None:
    module_path = tmp_path / "student_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.1)\n",
        encoding="utf-8",
    )

    submission = load_student_submission(module_path)

    assert submission.display_name is None
    assert submission.car_color is None
    assert submission.controller(RobotSensors()) == RobotCommand(throttle=0.1)


def test_load_student_submission_reads_optional_display_name_and_car_color(tmp_path: Path) -> None:
    module_path = tmp_path / "named_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_NAME = '  Blue Steel  '\n"
        "RACING_COLOR = '#ff8000'\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )

    submission = load_student_submission(module_path)

    assert submission.display_name == "Blue Steel"
    assert submission.car_color == (1.0, 128 / 255, 0.0, 1.0)
    assert submission.controller(RobotSensors()) == RobotCommand(throttle=0.4)


def test_load_student_submission_reads_tuple_car_color(tmp_path: Path) -> None:
    module_path = tmp_path / "tuple_color_driver.py"
    module_path.write_text(
        "from racing import RobotCommand, RobotSensors\n"
        "\n"
        "RACING_COLOR = (0.1, 0.2, 0.3)\n"
        "\n"
        "def control(sensors: RobotSensors) -> RobotCommand:\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )

    submission = load_student_submission(module_path)

    assert submission.car_color == (0.1, 0.2, 0.3, 1.0)


def test_load_student_submission_rejects_blank_display_name(tmp_path: Path) -> None:
    module_path = tmp_path / "blank_named_driver.py"
    module_path.write_text(
        "RACING_NAME = '  '\n"
        "\n"
        "def control(sensors):\n"
        "    from racing import RobotCommand\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="RACING_NAME"):
        load_student_submission(module_path)


def test_load_student_submission_rejects_invalid_car_color(tmp_path: Path) -> None:
    module_path = tmp_path / "bad_color_driver.py"
    module_path.write_text(
        "RACING_COLOR = (1.2, 0.0, 0.0)\n"
        "\n"
        "def control(sensors):\n"
        "    from racing import RobotCommand\n"
        "    return RobotCommand(throttle=0.4)\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="RACING_COLOR"):
        load_student_submission(module_path)


def test_load_student_controller_requires_robot_command_return(tmp_path: Path) -> None:
    module_path = tmp_path / "bad_driver.py"
    module_path.write_text("def control(sensors):\n    return 123\n", encoding="utf-8")

    controller = load_student_controller(module_path)

    with pytest.raises(TypeError, match="RobotCommand"):
        controller(RobotSensors())
