from __future__ import annotations

import sys
from pathlib import Path

import pytest

from racing.race.head_to_head import controller_for_copy
from racing.student.api import RobotController, RobotSensors, load_student_controller


@pytest.mark.parametrize("package_init", [False, True])
@pytest.mark.parametrize("suppress_prints", [False, True])
def test_submission_helpers_remain_local_during_import_factory_and_ticks(
    tmp_path: Path, package_init: bool, suppress_prints: bool
) -> None:
    baseline = load_student_controller("controllers.crash_fast")(RobotSensors())
    original_packages = {
        name: module for name, module in sys.modules.items() if name == "controllers" or name.startswith("controllers.")
    }
    original_path = sys.path.copy()
    controllers: list[RobotController] = []
    # Cover both export layouts, with conflicting helper module names.
    for directory, throttle in (("first/controllers", 0.2), ("second/src/controllers", 0.7)):
        package = tmp_path / directory
        package.mkdir(parents=True)
        (package / "throttle.txt").write_text(str(throttle))
        (package / "helpers.py").write_text(
            "from dataclasses import dataclass\n"
            "from pathlib import Path\n"
            "@dataclass\n"
            "class Settings:\n"
            "    throttle: float\n"
            "settings = Settings(float(Path(__file__).with_name('throttle.txt').read_text()))\n"
        )
        if package_init:
            (package / "__init__.py").write_text("from .helpers import settings\n")
        (package / "nested").mkdir()
        (package / "nested" / "__init__.py").write_text("from ..helpers import settings\n")
        (package / "lazy.py").write_text(
            "from controllers.nested import settings\n"
            "from racing import RobotCommand\n"
            "def command(calls):\n"
            "    return RobotCommand(throttle=settings.throttle + calls / 100)\n"
        )
        module = package / "driver.py"
        module.write_text(
            "from controllers import helpers\n"
            "import controllers.helpers\n"
            "import controllers.helpers as aliased\n"
            "import helpers as bare_helper\n"
            "from .helpers import settings\n"
            "assert controllers.helpers is helpers is aliased is bare_helper\n"
            "assert settings is helpers.settings\n"
            "def create_controller():\n"
            "    from controllers.nested import settings as nested_settings\n"
            "    assert nested_settings is settings\n"
            "    calls = 0\n"
            "    def control(sensors):\n"
            "        from controllers.lazy import command\n"
            "        nonlocal calls\n"
            "        calls += 1\n"
            "        return command(calls)\n"
            "    return control\n"
        )
        controllers.append(load_student_controller(module, suppress_prints=suppress_prints))

    sensors = RobotSensors()
    assert [controller(sensors).throttle for controller in controllers] == pytest.approx([0.21, 0.71])
    copies = [controller_for_copy(controller) for controller in controllers]
    assert [controller(sensors).throttle for controller in copies] == pytest.approx([0.21, 0.71])
    assert [controller(sensors).throttle for controller in controllers] == pytest.approx([0.22, 0.72])
    assert sys.path == original_path
    assert {
        name: module for name, module in sys.modules.items() if name == "controllers" or name.startswith("controllers.")
    } == original_packages
    assert load_student_controller("controllers.crash_fast")(sensors) == baseline


def test_missing_submission_helper_does_not_fall_back_to_another_controllers_package(tmp_path: Path) -> None:
    load_student_controller("controllers.crash_fast")
    package = tmp_path / "controllers"
    package.mkdir()
    module = package / "driver.py"
    module.write_text("from controllers.crash_fast import control\n")

    with pytest.raises(ModuleNotFoundError, match="crash_fast"):
        load_student_controller(module)
