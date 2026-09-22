"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

__author__: str = "Your PID goes here."
RACING_NAME: str = "Crash Fast"
RACING_COLOR: str = "#c110c1"


def control(sensors: RobotSensors) -> RobotCommand:
    """This demo is all gas, no steering."""
    return RobotCommand(throttle=1.0, steer=0.0)
