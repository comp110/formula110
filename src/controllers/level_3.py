"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

__author__: str = "Your PID goes here."
RACING_NAME: str = "Level 3"
RACING_COLOR: str = "#c1f0c1"


def control(sensors: RobotSensors) -> RobotCommand:
    """This demo is all gas, no steering."""
    steer: float
    throttle: float

    steer: float = (sensors.camera.heading_error_degrees / 180.0) * 10.0
    throttle = (20.0 - sensors.odometry.speed_mps) / 20.0

    return RobotCommand(throttle=throttle, steer=steer)
