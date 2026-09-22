"""Self-driving robotic race car controller demo."""

from racing import RobotCommand, RobotSensors

__author__: str = "Your PID goes here."
RACING_NAME: str = "Level 1"
RACING_COLOR: str = "#c110c1"


def control(sensors: RobotSensors) -> RobotCommand:
    """This demo is all gas, no steering."""
    steer: float
    throttle: float

    if sensors.wall_lidar.front_left_m < 5.0:
        steer = 1.0
    else:
        if sensors.wall_lidar.front_right_m < 5.0:
            steer = -1.0
        else:
            steer = 0.0

    if sensors.odometry.speed_mps < 10.0:
        throttle = 1.0
    else:
        throttle = 0.0

    return RobotCommand(throttle=throttle, steer=steer)
