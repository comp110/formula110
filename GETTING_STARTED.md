# Getting Started with Formula 110

This guide takes you from a fresh checkout to driving the simulator and writing
your first controller. For the complete controller contract, packaging rules,
experiment APIs, and evaluation guidance, continue with the
[project README](README.md). Every available input is documented in the
[sensor reference](SENSORS.md).

Formula 110 is a Python 3.11 project managed with `uv`. Run the commands below
from the project folder.

## 1. Check the host requirements

You need:

- `uv`, the Python project and package manager
- A terminal
- This repository on your machine

Check that `uv` is installed:

```bash
uv --version
```

You do not need to install Python 3.11 yourself. The project requires Python
`>=3.11,<3.12`, and `uv` can download a matching version. If `uv` is missing,
follow the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/).

## 2. Install Python and the project

Check the repository's local Python pin:

```bash
cat .python-version
```

It should say `3.11`. Then install Python and the project dependencies:

```bash
uv sync --managed-python
```

This creates or updates `.venv/`. You do not need to activate it; `uv run ...`
uses the project environment automatically.

Verify the installation:

```bash
uv run python --version
uv run racing --help
```

The Python command should report `3.11.x`.

## 3. Drive the car

Start manual keyboard control:

```bash
uv run racing --seed 110
```

The seed chooses a random position on the track. Reusing the same seed starts
at the same position, which is useful when comparing driving strategies.

Keyboard controls:

- Up arrow: request forward throttle
- Down arrow: request reverse; the car brakes first while moving forward
- Left arrow: steer left
- Right arrow: steer right
- `V`: cycle camera views
- `Q` / `W` / `E` / `R` / `T` / `Y`: top-down / three-quarter / drone / helicopter / cinematic / close follow
- `M`: mute or unmute audio

Things to notice:

- Hitting a wall damages the car.
- The damage bar runs from `0.0` to `1.0`; at `1.0`, the car is eliminated.
- Different camera views make different driving problems easier to see.
- Signed throttle handles both directions. An opposite-direction request first
  brakes the moving car and changes direction only after it slows down.

Start with audio muted if desired:

```bash
uv run racing --seed 110 --muted
```

You can record manual driving for later analysis or imitation learning:

```bash
uv run racing --seed 110 \
  --record-human artifacts/human-driving.jsonl
```

See [Capturing human demonstrations](README.md#capturing-human-demonstrations)
for the JSONL schema and data boundaries.

## 4. Run the starter controller

The intentionally tiny controller in `src/controllers/crash_fast.py` always
requests full forward throttle and never steers:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Crash Fast"


def control(sensors: RobotSensors) -> RobotCommand:
    return RobotCommand(throttle=1.0, steer=0.0)
```

Run it from the same seeded start:

```bash
uv run racing \
  --seed 110 \
  --student-module controllers.crash_fast
```

Watch what happens when it reaches a wall. Avoiding that first crash is a useful
initial controller goal.

## 5. Create your own controller

Copy the starter:

```bash
cp src/controllers/crash_fast.py src/controllers/avoid_walls.py
```

Give the copy a name, color, and slower initial command:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME: str = "Avoid Walls"
RACING_COLOR: str = "#1EB4FF"


def control(sensors: RobotSensors) -> RobotCommand:
    throttle: float = 0.4
    steer: float = 0.0
    return RobotCommand(throttle=throttle, steer=steer)
```

Run it:

```bash
uv run racing \
  --seed 110 \
  --student-module controllers.avoid_walls
```

The simulator calls `control` 60 times per simulated second. Each call receives
the latest `RobotSensors` and must return a `RobotCommand` with:

- `throttle` from `-1.0` (reverse request) to `1.0` (forward request)
- `steer` from `-1.0` (left) to `1.0` (right)

## 6. Try a first sensor strategy

Begin with a small subset of the available observations:

- `sensors.wall_lidar`: distances to track barriers
- `sensors.odometry.speed_mps`: signed forward speed in meters per second
- `sensors.camera.center_offset_m`: direction and distance to track center
- `sensors.camera.heading_error_degrees`: turn toward the local track heading

Useful wall readings include:

- `sensors.wall_lidar.front_m`
- `sensors.wall_lidar.front_left_m`
- `sensors.wall_lidar.front_right_m`
- `sensors.wall_lidar.left_m`
- `sensors.wall_lidar.right_m`

A reasonable first goal is:

- Steer back when the car moves away from the centerline.
- Steer away from a nearby side wall.
- Request negative throttle and turn toward open space when a wall is close in
  front.
- Use signed speed to avoid entering turns too quickly.

Tune one behavior at a time and rerun it with the same seed. Once the controller
can survive part of the track, continue with
[Comparing controllers](README.md#comparing-controllers) and the complete
[sensor reference](SENSORS.md).
