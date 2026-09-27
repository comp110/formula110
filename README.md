---
author: Kris Jordan
---

# Formula 110

Formula 110 is a deterministic racing simulator for designing, testing, and
comparing autonomous car controllers. At each 60 Hz simulation tick, your
controller maps a public sensor snapshot to signed throttle and steering
commands. The controller can be rules, an MLP, an evolved policy, a search
procedure, or another method.

Start here:

- [Getting Started](GETTING_STARTED.md): installation, manual driving, and your
  first controller
- [Sensor Reference](SENSORS.md): every input field, type, unit, range, and
  sentinel value
- [Autograder Guide](autograder/README.md): building and operating the isolated
  Gradescope evaluator

## Runtime contract

A controller receives an immutable `RobotSensors` snapshot and returns one
`RobotCommand`:

```python
from racing import RobotCommand, RobotSensors


def control(sensors: RobotSensors) -> RobotCommand:
    return RobotCommand(throttle=0.2, steer=0.0)
```

Command ranges are:

| Field | Range | Meaning |
| --- | --- | --- |
| `throttle` | `-1.0` to `1.0` | Reverse to forward drive request |
| `steer` | `-1.0` to `1.0` | Full left to full right |

When signed throttle opposes the car's current motion, the simulator brakes
before applying drive in the new direction. `0.0` coasts. Values outside the
normalized ranges are clamped; `NaN` and infinite command values are rejected.

`RobotSensors` exposes:

| Group | Available information |
| --- | --- |
| `imu` | Heading, turn rate, pitch, roll, and acceleration |
| `odometry` | Signed speed and accumulated travel distance |
| `lidar` | Ranges that detect walls, cars, and blockers |
| `wall_lidar` | Wall-only ranges |
| `camera` | Processed track geometry and nearby competitors |
| `contact` | Current contact durations and accumulated damage |

The processed camera values are geometry, not raw pixels. Controllers do not
receive the mutable physics world, official race progress, future state, or
another controller's private state. See [SENSORS.md](SENSORS.md) for the full
field-by-field contract.

## Deterministic starting positions

Both single-car racing and head-to-head racing accept `--seed`. The seed chooses
a random position along the track using the same deterministic spawn algorithm
in both modes:

```bash
uv run racing --seed 110

uv run racing h2h \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110
```

The same seed reproduces the same single-car start and the same head-to-head
race sequence. For multiple head-to-head races, the race index deterministically
selects the next position in that sequence. Programmatic single-car callers can
use `GameConfig.random_seed`; explicit `spawn_position`,
`spawn_heading_degrees`, and `spawn_progress_distance_m` values take precedence
over their corresponding seeded defaults.

The simulator seed controls simulator placement only. It does not seed PyTorch,
NumPy, a genetic algorithm, or stochastic controller inference.

## Packaging a controller

A simple function is the smallest supported controller shape. Keep function
controllers stateless because a module-level object may otherwise be shared by
controller copies during a local multi-car run.

A model-backed or otherwise stateful controller should expose
`create_controller()`. The runtime calls the factory for every car and repeated
race so each receives independent state:

```python
from racing import RobotCommand, RobotSensors

RACING_NAME = "My Controller"
RACING_COLOR = "#4C8DFF"


class Controller:
    def __init__(self) -> None:
        # Load fixed parameters or a trained artifact here, on CPU.
        ...

    def __call__(self, sensors: RobotSensors) -> RobotCommand:
        # Convert public sensor values to the policy's representation.
        ...
        return RobotCommand(throttle=0.0, steer=0.0)


def create_controller() -> Controller:
    return Controller()
```

The callable object and function forms implement the same public
`RobotController` protocol. `RACING_NAME` and `RACING_COLOR` are optional
display metadata. Keep model files and controller helper modules under
`src/controllers/` so a controller can move without private simulator files.

## CPU and memory boundary

Submitted inference must run on CPU. CUDA, MPS, ROCm, and other accelerators
must not be required or selected. The complete controller process—including
Python, imported libraries, model parameters, temporary tensors, caches, and
controller state—must remain at or below **1.5 GiB of resident memory**.

For PyTorch, load artifacts onto CPU, enter evaluation mode, and use inference
mode:

```python
import torch

model = build_your_model()
state = torch.load(model_path, map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.to("cpu")
model.eval()

with torch.inference_mode():
    output = model(inputs)
```

Avoid retaining computation graphs, growing history buffers without bounds, or
creating a model on every control tick. The official isolated controller worker
hides common accelerator backends and stops a process tree that exceeds the
memory boundary. Local in-process races do not provide that security sandbox.

## Dependencies and controller artifacts

Add libraries for training or inference with:

```bash
uv add PACKAGE_NAME
uv sync --managed-python
```

Commit both `pyproject.toml` and `uv.lock` when dependency versions change.
Prefer CPU-capable packages and include imported library memory in the 1.5 GiB
limit. Training-only libraries do not need to be imported by the runtime
controller.

Keep inference artifacts small, read-only, and addressed relative to the
controller module rather than the current working directory. Export a
submission by naming the one controller the grader should evaluate:

```bash
uv run python scripts/export_student_controllers.py controllers.candidate
```

The archive packages the complete `src/controllers/` tree, including checkpoints
and other runtime assets, along with `pyproject.toml`, `uv.lock`, and a manifest
that identifies the selected module. The grader syncs the declared runtime
dependencies before loading only that controller. Do not put training-only
datasets, virtual environments, or experiment logs under `src/controllers/`;
generated `__pycache__` directories, `.pyc` files, and `py.typed` markers are
omitted from the archive.

## Capturing human demonstrations

Manual keyboard and gamepad driving can be captured as observation/action pairs:

```bash
uv run racing \
  --seed 110 \
  --record-human artifacts/human-driving.jsonl
```

The destination is append-only JSON Lines. Each physics tick produces one
independently parseable record:

```json
{
  "schema_version": 2,
  "record_type": "human_control_step",
  "session_id": "...",
  "simulation_time_s": 0.016666666666666666,
  "sensors": {
    "dt_s": 0.016666666666666666,
    "tick": 0,
    "imu": {},
    "odometry": {},
    "lidar": {},
    "wall_lidar": {},
    "camera": {},
    "contact": {}
  },
  "command": {"throttle": 1.0, "steer": 0.0}
}
```

The empty sensor objects only keep this example compact; actual records contain
every public field. A row captures the state immediately before its command is
applied, so the result of the action appears in the next row. Recording stops
when the car is eliminated or the app exits.

Commands contain normalized simulator controls rather than raw input events.
Infinite LiDAR no-hit values are serialized as JSON `null`. Each launch appends
with a new `session_id`; split trajectories on that ID rather than treating the
first row of a new session as following the previous session.

`--record-human` is limited to single-car manual mode and cannot be combined
with `--student-module` or `h2h`. The recording format does not prescribe an
observation vector, normalization strategy, imitation objective, or train/test
split.

## Comparing controllers

### Racing exported submissions by ID

To race two submissions from `assignment_8706145_export/`, run:

```bash
uv run python scripts/run_submissions.py 429149360 429458989 --seed 110
```

The first ID is the challenger and the second is the incumbent. This opens a
watched, 30-second race in three-quarter view with each controller's `RACING_NAME`
as its car label. Controllers without a name use their submission ID as a fallback.
Each submission's `formula110-submission.json` selects its controller; older
`formula110-exercise-submission.json` exports use their level 3 controller.
The script also accepts IDs written as `submission_429149360`.

Pass four distinct IDs to put four submissions on the same starting grid:

```bash
uv run python scripts/run_submissions.py 429149360 429458989 429929224 429859475 \
  --seed 110 --track-seed 2026 --fullscreen
```

For an eight-car heat, pass eight distinct IDs:

```bash
uv run python scripts/run_submissions.py \
  429929224 429859475 429894275 429570443 \
  429692891 429458989 429149360 429712675 \
  --seed 110 --fullscreen
```

Each submission controls one car, with its own `RACING_NAME` and `RACING_COLOR`.
Missing names fall back to submission IDs, and missing colors receive distinct
default paints. A heat runs all its controllers together and ranks their race
progress. With `--races`, standings use total scored distance across those races.
The default three-quarter view shows the shared race, and `--camera follow` follows the leader. The
`split_follow` camera is available only for two-ID head-to-head races.

Car-to-car contact counts toward the marshal's stuck timer only while the
affected car is moving at 3 mph or less. Faster contact does not add stuck time;
wall-contact, stationary-car, and off-track recovery still apply.

Use `--seed random` to choose a new seed; the script prints it so you can replay
the matchup with `--seed INTEGER`. This seed changes the starting position and
grid order. Use `--track-seed INTEGER` to generate a reproducible procedural
track. Other `racing h2h` or `racing heat` options pass through when placed after
the IDs:

```bash
uv run python scripts/run_submissions.py 429149360 429458989 \
  --seed random --races 3 --round-seconds 60 --no-music

uv run python scripts/run_submissions.py 429149360 429458989 \
  --headless --seed 42 --races 7

uv run python scripts/run_submissions.py 429149360 429458989 429929224 429859475 \
  --headless --seed 42 --races 3 --round-seconds 30 --json
```

Add `--fullscreen` to start the viewer in fullscreen. Use `--export-dir PATH`
for another extracted assignment, `--camera follow` to change the view, or
`--dry-run` to inspect the resolved command without running
submission code. Controller files load through the existing file-path
loader, so submissions may have the same controller filename.

For side-by-side `follow` cameras, use:

```bash
uv run python scripts/run_submissions.py 429929224 429859475 \
  --camera split_follow --fullscreen --seed 110
```

The challenger stays on the left and the incumbent on the right. Each pane
follows its team's first car when using multiple copies. Press `v` to cycle
between this view and the other h2h camera modes; race results and controls stay
shared across the window. The direct `racing h2h --watch` runner accepts the
same `--camera split_follow` option.

Watched head-to-head races and heats include a timing tower in the top-left
corner, with the blue F110 logo, race countdown, positions, and gaps to the
leader. Click **Timing [L]** or press **L** to hide or show it.

Click a timing row to switch to the close `follow` camera and keep following
that entrant through position changes and subsequent races. Hiding the tower
preserves that selection. **AUTO** returns to automatic camera targeting;
**V** still cycles views, and returning to `follow` retains the selected car.

Gaps compare the current time with the time the leader reached each trailing
car's scored race distance. Timing resets for each race and does not interpolate
across marshal recoveries; a dash means there is not enough history for a gap.

### Selecting heats from leaderboard results

Print one eight-car heat command for each category in an extracted Gradescope
assignment's `submission_metadata.yml`:

```bash
uv run python scripts/leaderboard_heats.py
```

Each command is preceded by the selected cars and their scores. Copy a command
to run that group. The script only prints commands; it does not launch races.
Car labels use submitters' first and last initials from the submission metadata:
`KJ` for Kris Jordan, or `KJ+MJ` for a team with Kris Jordan and Morgan Jordan.
The printed rankings include submission IDs to distinguish matching initials.

Use `--metric` repeatedly to select categories by name or unique substring.
Put race options after `--` to include them in every generated command:

```bash
uv run python scripts/leaderboard_heats.py \
  --metric "hits different" --metric "clock it" --metric "g's going crazy" \
  -- --fullscreen --seed 110 --races 3

uv run python scripts/leaderboard_heats.py --export-dir /path/to/assignment_export --list
```

Categories and ranking directions come from the exported leaderboard entries:
`order: asc` selects the lowest scores; `desc` or an omitted order selects the
highest. Only current results are used, never submission history. Missing,
`N/A`, and nonfinite scores are excluded, as are submissions whose selected
controller files cannot be found. Ties are broken by ascending submission ID.
Middle names are ignored when forming initials; a single-word name uses its
first two letters. Missing submitter names fall back to the submission ID.
Generating commands does not import student code.

If fewer than eight eligible cars remain, the script reports the shortfall and
skips that category's command. Use `--cars 4` for four-car heats, or `--list` to
see available categories, directions, and eligible counts. The output uses
shell comments for annotations, so it can also be saved as a shell script.

### Racing local controllers

For a heat with local controllers, repeat `--module` four or eight times:

```bash
uv run racing heat --watch --fullscreen \
  --module controllers.level_1 --module controllers.level_2 \
  --module controllers.level_3 --module controllers.crash_fast \
  --seed 110 --track-seed 2026
```

Omit `--watch` for a headless heat; add `--json` for machine-readable results.
Use `--races`, `--round-seconds`, and the same marshal settings as head-to-head
to configure each heat. If labels need overrides, repeat `--name` once per
module in module order. Repeat `--fallback-name` once per module to provide labels used
only when controllers omit `RACING_NAME`.

Use a watched race when you need to understand behavior:

```bash
uv run racing h2h --watch \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --races 1 \
  --round-seconds 30
```

Add `--fullscreen` to `racing h2h --watch` to start the viewer in fullscreen.

One side can use keyboard control in a watched race:

```bash
uv run racing h2h --watch \
  --challenger-keyboard \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --camera follow
```

Keyboard head-to-head requires `--watch`, and a headless race requires automated
controllers on both sides. Run several headless comparisons when you need
faster evidence:

```bash
uv run racing h2h \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --races 7 \
  --round-seconds 30
```

Races default to 30 seconds. On the starting grid, the car in the outside lane
starts ahead of the car in the inside lane. Scored distance is forward track
progress minus marshal penalties.
sides' scored distances.

### Machine-readable evaluation results

Evaluation and training harnesses do not need to scrape the terminal table. Add
`--json` to an automated headless race to print one versioned JSON document to
standard output, which can be parsed directly or redirected to a file:

```bash
uv run racing h2h \
  --challenger-module controllers.candidate \
  --incumbent-module controllers.baseline \
  --seed 110 \
  --races 7 \
  --round-seconds 30 \
  --json > evaluation.json
```

The top-level `summary` reports the suite winner, win counts, ties, and aggregate
margin. `races` contains each race's winner and margin plus both teams' scored
and raw distances, laps, damage, contact time, speed, off-track time, and marshal
activity. The document also records the seed, timestep, duration, and race rules
needed to interpret or reproduce the result. Check `schema_version` before
consuming saved results across simulator versions.

`--json` is intentionally limited to automated headless head-to-head races.
Use `--watch` for visual debugging, then run the same controllers and seed
without `--watch` to collect structured evaluation results.

For an in-process evaluation loop, the public Python API returns typed result
objects and exposes the same data as a JSON-compatible dictionary:

```python
from racing import load_student_submission, run_headless_head_to_head

candidate = load_student_submission("controllers.candidate")
baseline = load_student_submission("controllers.baseline")

result = run_headless_head_to_head(
    challenger_controller=candidate.controller,
    incumbent_controller=baseline.controller,
    challenger_name=candidate.display_name or "candidate",
    incumbent_name=baseline.display_name or "baseline",
    race_count=7,
    random_seed=110,
)
record = result.to_dict()

winner = result.winner
candidate_distance_m = sum(
    race.challenger.team_sum_distance_m for race in result.races
)
```

`run_headless_head_to_head` also accepts `fixed_delta_seconds`, copy counts,
race rules, and a `sensor_sample_callback` observation hook for a training or
analysis pipeline that also needs per-tick public sensor snapshots. Formula 110
does not define a replay buffer, reward, fitness function, optimizer, or
training loop.

## Reproducibility and fair comparison

Record at least the controller version, seed, race count, round duration,
timestep, and race rules. One seed or opponent is weak evidence; evaluate
across several seeds and retain a baseline controller for regressions.

Head-to-head outcomes can depend on traffic and contact, so solo distance is not
a substitute for racing against an opponent. Results include scored and raw
distance, laps, damage, contact, speed, off-track time, marshal activity, and
per-race winners.

Keep working controller versions and compare them directly:

```bash
cp src/controllers/candidate.py src/controllers/baseline.py
```

Improve the candidate, then evaluate both from identical seeds. A change that
looks better in one watched run can still lose distance or take more damage over
a multi-seed suite.

## Intentional non-goals

Formula 110 does not prescribe or provide a neural-network architecture,
genetic algorithm, observation vector, normalization scheme, reward, fitness
function, replay buffer, optimizer, training schedule, hyperparameters, or
experiment tracker. Those are controller-design decisions. The stable handoff
point is a CPU controller that fits within 1.5 GiB and maps the documented
sensor snapshot to a valid command.
