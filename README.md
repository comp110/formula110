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

## Event opening credit

Run the separate 3D intro from the project root:

```bash
uv run python intro.py
uv run python intro.py --fullscreen
```

The 40-second loop features the actual F110 mark as beveled Carolina blue
geometry, a slow full rotation, sweeping light beams, a polished argyle lane,
floor reflections, and a
Carolina blue Formula car that parks beneath the logo, performs a powered donut,
and accelerates off to the right. The donut alternates direction on each loop:
away from the camera, then toward it, with a camera pullback for the closer pass.
The stage stays lit continuously across the
loop. After exiting fully, the car teleports to its hidden starting point and
waits for the next entrance. Playback is silent.

The car uses the game's Bullet vehicle physics: throttle, brakes, steering,
suspension, and tire grip determine the motion and wheel transforms. Rear grip
is reduced and rear-wheel power boosted for the tight donut; normal grip and
power return for the full-throttle exit. Both runs are simulated at 120 Hz at
startup, then interpolated for smooth playback. The complete pair repeats every
80 seconds.

**Space** pauses, **R** restarts, **F** toggles fullscreen, and **Esc** exits.
Use `--duration 60` to slow the entire sequence to a minute, `--size 1920x1080`
to choose a window size, and `--title "FORMULA 110" --subtitle "RACE NIGHT"`
to customize the small captions. Pass an empty string to hide either caption.
It uses the existing project dependencies and requires a graphical desktop
with OpenGL support.

Capture a deterministic frame without opening a window:

```bash
uv run python intro.py --capture artifacts/intro-hero.png --time 10.5 --size 1920x1080
```

The capture path also needs access to the GPU; on macOS it must run outside a
sandbox that blocks the window server. The intro is independent of the racing
CLI and does not start a race or load a student controller.

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

## Track layouts

Use `--track bahrain` for a named layout following the Bahrain circuit's outline,
including the long main straight, opening bends, tight infield, and diagonal
return straight. The layout uses a roughly 598-meter lap at game scale, with the
standard road width. Rounded apexes and extra infield spacing
keep the road and barriers clear. View the entire circuit from above with:

```bash
uv run racing --track bahrain --camera top_down
```

The same track selection works in head-to-head and heat races:

```bash
uv run racing h2h --track bahrain --watch \
  --challenger-module controllers.level_2 \
  --incumbent-module controllers.level_3
```

Other named layouts are `mugello-short` (the default), `mugello-short-wide`, and
`mugello-short-long`. Use `--track-seed INTEGER` for a procedural track instead.

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
watched, 30-second race with a full-screen starting grid. The grid lists every
participant (both partners for pairs) as full first name and last initial, alongside
their car's color and label. Names come from `submission_metadata.yml`; missing
names fall back to submission IDs. Each controller's `RACING_NAME` remains its
car label in the smaller race leaderboard; unnamed cars use their submission ID.

Click anywhere or press **Space** to reveal the track. The default cinematic
camera holds above the entire track for two seconds, then begins the countdown
and smoothly pans and zooms into the grid
shot by the third red light, with two lights still to illuminate. Each subsequent
round returns to the starting grid and waits for another click or Space.
An explicit `--camera` option still selects another view.

Use `--title` to identify the race in the window and above the starting-grid
table. The title and table scale together to fill about 90% of the screen height,
with the table's left edge at the screen midpoint and a large F110 logo centered
in the left half:

```bash
uv run python scripts/run_submissions.py 429149360 429458989 --title "Grand Final"
```

Each submission's `formula110-submission.json` selects its controller; older
`formula110-exercise-submission.json` exports use their level 3 controller.
The script also accepts IDs written as `submission_429149360`.

Controllers can import their submission's sibling modules with `controllers.*`
or relative imports, including from a `src/controllers/` layout. Each loaded
submission has its own package namespace so matching helper names in different
submissions resolve to their own files.

Student `print(...)` calls are skipped by default, including at import time and
inside controller factories, without editing the submitted files. This applies
to the selected controller module; imported helpers are unchanged. Use
`--allow-student-prints` to keep debug output. Direct `racing` commands can opt in
with `--suppress-student-prints`.

Every watched race starts with five red lights illuminating one per second,
each with an F1 starting-light beep. After all five stay lit for 1.5 seconds,
they go out together and the cars and race clock start. This sequence repeats
for each round with `--races` and works with two to twenty cars. `M` mutes
the beeps along with the other audio; `--no-audio` keeps the visual countdown.
Headless runs start immediately without the presentation delay.

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

For a nine-car heat, pass nine distinct IDs using the same options.

Each submission controls one car, with its own `RACING_NAME` and `RACING_COLOR`.
Four-, eight-, and nine-car heats start in CLI order: the first ID takes pole, followed
by the remaining IDs from front to back. This order is preserved across races.
Missing names fall back to submission IDs, and missing colors receive distinct
default paints. A heat runs all its controllers together and ranks their race
progress. With `--races`, standings use total scored distance across those races.
The default three-quarter view shows the shared race, and `--camera follow` follows the leader.
Use `--camera split_follow` to watch the focused car and its closest trailing
competitor side by side, in either a heat or a head-to-head race.

Car-to-car contact counts toward the marshal's stuck timer only while the
affected car is moving at 3 mph or less. Faster contact does not add stuck time;
wall-contact, stationary-car, and off-track recovery still apply.

Use `--seed random` to choose a new seed; the script prints it so you can replay
the matchup with `--seed INTEGER`. This seed changes the starting position;
two-car head-to-head races also shuffle grid order. Use `--track-seed INTEGER`
to generate a reproducible procedural track. Other `racing h2h` or `racing heat`
options pass through when placed after the IDs:

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

Use `--camera helicopter` for a distant aerial view that pans toward the selected
car while slowly following it. The camera holds a steady world-space angle through
corners. It follows the leader in AUTO mode and works in single-car, h2h, and heat
viewers. Press `V` to cycle through top-down, three-quarter, drone, helicopter,
cinematic, leaders, and close follow views (plus split follow in h2h and heats).

Jump directly to a view with these keys:

| Key | View |
| --- | --- |
| **A** | Leaders / finishing camera |
| **Q** | Cinematic broadcast |
| **W** | Top-down overview |
| **E** | Three-quarter overview |
| **R** | Helicopter |
| **T** | Drone |
| **Y** | Split follow (h2h and heats) |
| **U** | Close follow |

These shortcuts preserve the selected car. Overview, cinematic, and leaders views
keep their usual framing; returning to drone, helicopter, close follow, or split follow resumes
the selected car. Pressing the current view's key again leaves its motion uninterrupted.
Camera shortcuts, **V**, and timing-row selections cut directly to their new
views, including opening and closing split-screen.

Press **A** or use `--camera leaders` for a drone view that keeps P1 and P2 in
frame. It includes P3 when its measured gap to P1 is at most three seconds;
missing timing history keeps the view on P1/P2. Framing adjusts for corners,
car separation, screen shape, and the timing tower. In a lap-limited race,
P1's final crossing locks a stationary shot with the finish line in view.
After P2's final crossing, this shot holds for one second, then smoothly pulls
out to the full-track top-down view over three seconds. Results wait for this
sequence to finish if the race has already ended. Manual view changes still
cut immediately and override the finishing shot.

Use `--camera cinematic` for automatic broadcast coverage. The opening shot
frames the entire grid through the countdown and lights-out display. It prefers
a low view from the next corner, checking sightlines against the barriers and
Formula110 banner. When the grid wraps around bends, it chooses a clear angle
from around the field and rises just enough to see the cars. The lens fits the
cars and complete Formula110 banner, with the banner along the top edge and
the starting lights below the cars. It then eases into race coverage. The director follows the most interesting battle
involving the top three, including a challenge from fourth for the final podium
place. It weighs gaps, closing speeds, and recent
overtakes. Shots hold for at least eight seconds when their subjects remain
eligible, and a better battle must stay interesting before the director switches.
Each shot keeps a steady world-space angle through bends. Changes between chase,
side, and aerial angles are limited to six degrees per second, with a steady race lens,
slower zoom, and gentle height corrections above barriers. Subject changes blend
over 5.5 seconds; quiet shots stay put instead of cycling angles on a timer.
As a close battle tightens, the camera gradually moves in while framing both cars.
When the field spreads out it returns to the leader. This view directs itself
even with a previously selected timing row; click a row to take over in helicopter
view. Single-car sessions get the same cinematic camera focused on their car.
Perspective views adjust depth precision with camera height to keep distant
red-and-white kerbs and asphalt from showing surface interference.

For side-by-side `follow` cameras, use:

```bash
uv run python scripts/run_submissions.py 429929224 429859475 \
  --camera split_follow --fullscreen --seed 110
```

The focused car appears on the left, and the closest active competitor behind
it in race order appears on the right. Automatic targeting follows P1 and P2.
Click a timing row or press **1**–**0** to change the focus while keeping both
panes; the trailing car updates as positions change. If the focus is last, the
other pane follows the nearest active car ahead. With no active competitor
left, the focus fills the window. Retired cars and DNFs are skipped when choosing
the companion. Each pane shows only its focused car’s place card, at any race
position, with its name, color, and a translucent triangle pointing down to the car.
Press **Y** to open this view directly, or **V** to cycle to it; results and
controls stay shared across the window. Both `racing h2h --watch` and
`racing heat --watch` accept `--camera split_follow`.

Watched head-to-head races and heats include a timing tower in the top-left
corner, with the blue F110 logo, race countdown, positions, and time gaps to the
car immediately ahead. It shows every entrant, including nine- and ten-car heats.
Click the **F110 logo** or press **L** to hide or show it.
Press **1**–**9** to focus the car currently in **P1**–**P9**, or **0** for **P10**,
just like clicking its leaderboard row. These shortcuts also work with the
tower hidden. The camera stays with the selected car as positions change;
press a number again to select whoever occupies that position now. Selecting
the same car again moves into close follow, except in split view, which keeps
both panes focused on the battle.
Floating badges show just the names. In single-camera views, the top three
also have `P1`–`P3` cards in a stable row across the top, clear of the timing
tower. Each card shows the car's name and color with contrasting rank text;
a translucent triangle points just above the visible car. Split-screen uses
the same style for each pane's focused car, regardless of its position, and
omits cards for other cars. Cards and pointers stay inside their own pane.
Ranks and colors update immediately with the race order. Positions account for
grid offsets and completed laps from the shared start/finish line. Retired cars
move below active cars, ordered with the newest retirement first. Cards hide
when the car is outside the view, behind the HUD, or retired, and during the
countdown, camera transitions, and final results.
Audio has no status overlay; press **M** to mute or unmute in any view.
When cars share a color, the leading car of that color keeps the original shade.
Matching cars darken by 10% of the original color per overall position behind
that leader (90%, 80%, and so on, with a 10% brightness floor). Shades update
with live order across the car paint, name badges, timing rows, and circles.
The bottom damage bars are hidden in watched races. At the end, the results
fade over the finishing scene in 1.5 seconds, using the same layout as the title
screen: F110 logo on the left, race title and final standings on the right.
Participant names and car colors carry over from the grid. Lap races show each
finisher's time, gap to the winner, and DNF classifications. Timed heats show
scored distance and available timing gaps; head-to-head results show team order
and best lap times (or elapsed time if no lap was completed). Multi-race heats
retain the official aggregate order, with placement totals and combined finish
times for lap races. The results stay visible when the race is over; press
Escape to exit the viewer.
Terminal and JSON results also include damage, summed across races (and across
copies in a head-to-head team), so a multi-race total can exceed 100%.

Add `--no-damage` to `racing h2h`, `racing heat`, or
`scripts/run_submissions.py` to disable collision damage and damage retirements
for every car and round. This works in watched and headless races; collisions,
contact sensors, and marshal recovery still operate normally. Damage is enabled
by default, and JSON results record the setting as `rules.damage_enabled`.

```bash
uv run python scripts/run_submissions.py 429929224 429859475 --no-damage
```

Click a timing row to focus on that entrant through position changes and
subsequent races. From top-down, three-quarter, cinematic, or leaders view, the first click
opens `helicopter`; clicking the same car again switches to close `follow`.
Selecting a different car in helicopter, drone, close follow, or split follow keeps that view.
Hiding the tower preserves the selection. Clicking the **lap counter or race clock** restores automatic camera
targeting while keeping the current view. **V** cycles views and retains the
selected car.

Gaps compare the current time with the time the car immediately ahead reached
each trailing car's lap-aware track position. Finishers show the difference
between their finish time and the previous finisher's. Timing resets for each race and does not interpolate
across marshal recoveries; a dash means there is not enough history for a gap.

### Searching Bahrain starting seeds

Search the eight-car field LB+NN, AR, YZ, JB, YH, KC, ZY+JL, and JS+MH for
LB+NN finishing first and JB second:

```bash
uv run python scripts/search_bahrain_seeds.py
```

The script tries seeds 110 through 1109, stopping at the first match. Each seed
runs in a fresh headless process on **Bahrain**, with five laps, a 10-second
finish timeout, and damage disabled. The grid keeps the listed car order.
Only actual P1/P2 finishes qualify; DNFs do not. Each match prints a complete
fullscreen cinematic replay command, including `--track bahrain`.

Set `--round-laps 3` (or `--laps 3`) to search three-lap races instead. The lap
count must be a positive integer and is included in matching replay commands.

To accept either finishing order, search a different range, find more matches,
and save results as an append-only JSONL log:

```bash
uv run python scripts/search_bahrain_seeds.py \
  --either-order --start-seed 0 --count 1000 --matches 5 \
  --output artifacts/bahrain-seeds.jsonl
```

Use `--export-dir` for another extracted assignment directory. Each seed has
a 300-second wall-clock limit; change it with `--timeout-seconds`. Timed-out
seeds are reported as skipped, not as nonmatches. Ctrl-C stops the search;
resume with the printed `--start-seed`. Exit status is 0 if any matches were
found, 1 if the range completed without matches, 2 for errors or an unmatched
search with timeouts, and 130 on interruption.

### Selecting heats from leaderboard results

Print one heat command for each category in an extracted Gradescope
assignment's `submission_metadata.yml`, using ten cars when available, nine when
only nine qualify, and eight otherwise:

```bash
uv run python scripts/leaderboard_heats.py
```

The default export is `assignment_8706145_export/`. Use `--export-dir` to select
`assignment_8488391_export/` instead:

```bash
uv run python scripts/leaderboard_heats.py --export-dir assignment_8488391_export
```

Generated race commands include the selected export directory automatically.

Each command lists cars from best to worst for that category, placing its top
qualifier in pole position. It is preceded by the selected cars and their scores.
Copy a command to run that group. The script only prints commands; it does not
launch races.
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

Use `--cars 4`, `--cars 8`, `--cars 9`, or `--cars 10` to require a specific heat size:

```bash
uv run python scripts/leaderboard_heats.py --export-dir assignment_8488391_export --cars 9
```

If fewer than the requested number of eligible cars remain, the script reports
the shortfall and skips that category's command. Use `--list` to
see available categories, directions, and eligible counts. The output uses
shell comments for annotations, so it can also be saved as a shell script.

### Planning race night from a final export

`scripts/plan_race_night.py` builds five exclusive ten-car category groups,
optional Juiced exhibition heats, and two independent ten-car Bahrain fields.
It calibrates lap counts toward 90-second heats and searches for interesting
track/starting seeds. Every top replay command includes a descriptive `--title`
for the window and starting-grid display.

Run the current show search with:

```bash
uv run python scripts/plan_race_night.py \
  --export-dir assignment_8706145_export \
  --selection-priority lowest \
  --review-csv artifacts/submission_review_scores.csv \
  --juiced-threshold 2 \
  --juiced-min-laps 3 \
  --track-count 10 \
  --starts-per-track 10 \
  --fixed-seed-count 100 \
  --bahrain-seed-count 100 \
  --calibrate-laps \
  --target-seconds 90 \
  --duration-tolerance-seconds 10 \
  --duration-weight 0.2 \
  --jobs 8 \
  --marshal-penalty-m 3 \
  --marshal-cooldown-seconds 0.5 \
  --marshal-stuck-seconds 1.0 \
  --fullscreen \
  --camera cinematic \
  --weights 0.4 0.2 0.4 \
  --output artifacts/race-night-all-spawns-bahrain/plan.json
```

Category allocation uses current export leaderboard results, never history.
The default `--selection-priority highest` allocates Clock It, Gas Locked In,
Hits Different, Gs Going Crazy, then Sips Tea. `lowest` reverses that allocation
order. Each board selects its best remaining scores in its normal ranking
direction and backfills after earlier allocations and Juiced exclusions.
Show order always runs from Sips Tea through Clock It, then Juiced and Bahrain.
A short field is an error; submissions are never duplicated to fill a category.

Juiced selection requires `total_score >= --juiced-threshold` (default 1),
**and**, if configured, `all_spawns_no_crumbs_laps >= --juiced-min-laps`.
The command above uses **score >= 2 AND All Spawns laps >= 3**. Blank lap
values fail the lap condition. The review CSV must contain `submission_id` and
`total_score`, plus the lap column when filtering by laps. Missing/blank review
scores for eligible entrants, duplicate IDs, invalid numbers, and IDs outside
this export are errors. Additional experience columns are accepted.

Juiced entrants are removed before regular category selection and race in
separate balanced heats of two through ten cars, without regular fillers.
A single held-out entrant stays excluded and produces a warning instead of a
one-car heat. Omitting `--review-csv` disables holdouts.

Bahrain selection starts afresh from the **All Spawns, No Crumbs leaderboard**, independent
of category allocations and earlier race outcomes:

- **Bahrain - All Spawns, No Crumbs Top 10 (No Juiced)** selects the ten highest
  lap scores after excluding the Juiced holdouts.
- **Bahrain - All Spawns, No Crumbs Top 10 (Open, Including Juiced)** selects the
  ten highest lap scores across everyone, including Juiced entrants.

Scores rank in descending order; ties use ascending submission ID. The metric
comes from the current export leaderboard, independently of the review CSV.

Cars can appear in a regular category and either or both Bahrain fields.
Each Bahrain field runs exactly **three laps** and searches **100 starting
seeds by default**, for 200 Bahrain trials total.

For a twenty-car Bahrain experiment, add `--bahrain-cars 20` and use a new
output path. This expands both Bahrain fields while retaining ten-car regular
categories. Bahrain's field size defaults to 10 until specified. The timing
tower currently displays all entrants together; paging is deferred.

Compare running ten- and twenty-car Bahrain races in overhead view:

```bash
uv run python scripts/benchmark_bahrain_race.py \
  --export-dir assignment_8706145_export \
  --counts 10 20 --size 1920x1080 --seconds 20
```

The benchmark uses the highest All Spawns scores, including Juiced entrants,
with the same seed, real-time physics, and student controllers in fresh
processes. It measures uncapped offscreen rendering with audio disabled;
window/compositor and audio costs are excluded. Results, screenshots, and
`replay-20.sh` are saved in `artifacts/bahrain-car-count-performance/`.

| Show order | Field | Laps | Default seed search |
| --- | --- | --- | --- |
| 1 | Sips Tea | Calibrated | 10 procedural layouts × 10 starts |
| 2 | Gs Going Crazy | Calibrated | 10 procedural layouts × 10 starts |
| 3 | Hits Different | Calibrated | 10 procedural layouts × 10 starts |
| 4 | Gas Locked In | Calibrated | 10 procedural layouts × 10 starts |
| 5 | Clock It (no Juiced) | Calibrated | Default Mugello Short layout × 100 starts |
| 6 (optional) | Each Juiced exhibition | Calibrated | 10 procedural layouts × 10 starts |
| Last two | Bahrain without Juiced; Bahrain open | 3 each | 100 starts each |

Without Juiced exhibitions this is **700 seed trials**, plus lap calibration.
Each Juiced heat adds 100 seed trials. Procedural track seeds are 0–9 and starting
seeds are 110–119 per layout; Clock It and Bahrain starts are 110–209.
`--start-track-seed`, `--start-seed`, `--track-count`, `--starts-per-track`, and
`--bahrain-seed-count` change these ranges. The last flag is the count **per
Bahrain field**. Clock It always uses the game's default `mugello-short` layout,
without a procedural track seed, and excludes Juiced holdouts. `--fixed-seed-count`
sets the number of Clock It starting seeds and any fixed non-Bahrain exhibition
trials; its default is `track-count * starts-per-track`. `--juiced-track` can
explicitly choose a fixed exhibition track.

Before the full search, each non-Bahrain field/layout runs pilots using its
first starting seed. The planner measures the initial lap count, estimates a
better count, and tests additional counts until the heat reaches **80–100
seconds**. This is simulated time from race start until the heat ends, including
the post-P1 finish deadline, excluding countdown and presentation screens.
Different layouts can use different lap counts. Initial guesses are 6 laps for
Sips Tea, Gs Going Crazy, Hits Different and Juiced, 8 for Gas Locked In, and 10
for Clock It.

`--target-seconds` and `--duration-tolerance-seconds` set the target and window.
Calibration is bounded by `--calibration-attempts` (6 per field/layout) and
`--max-calibration-laps` (30). If no tested lap count reaches the window, a warning
records the closest measured count used instead. If no pilot has a finisher,
calibration is marked unavailable and search uses the initial count. Different
starting seeds can still change the eventual duration.

`--no-calibrate-laps` skips pilots. Repeated `--laps CATEGORY=N` options fix a
category's length and skip its calibration (for example, `--laps clock-it=9`).
Category IDs are `sips-tea`, `gs-going-crazy`, `hits-different`, `gas-locked-in`,
`clock-it`, and `juiced`. Both Bahrain fields stay at three laps.

Ranking blends the existing action score with a duration score:

```text
duration_score = 100 * exp(-0.5 * ((heat_seconds - target_seconds) / tolerance_seconds)^2)
score = (1 - duration_weight) * action_score + duration_weight * duration_score
```

All-DNF races get zero duration points. `--duration-weight` is a share from 0 to
1, default 0.2. `--weights LEAD PASS FINISH` controls the relative action terms,
so the example contributes 32% lead changes, 16% overtakes, 32% close finishes,
and 20% duration. JSON retains all original action components and the duration
breakdown. Bahrain uses this ranking while keeping its three-lap length.

Workers run in fresh interpreters with a default 300-second wall-clock limit
(`--timeout-seconds`). `--jobs` defaults to 4. Damage defaults off and marshals
on, with a default stuck threshold of 1.0 second. Marshal/finish-deadline settings
apply to pilots, seed search and replays.
Playback flags (`--camera`, `--fullscreen`, `--no-music`, `--no-audio`, `--muted`)
do not change simulation or scores.

Add `--plan-only` to inspect fields and seed counts without calibration or race
simulation. Resume an interrupted run with the same command plus `--resume`;
`--retry-failed` also retries recorded errors/timeouts. Completed pilots are
checkpointed, reused on resume, and count toward the full seed search when
their lap count is selected. They are not simulated twice.

Use a **new output path** for this plan format or after changing the export,
controllers, engine, review CSV, thresholds, calibration or race settings.
Playback-only flags may change on resume. The plan uses schema version 2:

- `groups`, `juiced_groups`, and `bahrain_groups` record the fields and their
  compositions. `lap_calibration` records chosen laps, pilot durations, statuses,
  and trial IDs by layout. Each stage retains its top three races.
- Each top race includes its actual lap count in `spec`, original `interest`,
  combined `quality`, full `classification`, `podium`, and `launch` with `cwd`,
  `argv`, and a shell-quoted command containing `--title`.
- `show_runner` includes introductions, launches, results, and alternatives for
  the regular groups, optional exhibitions, and both Bahrain fields.
- `plan.trials.jsonl` stores pilot/search phases, exact settings, durations,
  scores, events and errors. `plan.json` is checkpointed after each trial.

Final console output lists each field's top three titled replay commands,
with actual laps, seconds, scores and podiums. Print each field's best command:

```bash
jq -r '.show_runner[] | .steps[] | select(.type == "launch_race") | .shell_command' \
  artifacts/race-night-all-spawns-bahrain/plan.json
```

### Scoring races for interesting replays

Use `scripts/score_races.py` to run headless lap races and rank starting seeds
by sustained lead changes, overtakes, and the closeness of the top-three finish.
It accepts two to ten submission IDs and optional repeated `--name` labels,
or reads commands printed by `leaderboard_heats.py`:

```bash
uv run python scripts/leaderboard_heats.py --metric "Clock It" | \
  uv run python scripts/score_races.py --commands - \
    --laps 5 --no-damage --start-seed 110 --count 20 \
    --output artifacts/clock-it-interest.jsonl
```

This uses the default **Mugello Short** track. Use `--track bahrain` for Bahrain
or `--track-seed 42` for a procedural layout. `--seed`/`--start-seed` changes
the starting position, independently of the track seed. `--count` tries that
many consecutive starting seeds for each track and input field; every trial is
race 1 in a fresh process. Replay commands include both seeds when procedural
tracks are used.
Grid order is preserved from the input command. Damage is on by default;
`--no-damage` disables it, and `--damage` explicitly enables it.

To search procedural layouts, combine `--start-track-seed` (an alias of
`--track-seed`) with `--track-count`:

```bash
uv run python scripts/leaderboard_heats.py --metric "Clock It" | \
  uv run python scripts/score_races.py --commands - \
    --start-track-seed 0 --track-count 20 \
    --start-seed 110 --count 3 --laps 5 --no-damage \
    --output artifacts/clock-it-track-search.jsonl
```

This runs **60 races**: track seeds 0–19, each with starting seeds 110–112.
Use `--count 1` to hold the starting seed fixed while searching layouts.
The default `--track-count 1` keeps a single layout. A track range requires a
procedural track seed, supplied directly or inherited from each input command.
Results rank individual track/starting-seed combinations, and include both
seeds in the log, console rankings, and replay commands.

To reuse saved commands, replace `--commands -` with `--commands heats.txt`.
Command text is parsed as arguments, never executed as shell code. Rendering
options such as `--fullscreen` and `--camera` do not affect headless trials.
Explicit scorer flags override inherited settings. Source commands with
`--round-seconds` or `--races` greater than 1 are rejected: use laps and the
scorer's `--count`. Defaults are five laps, seed 110, and a 10-second wait after
P1 finishes; explicit command values for these settings are preserved. Use
`--finish-timeout-seconds` to change the post-winner deadline. A separate
`--timeout-seconds` limits each trial to 300 wall-clock seconds by default.

The default score is:

```text
score = 30 * L/(L + laps)
      + 30 * O/(O + laps*(N - 1))
      + 40 * exp(-mean(P2_time - P1_time, P3_time - P1_time) / 2 seconds)
```

Here `L` is confirmed lead changes, `O` is confirmed overtakes, and `N` is the
number of cars. A pass for the lead earns overtake credit plus the lead-change
bonus. Normalizing event counts by laps and field size reduces the automatic
advantage of longer/larger races; diminishing returns keep repeated exchanges
from overwhelming the finish. One lead change per lap earns 15 lead points;
one pass per opponent per lap earns 15 overtake points. The finish term uses
actual interpolated finish timestamps. A zero-gap podium earns all 40 points;
a 2-second average gap earns about 14.7. Fewer than three finishers earns zero
finish points, with missing gaps recorded as null.

Change the weights with `--weights 30 30 40` (lead, overtakes, finish); they
are normalized to 100. `--finish-gap-scale-seconds 2` sets the finish-gap decay
scale. Inspect the formula and hypothetical sanity checks without running cars:

```bash
uv run python scripts/score_races.py --explain-score
```

Events use lap-aware race progress at every physics tick. An overtake requires
a 0.5-meter advantage held for 0.3 seconds. Initial grid placement, lapping,
retirement promotions, and promotions after a finisher crosses are excluded.
Marshal teleports reset affected comparisons with a 2-second cooldown. This
deliberately filters very brief exchanges, including last-instant passes that
do not persist before finishing; the finish-gap term still captures that close
finish. Passes of active cars slowed by collisions can count, so this is an
audience-interest heuristic rather than a measure of clean driving.

The append-only JSONL log records every trial's settings, weights, score
breakdown, full classification (including DNFs and marshal counts), timestamped
events with car-name mappings, and a fullscreen cinematic replay command.
Errors and wall-clock timeouts have no score. The console prints the best three
replay commands; change that with `--top`. Exit status 2 indicates a failed or
timed-out trial; Ctrl-C preserves completed trials and returns 130. Finisher
counts are reported alongside scores without an additional hidden penalty.
Compare the same field/track/laps/rules first, then watch high- and low-scoring
examples to tune what feels interesting. Seed reproducibility also depends on
the submitted controllers being deterministic.

### Racing local controllers

For a heat with local controllers, repeat `--module` two to ten times:

```bash
uv run racing heat --watch --fullscreen \
  --module controllers.level_1 --module controllers.level_2 \
  --module controllers.level_3 --module controllers.crash_fast \
  --seed 110 --track-seed 2026
```

Omit `--watch` for a headless heat; add `--json` for machine-readable results.
Use `--races`, `--round-seconds` (or `--round-laps`), and the same marshal settings as head-to-head
to configure each heat. If labels need overrides, repeat `--name` once per
module in module order. Repeat `--fallback-name` once per module to provide labels used
only when controllers omit `RACING_NAME`.

For a lap-based heat, replace `--round-seconds` with `--round-laps 3`. This works
with `racing heat` and the heat-mode `run_submissions.py` commands, both
watched and headless. The two length flags are mutually exclusive.

Each car must complete the specified laps from the shared start/finish line;
the initial crossing from the grid does not count as a lap. Finishers lock in
P1, P2, and so on in crossing order. Cars eliminated before finishing are DNFs.
The round ends when everyone has finished or retired, or when the finish timeout
expires after P1 finishes, with any remaining cars marked DNF. The timeout defaults
to 20 simulation seconds; set `--finish-timeout-seconds 10` to change it, or `0`
to end immediately after P1. For example, use `--round-laps 3 --finish-timeout-seconds 10`.
There is no countdown
until the first finisher; an all-DNF round ends immediately. The timing tower
shows lap progress, then the configured countdown and final classifications.

With `--races` greater than one, the lowest sum of finishing places wins. Each
DNF adds field size + 1 (5 for four cars, 9 for eight, 10 for nine), and equal totals share
the same rank. Distance and marshal distance penalties do not decide lap-race
placements. JSON results include each car's finish position, finish time, and
DNF status, as well as aggregate placement totals.

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
Press **F** or **F11** to toggle borderless fullscreen in any racing view.

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

## Reviewing submission conventions

Scan an extracted submission directory and write every submission's ID, total
review score, exported race metrics, and reported programming experience to a
CSV, sorted by review score highest first:

```bash
uv run python scripts/review_submissions.py assignment_8706145_export \
  --experience-csv submission_metadata.csv \
  --csv artifacts/submission_review_scores.csv
```

Without `--csv`, the script writes CSV to stdout, so shell redirection also works.
Diagnostics go to stderr. The scanner parses submitted code without importing
or executing it. It inspects only the
controller selected by the root submission JSON, including `src/controllers/`
layouts and level 3 in older exercise manifests. Other submitted files and
imported helpers do not contribute to the score. Missing, invalid, or unreadable
controllers remain in the CSV with a blank score, after scored submissions.

The CSV columns are `submission_id`, `total_score`, `all_spawns_no_crumbs_laps`,
`clean_lap_seconds`, `experience_level`, `experience_rank`, and `experience_match`.
Race metrics come from each submission's current
leaderboard in `submission_metadata.yml`, read with the project's existing
PyYAML dependency. `clean_lap_seconds` is the exported **Clock It (s)** metric:
the average of each starting offset's fastest clean lap. Missing/ineligible
metrics remain blank; historical results are never substituted. Race metrics
do not change the review score or sorting, and are included even if a controller
could not be scanned. Exports without a metadata file still produce the CSV,
with blank race metric columns.

The experience columns use **Question 1.1 Response** in the survey CSV.
`--experience-csv` selects a survey explicitly; otherwise the script automatically
uses `submission_metadata.csv` beside the export directory if that file exists.
Student IDs and normalized email addresses join survey responses to the current
submitters listed in the export's `submission_metadata.yml`. Survey submission
IDs belong to the survey assignment and are not used to join racing submissions.
Ambiguous or conflicting identity matches remain unknown.

For pairs, `experience_level` contains the highest known response, with
`experience_rank` ordered as **0 = None**, **1 = A little experience**,
**2 = Some experience**, **3 = Substantial experience**. The level column preserves
the full response label. `experience_match` is `complete` when all submitters
have a matched response, `partial` when only some do, and `unmatched` when none
do. Blank or absent responses remain unknown, distinct from an actual **None**
answer. Without a survey, all three experience columns are blank. Experience is
descriptive context and does not contribute to the review score.

The score sums each rule's weight times its occurrence count, capped per rule:

| Rule (`--weight` name) | Points per occurrence | Maximum counted occurrences |
| --- | ---: | ---: |
| `precise_float` | 2 | 5 |
| `nonlocal` | 5 | 2 |
| `callable` | 3 | 2 |
| `persistent_state` | 5 | 4 |
| `unannotated_local` (disabled by default) | 0 | 10 |
| `versioned_name` | 2 | 1 |
| `tuple` | 2 | 4 |
| `for_loop` | 3 | 4 |
| `zip` | 4 | 2 |
| `class` | 6 | 2 |

Floating-point literals need at least eight significant mantissa digits by
default; magnitude and zero padding alone do not count. Override this with
`--float-digits 7`. Unannotated locals contribute no points by default. If enabled
with `--weight unannotated_local=1`, they count once per name per function, excluding
parameters, `global`/`nonlocal` bindings, and names annotated elsewhere in that
same function. Tuple annotations, tuple values/construction, and comprehension
loops count; tuple unpacking and generic type argument lists do not. Version
markers are read from literal `RACING_NAME` assignments, not controller filenames.

Potential persistent state includes function/object attributes, mutable default
arguments, mutation of captured closure bindings, and `functools` caches.
`nonlocal` has its own rule; ordinary module globals are allowed. Static analysis
cannot establish whether state actually survives ticks and does not resolve all
dynamic Python behavior or imported aliases. Comments and prose do not score.

Weights are review heuristics, not calibrated AI probabilities or proof of AI
assistance. Different rules can describe the same code (for example, a class
storing instance state). Adjust weights with repeatable options such as
`--weight precise_float=3 --weight versioned_name=0`; a zero weight retains
the scan but removes that rule's contribution. The default maximum is 88.

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
