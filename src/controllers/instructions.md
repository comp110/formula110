# Formula 110: Teach a Car to Race

In this exercise, you are programming the decision-making logic for a self-driving Formula 110
race car.

The simulator calls your `control` function 60 times each second.
Every call receives a new snapshot of the car's sensors, and your function must
decide how much to accelerate and which way to steer.

## Get started in the workspace

Before beginning the exercise, synchronize the COMP110 workspace:

1. In VS Code, select **Terminal > Run Task**.
2. Select **Sync all workspace projects** for the `support` folder.
3. Follow the messages in the terminal. The first sync may take a few minutes
   while the project dependencies are installed.

In the VS Code Explorer, `formula110` should appear as a top-level workspace
folder. If it does not, make sure you opened `workspace.code-workspace` rather
than opening only the `trailhead` folder.

## Learning goals

By the end of this exercise, you will be able to:

- define variables and use their values in a returned object;
- use conditional statements to make decisions from sensor readings;
- access an object's attributes with the dot operator;
- write arithmetic expressions whose values change as their inputs change;
- compare two programs with a repeatable experiment; and
- use documentation to learn an unfamiliar interface.

## The racing progression

This exercise has four levels. Preserve each completed level in its own file
before moving on so you can race the levels against one another.

- **Level 0:** Use two wall sensors and conditional statements to complete a
  lap.
- **Level 1:** Use the speed sensor to go faster without exceeding a chosen
  maximum speed.
- **Level 2:** Scale the throttle continuously with speed instead of treating
  it as only on or off.
- **Level 3:** Replace the two wall sensors used so far with a camera-based
  strategy you design from the sensor documentation. You may continue using
  the speed sensor.

## Before you begin

Your programs belong in `src/controllers/`. The starter is
`src/controllers/demo.py`, and your four programs will be:

```text
src/controllers/level_0.py
src/controllers/level_1.py
src/controllers/level_2.py
src/controllers/level_3.py
```

Each submitted file must keep a non-empty module docstring as its first Python
statement and define `__author__` as your nine-digit PID. Replace the starter
placeholder while working on Level 0, then preserve it when copying the file
for later levels:

```python
"""Describe this controller."""

from racing import RobotCommand, RobotSensors

__author__: str = "123456789"
```

Every function you write must annotate all parameters and its return type. The
required `control` function must retain the exact typed shape
`control(RobotSensors) -> RobotCommand` (the parameter name may differ) and a
non-empty function docstring as its first body statement.

Run all commands in this handout from a terminal opened for the `formula110`
project.

### Run the demo

First, open `src/controllers/demo.py`. Its controller always requests full
throttle and never steers. Run it with:

```text
uv run racing --student-module controllers.demo
```

Watch what happens when the car reaches a wall. The crash is useful: it shows
why a controller needs information about the world rather than returning the
same command forever.

Close the simulator window when you are done. A running simulator does not
automatically reload a program after you edit it, so close and rerun it after
each meaningful change.

### Drive one lap yourself

Now start the simulator without a student module:

```text
uv run racing
```

Use the arrow keys:

- `Up Arrow` requests forward throttle;
- `Down Arrow` requests reverse and first brakes when the car is moving
  forward;
- `Left Arrow` steers left; and
- `Right Arrow` steers right.

Press `V` to cycle through camera views and `M` to mute or unmute the audio.
Try to complete a lap. Notice that you naturally make decisions of when to turn
left or right and when to accelerate, brake, or go into reverse.

Your goal in this exercise is to write logic that _controls_ the car by expressing decisions
as conditional statements and floating-point arithmetic.

## Meet the controller interface

The important part of the starter is:

```python
def control(sensors: RobotSensors) -> RobotCommand:
    return RobotCommand(throttle=1.0, steer=0.0)
```

The simulator supplies the `sensors` argument. The data type `RobotSensors` contains lots of sensory input inforamation
collected about the current state of your car in the present moment. These sensors include things like wall distance
sensors, current speed sensors, and so on.

Your function returns one `RobotCommand` object that bundles two `float` values:

| Command    | Range           | Meaning                          |
| ---------- | --------------- | -------------------------------- |
| `steer`    | `-1.0` to `1.0` | full left to full right steering |
| `throttle` | `-1.0` to `1.0` | reverse to full forward drive    |

The values between the endpoints influence the intensity of each. Think of them like percentages.

Steer `steer=-0.5` instructs turning the steering wheel to 50% of its maximum amount to the left, `steer=1.0` instructs
turning the steering wheel 100% to the right.

For acceleration, `throttle=0.2` is pressing the gas pedal down 20%, where as `throttle=1.0` is "pedal to the metal"
fully depressing the gas pedal for maximum acceleration. The `throttle` output is nuanced, while moving forward a
negative throttle will _brake_ to decellerate. Once stopped, a negative throttle will accelerate the car in reverse.
This is a racing simulation so, hopefully, you won't need reverse!

## The first three sensors

Levels 0 through 2 use three sensor properties:

| Sensor property                    | Type and unit              | Meaning                                             |
| ---------------------------------- | -------------------------- | --------------------------------------------------- |
| `sensors.wall_lidar.front_left_m`  | `float`, meters            | distance from a forward-left ray to a track wall    |
| `sensors.wall_lidar.front_right_m` | `float`, meters            | distance from a forward-right ray to a track wall   |
| `sensors.odometry.speed_mps`       | `float`, meters per second | signed forward speed; positive means moving forward |

The wall LiDAR's `front_left_m` and `front_right_m` rays point slightly to either side of the car's nose:

```text
               direction of travel
                        ^
                       / \
     front-left ray  / [car] \  front-right ray
          -20 degrees           +20 degrees
```

Each wall reading reports the number of meters to the first track barrier in
that direction. A smaller value means the wall is closer.

Python's **dot operator** accesses an attribute stored on a compound object. Reading
`sensors.wall_lidar.front_left_m` works from left to right:

1. begin with the `RobotSensors` object named `sensors`;
2. access its `wall_lidar` object; and
3. access that object's `front_left_m` value.

This is the same idea as accessing one folder inside another: each dot moves
one level deeper to a specific piece of information. We will learn how to create our own
types of objects like this later in the course.

## Level 0: Complete a lap with conditionals

Make a copy of `src/controllers/demo.py` named
`src/controllers/level_0.py`. Work only in the new file for this level.

Change `RACING_NAME` to `"Level 0"`. If you'd like to change the car's paint color, you can
use an "RGB Hexadecimal" string such as found on a color palette site. As a bit of trivia,
these strings encode binary representations of three integer values: red, green, and blue
intensities. A string like `#000000` is black and `#990000` is red because the two 9s are in
the red color bits' positions. You do not need to know this for this course right now, but it does all
map down to bits and integers like you learned about in the first week of COMP110! (And it's
fun to choose your race car's paint color!)

Update the `control` function's docstring to describe it as a minimally viable self-driving controller.

Inside `control`, define two `float` variables before the return statement: `throttle` and `steer`. Then update the existing `RobotCommand` call so its `throttle` and `steer` arguments use these variables as their assigned values rather than fixed numbers.

These variable names are suggestions; other descriptive names are accepted
throughout the levels. You may pass your variables to `RobotCommand` by
position (throttle first, steering second), by keyword, or a valid mixture of
the two. The sensor parameter's name is also your choice.

### Translate the steering plan into Python

Use conditional statements to express this plan:

1. If the forward-left wall is less than four meters away, steer 100% to the right.
2. Otherwise, if the forward-right wall is less than four meters away, steer fully left.
3. Otherwise, point the wheels straight ahead.
4. Request a forward throttle of only 10%.

Remember that positive `steer` values turn right and negative values turn
left. Keep the return statement last so it sends the values chosen by your
conditional logic to the car.

Run Level 0 by itself from the terminal:

```text
uv run racing --student-module controllers.level_0
```

Let it run until it completes a full lap. If it crashes, compare the order,
indentation, and signs of your steering assignments with the English plan.

From the terminal, know that you can use the up arrow key to scroll back
to commands you have previously run and reuse them.

Try increasing the throttle to find a value between 10% and 20% that successfully
completes a lap without crashing.

### Level 0 checkpoint

Before moving on, confirm that:

- `RACING_NAME` is `"Level 0"`;
- `control` defines both `throttle` and `steer` variables;
- the return statement uses those variables;
- the program contains conditional logic;
- it reads both forward wall sensors; and
- it completes at least one lap

## Level 1: Set a maximum speed

Make a copy of `src/controllers/level_0.py` named
`src/controllers/level_1.py`. Work only in the new file and change
`RACING_NAME` to `"Level 1"`.

Level 0 leaves most of the available motor power unused. Level 1 should request
more throttle while the car is below a maximum speed, then back off the
throttle when the car reaches that speed.

Read the current speed from `sensors.odometry.speed_mps` in a conditional
statement. Choose a maximum speed, compare the reading with that maximum, and
assign different throttle values on the two branches. A maximum between `8.0` and `12.0`
meters per second is a useful starting point, but you may tune it.

Try tuning your turning logic by aiming to stay a little further away from walls.

Run Level 1 by itself while tuning:

```text
uv run racing --student-module controllers.level_1
```

### Race Level 1 against Level 0

The most direct test of an improvement is to run both programs in the same
race. The level you are testing is the challenger:

```text
uv run racing h2h \
  --challenger-module controllers.level_1 \
  --incumbent-module controllers.level_0 \
  --round-seconds 30 \
  --watch
```

Your goals are for Level 1 to win and to travel at least `250.0` meters during
the race. The results appear in the simulator when the round ends. If the car
does not reach both goals, adjust its maximum speed or throttle assignments and
race again.

### Level 1 checkpoint

Confirm that:

- `RACING_NAME` is `"Level 1"`;
- the two wall sensors still control steering;
- a conditional reads `sensors.odometry.speed_mps`;
- the controller requests less throttle at or above its chosen maximum speed;
- Level 1 travels at least `250.0` meters in the seeded race; and
- Level 1 beats Level 0.

## Level 2: Scale throttle with speed

Make a copy of `src/controllers/level_1.py` named
`src/controllers/level_2.py`. Work only in the new file and change
`RACING_NAME` to `"Level 2"`.

The Level 1 throttle is **discrete**: its conditional chooses one of a small
number of fixed values. Throttle and steering do not have to be all or
nothing. In Level 2, calculate throttle with an arithmetic expression whose
result changes continuously as speed changes.

Replace the Level 1 maximum-speed conditional with an assignment statement
where you need to write the expression to fill in the blank:

```python
throttle = _________________
```

Your expression has useful behavior at several speeds:

| When Speed Is | Calculate next `throttle` to be |
| ------------: | ------------------------------: |
|     `0.0` m/s |                           `1.0` |
|     `7.5` m/s |                           `0.5` |
|    `15.0` m/s |                           `0.0` |
|    `22.5` m/s |                          `-0.5` |

Try pluging some speed and throttle values in from the table above to compute what the `?` value should be:

`throttle = (? - speed) / ?`

You may store `sensors.odometry.speed_mps` in a temporary variable such as
`speed` and use that variable in your expression. Algebraically equivalent
expressions are also accepted; the autograder checks the throttle returned at
several speeds, including speeds between the examples above.

The expression provides strong acceleration when the car is slow and gradually
backs off as it approaches `15.0` meters per second.

Preserve the steering logic from the previous levels.

Run Level 2 by itself:

```text
uv run racing  --student-module controllers.level_2
```

Then race it against Level 1:

```text
uv run racing h2h \
  --challenger-module controllers.level_2 \
  --incumbent-module controllers.level_1 \
  --round-seconds 30 \
  --watch
```

### Level 2 checkpoint

Confirm that:

- `RACING_NAME` is `"Level 2"`;
- the throttle calculation scales continuously with speed as shown above;
- the two wall sensors still control steering;
- the calculated throttle can take values between `0.0` and `1.0`; and
- Level 2 beats Level 1 in the seeded head-to-head race.

## Level 3: Design a camera sensor strategy

Make a copy of `src/controllers/level_2.py` named
`src/controllers/level_3.py`. Work only in the new file and change
`RACING_NAME` to `"Level 3"`.

For the final level, replace the steering strategy you have been given with one
you design from a processed camera input. Review the `sensors.camera` section of
`SENSORS.md` in the root of the `formula110` project. The camera properties are
`visible`, `center_offset_m`, `heading_error_degrees`, `lookahead_offsets_m`,
`lookahead_distances_m`, and `competitors`. Choose at least one and use that
reading to decide the value of `steer` or `throttle` in your `control` function.

Your Level 3 controller may use any documented sensor properties except the two
forward wall sensors used in the earlier levels:

- do not use `sensors.wall_lidar.front_left_m`;
- do not use `sensors.wall_lidar.front_right_m`.

All other sensors are fair game, including `sensors.odometry.speed_mps`. You may
keep the Level 2 throttle expression or use the speed sensor in a new expression
or conditional, but the new strategy must read a property through
`sensors.camera`.

The goal is open-ended: use at least one camera input, conditional logic, and/or
arithmetic expressions to make a Level 3 car that beats Level 2. A new steering
strategy may not be enough by itself, so you may also want to revisit the Level
2 throttle expression. Consider whether the camera and speed readings can help
you decide when to use more or less throttle. Make one change at a time so you
can tell whether your experimental design succeeded.

Run Level 3 by itself:

```text
uv run racing --student-module controllers.level_3 --seed 110
```

Then race it against Level 2:

```text
uv run racing h2h \
  --challenger-module controllers.level_3 \
  --incumbent-module controllers.level_2 \
  --round-seconds 30 \
  --watch
```

### Race your final controller yourself

If Level 3 wins, drive against it. This time your keyboard-controlled car is the
challenger:

```text
uv run racing h2h \
  --challenger-keyboard \
  --incumbent-module controllers.level_3 \
  --round-seconds 30 \
  --watch
```

### Level 3 checkpoint

Confirm that:

- `RACING_NAME` is `"Level 3"`;
- neither prohibited forward wall sensor property appears in the program;
- the new strategy reads at least one documented `sensors.camera` property and
  uses its value to decide `steer` or `throttle`;
- Level 3 returns valid throttle and steering commands throughout a race; and
- Level 3 beats Level 2 in the seeded head-to-head race.

## Create your Gradescope submission

After all four levels pass their checkpoints, create one Gradescope-ready ZIP
archive from the `formula110` project folder:

```text
uv run python scripts/export_formula110_exercise.py
```

The command checks that all four required files exist, then creates
`artifacts/formula110-exercise-submission.zip`. Upload that ZIP file to the
Formula 110 exercise assignment on Gradescope. Wait for the autograder to
finish and read the feedback for every level before considering the exercise
complete.

The autograder first checks that the upload does not contain game files under
a `racing` directory. If you upload the entire project, grading stops and asks
you to create and resubmit the ZIP using the instructions above.

The autograder awards separate credit for each level's required programming
ideas and race goal. A problem in one file does not prevent the structural
checks for the other files from running. It begins with a submission
requirements check for each level's docstrings, `__author__`, and function type
annotations.

For each "beats the previous level" check, the autograder allows a 5% tolerance
for variation between computers. Your newer controller earns win credit if
its scored distance is at least 95% of the previous level's scored distance.
For example, 285 meters against 300 meters counts as a win for grading. If
multiple races are graded, this must hold in a majority of them. Two cars
scoring zero do not earn win credit. The Level 1 250-meter checkpoint still
requires 250 meters of forward progress.

Gradescope also awards 5 extra percentage points for submitting at least 48
hours before the assignment deadline, or 3 points for submitting at least 24
hours early. This bonus is based on Gradescope's submission and due-date
timestamps and can raise a complete submission's score above 100.

## Optional leaderboard competition

Completing this exercise makes you eligible for the separate Formula 110
leaderboard competition. Team up with one other person in the course and build
a competition controller together. The competition has its own submission and
autograder which we will announce the week of September 7th; do not replace any
of your four individual exercise files with the team controller. You are
encouraged to start a new controller if you would like to begin experimenting, though!
