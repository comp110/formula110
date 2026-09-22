# Formula 110 progression exercise autograder

This is the Gradescope grader for the four individual exercise files described
in `src/controllers/instructions.md`. It is separate from
`autograder/gradescope`, which remains available for the team competition and
leaderboard.

Build the upload-ready exercise autograder with:

```bash
uv run python scripts/build_formula110_exercise_autograder.py
```

The command writes
`artifacts/formula110-exercise-autograder.zip`. Upload that ZIP as the
Gradescope autograder. The archive contains root-level `setup.sh` and
`run_autograder` files as Gradescope requires.

## Required submission files

Before any level checks or controller execution, the grader rejects uploads
containing files beneath a directory named `racing`, including nested full
project uploads such as `formula110/src/racing/`. It writes a zero-score result
and directs students to the project write-up's **Create your Gradescope
submission** instructions to generate and resubmit the official exercise ZIP.

The exercise grader expects these modules:

```text
controllers.level_0
controllers.level_1
controllers.level_2
controllers.level_3
```

It accepts preserved paths such as `controllers/level_0.py` and
`src/controllers/level_0.py`. It also accepts an unambiguous flattened upload
of the four Python files. Students should normally build the submission ZIP
with:

```bash
uv run python scripts/export_formula110_exercise.py
```

## Rubric and early-submission bonus

Each level is worth 25 base points, for 100 points total. Gradescope metadata
adds 5 extra percentage points when the submission is at least 48 hours early,
or 3 points when it is at least 24 hours early. Scores are rounded half-up to
one decimal place, with 99.9 promoted to 100.0, matching the earlier COMP110
autograder conventions.

The first four checks apply the common submission requirements independently to
each level: a non-empty module docstring first, a module-level `__author__`
string containing exactly nine digits, the exact typed shape
`control(RobotSensors) -> RobotCommand`, a non-empty `control` docstring, and
annotations on every function parameter and return type. The structural
checks use Python's syntax tree, so comments and similarly named plain text do
not satisfy them. Level 2's throttle calculation is checked by its returned
values, accepting temporary variables and algebraically equivalent expressions.
Student code is executed only by the same unprivileged,
resource-limited controller process used by the separate competition grader.

Levels 0 through 2 accept any names for the variables used for throttle and
steering. Positional, keyword, or mixed `RobotCommand` arguments are accepted,
such as `RobotCommand(power, turn)` and `RobotCommand(power, steer=turn)`.
The sensor parameter's name may also differ from `sensors`.

| Level | Check | Points |
| --- | --- | ---: |
| 0 | Common submission requirements | 2 |
| 0 | Variables, `RobotCommand`, conditional, forward wall sensors, and steering behavior | 6 |
| 0 | Valid `control(RobotSensors) -> RobotCommand` behavior | 4 |
| 0 | At least one completed lap in a 60-second seed-110 solo trial | 13 |
| 1 | Common submission requirements | 2 |
| 1 | Level 0 steering plus a conditional comparison of `odometry.speed_mps` | 5 |
| 1 | Valid control behavior | 4 |
| 1 | At least 250 meters in the seeded head-to-head round | 7 |
| 1 | Majority win over Level 0 | 7 |
| 2 | Common submission requirements | 2 |
| 2 | Level 0 steering plus proportional throttle behavior | 6 |
| 2 | Valid control behavior | 4 |
| 2 | Majority win over Level 1 | 13 |
| 3 | Common submission requirements | 2 |
| 3 | No forward wall sensors and at least one documented camera property | 8 |
| 3 | Valid control behavior | 4 |
| 3 | Majority win over Level 2 | 11 |

The Level 0 behavior check calls `control` with four combinations of near/far
forward wall readings at three speeds. It checks steering direction, straight
steering when walls are clear, left-wall priority when both are close, and
positive forward throttle. The near/far readings are `0.5` and `100.0` meters;
exact threshold distances, steering strengths, and throttle strengths are not
prescribed by this check. The static variables check verifies that two assigned
variables supply the command; their spelling does not determine their role.

The Level 2 throttle check calls `control` at the four speeds in the handout
(`0.0`, `7.5`, `15.0`, and `22.5` m/s) and five intermediate speeds, each with
four combinations of near/far forward wall readings. Returned throttle must
match `(15.0 - speed) / 15.0` within a floating-point tolerance of `1e-6`.
Failures identify the sensor inputs and returned throttle, then refer students
to the handout without revealing the target expression or expected output.

Head-to-head checks use the same 30-second round with seed `110` shown in the
student instructions, one car per side, a `1.0` meter win margin, and
deterministic marshal recovery. Race scoring uses the simulator's normal
scored-distance rules; the 250-meter milestone uses raw forward track
progress. For the three "beats the previous level" grading checks, a challenger
earns win credit when its scored distance is at least 95% of the incumbent's
scored distance, including the exact boundary. For example, 285 meters against
300 meters counts as a win. This 5% tolerance accommodates variation between
student computers and the grading environment. Both cars scoring zero does
not earn win credit.

Win credit is calculated separately for each race and requires a majority of
races. Feedback shows the simulator result and the grading decision with the
tolerance applied. The simulator's winner classification and the Level 1
250-meter milestone retain their existing rules. Increase `race_count` before
building if a longer majority-of-races evaluation is preferred, or adjust
`win_tolerance_fraction` to change the grading tolerance.

These parameters and every point value are visible in `build_config()` in
`scripts/build_formula110_exercise_autograder.py` and can be changed before
building the Gradescope archive.

## Security and reliability

The bundle copies the trusted simulator at build time. On Gradescope, the
simulator and grading files are installed read-only under `/opt`, the submitted
files are made read-only, and every student controller runs as an unprivileged
user behind the existing command timeout and 512 MiB process-tree memory
limit. The driver writes an initial `results.json` before starting any check and
reports failures separately for each level.

Before release, test the built archive in Gradescope with a known passing
submission and submissions containing a missing level, syntax error, invalid
return value, infinite loop, copied Level 3 strategy, and a controller that
loses each required race.
