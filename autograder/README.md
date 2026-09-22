# Formula 110 competition Gradescope autograder

This directory contains the competition and leaderboard grader. The separate
four-level individual exercise grader is documented in
[`exercise/README.md`](exercise/README.md); both graders are intentionally kept
in the repository.

Build an upload-ready archive by naming the two required student modules:

```bash
uv run python scripts/build_gradescope_autograder.py \
  controllers.minimum_viable \
  controllers.race_faster
```

The script prints the archive path. By default it writes
`artifacts/formula110-gradescope-autograder.zip`. Use `--output PATH` to choose
another destination. Upload the zip itself to Gradescope; `setup.sh` and
`run_autograder` are at the archive root as Gradescope requires.

## Submission layouts

The autograder accepts preserved package paths such as
`controllers/minimum_viable.py` and `src/controllers/minimum_viable.py`. It also
accepts Gradescope's flattened single-file upload when the matching basename is
unambiguous. Both configured modules must be present.

## Rubric (100 points)

| Check | Points |
| --- | ---: |
| Pyright strict mode, minimum module | 5 |
| Pyright strict mode, improved module | 5 |
| Ruff default lint rules, minimum module | 2.5 |
| Ruff default lint rules, improved module | 2.5 |
| Ruff default formatter, minimum module | 2.5 |
| Ruff default formatter, improved module | 2.5 |
| Valid `control` callable, minimum module | 5 |
| Valid `control` callable, improved module | 5 |
| Minimum module completes at least one lap on both seeds | 20 |
| Minimum module finishes 30 seconds with exactly zero damage on both seeds | 15 |
| Minimum module has zero wall-contact time on both seeds | 15 |
| Improved module finishes 30 seconds without elimination on both seeds | 10 |
| Improved module travels strictly farther than the minimum module on each seed | 10 |

The deterministic seeds are 110 and 2026. Race trials run for 30 simulated
seconds at 60 Hz with marshal recovery disabled. The minimum controller's
no-damage check requires a final damage value of zero. “Survives” for the
improved controller means the car was not eliminated and ended below 100%
damage. The progress comparison uses raw forward track progress. Damage is
reported separately and never multiplies the distance score.

Every check is module-specific. A submission containing only the minimum module
still receives Pyright, Ruff, callable, and seeded race feedback for that file
and can earn all 65 minimum-module points. The missing improved module receives
zero on its 35 points and is not eligible for the leaderboard.

Ruff runs with `--isolated`, so “defaults” means Ruff's own default lint and
format configuration, independent of this repository's instructor settings.
Pyright uses strict type-checking with Python 3.11 and the trusted simulator's
typed package available on its import path.

## Leaderboard

Leaderboard metrics come from the improved module and average its runs on
seeds 110 and 2026:

- Laps (partial): forward progress divided by track length; descending.
- Top speed in meters per second; descending.
- First lap time in seconds; ascending.
- Best completed lap time in seconds; ascending.

If the improved controller is eliminated or reaches 100% damage on either run,
the submission is disqualified and the result exports no leaderboard values.
If it survives both runs but fails to complete a lap on either seed, lap-time
fields display `No lap` while distance and speed are still reported.

## Exporting a student submission

Students can export only the required controller modules—without the simulator,
tests, notebooks, or other repository files—with:

```bash
uv run python scripts/export_student_controllers.py \
  controllers.minimum_viable \
  controllers.race_faster
```

The default output is `artifacts/formula110-student-controllers.zip`. Its files
are rooted at `controllers/`, which the autograder accepts directly. Local
`controllers.*` imports are followed recursively, so Python helper modules are
included automatically. A student who has completed only the minimum milestone
can export only that module:

```bash
uv run python scripts/export_student_controllers.py controllers.minimum_viable
```

For dynamically loaded modules or non-Python data files, export the complete
directory:

```bash
uv run python scripts/export_student_controllers.py --all-controllers
```

Enable leaderboards in the Gradescope assignment settings. Use the current
Ubuntu 22.04 base image. `setup.sh` installs an isolated Python 3.11 runtime,
Panda3D/Ursina, Pyright, and Ruff. It also installs the simulator source bundled
at build time as a read-only trusted package.

## Reliability and security choices

The grader writes an initial `results.json` before doing any work and replaces
it as checks complete. Student controllers execute in separate, timed,
resource-limited subprocesses as an unprivileged user. Inference runs in a
CPU-only environment, and the controller worker is stopped if its resident
process-tree memory exceeds 512 MiB. Autograder source and results are root-only,
and the student's submission is made read-only before execution. Each
module/seed pair gets a fresh Python process so controller global state cannot
leak between seeded trials.

When debugging through Gradescope SSH, run `/autograder/run_autograder`, then
inspect `/autograder/results/results.json`. The launcher prints those two stages
to the terminal. Controller startup failures include the unprivileged process's
exit code and startup diagnostic in the corresponding rubric item.

Before release, use Gradescope's **Test Autograder** workflow with at least one
known passing submission and submissions with missing modules, type errors,
format errors, exceptions, infinite loops, wall contact, and race-ending
damage. Headless physics can vary if the dependency versions or base image are
changed; this bundle pins all simulator and checking dependencies.
