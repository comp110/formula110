# Formula 110 competition Gradescope autograder

This directory contains the competition and leaderboard grader. The separate
four-level individual exercise grader is documented in
[`exercise/README.md`](exercise/README.md); both graders remain available.

Build the upload-ready autograder archive with:

```bash
uv run python scripts/build_gradescope_autograder.py
```

The script prints the archive path. By default it writes
`artifacts/formula110-gradescope-autograder.zip`. Use `--output PATH` to choose
another destination. Upload the zip itself to Gradescope; `setup.sh` and
`run_autograder` are at the archive root as Gradescope requires.

## Submission contract

Each submission selects exactly one controller when it is exported:

```bash
uv run python scripts/export_student_controllers.py controllers.race_faster
```

The default output is `artifacts/formula110-student-controllers.zip`. It
contains:

- `formula110-submission.json`, naming the one `controllers.*` module to grade.
- The complete `src/controllers/` tree rooted at `controllers/`, including
  checkpoints and other runtime assets.
- `pyproject.toml` and `uv.lock` for runtime dependency installation.

Generated `__pycache__` directories, `.pyc` files, and `py.typed` markers are
omitted. Other project content—such as the simulator, tests, notebooks, and
virtual environments—is not included.

The grader reads the manifest and evaluates only its selected controller. Other
Python modules in `controllers/` remain available as helpers but are never
implicitly selected. Before loading the controller, the grader syncs its
`pyproject.toml` runtime dependencies, honoring `uv.lock` when present.
Development dependency groups and installation of the submitted project itself
are skipped.

Invalid submission feedback directs students to **EX99 - Formula110 Qualifying**
on **Course AI > Resources > Exercises** for the submission instructions.

## Grading and leaderboard

The selected controller runs for 30 simulated seconds on deterministic seeds
110, 2026, 1893, 7656, and 9340. Every seed uses the same track shape and
distance; only the starting offset changes. The five starts cover the five
equal-length fifths of the lap:

| Seed | Centerline progress | Approximate landmark |
| ---: | ---: | --- |
| 110 | 4.95% | Start straight |
| 1893 | 28.56% | San Donato exit toward Luco |
| 2026 | 54.30% | Materassi exit |
| 7656 | 65.27% | Borgo crest toward Casanova approach |
| 9340 | 81.31% | Return Bend |

The single 100-point check passes when **All Spawns, No Crumbs >= 2.0**;
otherwise it awards 0 points. This is the minimum qualification for the F110
event. The metric is the lowest penalty-adjusted lap progress across all five
starting offsets, so every run must reach at least 2.0 laps within 30 simulated
seconds. Every run must finish the full duration without controller errors,
elimination, or reaching 100% damage. Minor damage or wall contact is allowed.
Qualification uses unrounded progress, not the rounded leaderboard display.
The qualification result and explanation appear in both the overall feedback
and the scored test.

Leaderboard eligibility and metric calculations are unchanged. Eligible
controllers below 2.0 still receive their leaderboard entries even though they
earn 0 points and do not meet the event qualification minimum.

Marshal recovery is enabled. A car that remains stuck for 2 seconds is reset
onto the track with its damage preserved, a 5-meter scored-distance penalty,
and a 2-second recovery cooldown. Crossing the outer off-track recovery
boundary can trigger an immediate reset. Raw progress remains available in the
trial diagnostics, while All Spawns, No Crumbs uses penalty-adjusted progress.

The leaderboard reports the recommended initial trophy set:

- All Spawns, No Crumbs: worst starting offset's penalty-adjusted partial-lap progress;
  descending.
- Clock It: mean fastest lap with no damage or wall contact; ascending.
- Hits Different: total final damage accumulated across all five completed runs;
  descending.
- Sips Tea: mean lowest accumulated horizontal g-load; ascending.
- Gs Going Crazy: mean highest accumulated horizontal g-load on a lap without
  wall contact; descending.
- Gas Locked In: mean fastest completed lap with no actual brake application;
  ascending.
- Serving Sideways: mean greatest qualifying drift distance in a completed lap;
  descending.
- Speedmaxxing: mean highest rolling one-second forward speed contained in a
  completed lap, reported in mph; descending.

The student-facing report prints every leaderboard metric, using `N/A` when a
metric is unavailable for that submission. Unavailable metrics are omitted
from the Gradescope `leaderboard` array so every emitted sortable value remains
numeric; qualifying controllers still appear through their available metrics.

A controller is disqualified from the leaderboard if a run fails, makes no
forward progress, ends early, or the car is eliminated or reaches 100% damage.
If every run qualifies but one starting offset has no eligible lap for a
trophy, that field displays `N/A` in the submission report and is omitted from
the sortable leaderboard values. All Spawns, No Crumbs is still reported, so a
qualifying controller without a completed lap still appears on the leaderboard.

## Runtime and operations

Enable leaderboards in the Gradescope assignment settings and use the current
Ubuntu 22.04 base image. `setup.sh` installs an isolated Python 3.11 runtime,
Panda3D/Ursina, and the simulator source bundled at build time as a read-only
trusted package. Media assets are omitted because grading is headless. Runtime
dependency syncing preserves the grader's installed packages.

The grader writes an initial `results.json` before doing any work and replaces
it as checks complete. Student controllers execute in separate, timed,
resource-limited subprocesses as an unprivileged user. Inference runs in a
CPU-only environment, and the controller worker is stopped if its resident
process-tree memory exceeds 1.5 GiB. Autograder source and results are root-only,
and the student's submission is made read-only before execution. Each seeded
trial gets a fresh Python process so controller state cannot leak between runs.
Controller imports, factory construction, and checkpoint loading have a separate
30-second startup budget; after readiness, every control response retains the
0.5-second per-tick deadline.

When debugging through Gradescope SSH, run `/autograder/run_autograder`, then
inspect `/autograder/results/results.json`. Controller startup failures include
the unprivileged process's exit code and startup diagnostic.

Before release, use Gradescope's **Test Autograder** workflow with known passing
and failing submissions, including a missing or invalid manifest, a missing
target module, dependency failures, exceptions, infinite loops, zero-progress
runs, and race-ending damage. Headless physics can vary if dependency versions
or the base image change; this bundle pins the simulator dependencies.
