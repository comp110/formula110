# Formula 110 Leaderboard Metrics

This document collects possible Gradescope leaderboard trophies for a
30-second Formula 110 run. The goal is to reward several distinct controller
strategies instead of reducing every submission to one dominant score.

The grader uses one fixed Mugello Short track whose centerline is approximately
183.1 meters long. Seeds do not select different track shapes or distances;
they only choose different starting offsets around the same closed course. A
strong controller can complete a lap in the grading window, while partial-run
metrics still give developing controllers something meaningful to compare.

## Shared scoring rules

- Run every controller for the full 30 simulated seconds from every configured
  deterministic starting offset.
- Preserve the existing leaderboard qualification rule: every run must finish,
  make positive forward progress, and end with a surviving car below 100%
  damage.
- Marshal a stuck car after 2 seconds, preserve its damage, and apply a 5-meter
  All Spawns, No Crumbs penalty plus a 2-second marshal cooldown. A car beyond the outer
  recovery boundary may be marshaled immediately.
- Measure distance using official forward track progress. Do not use raw
  odometry as progress because odometry also increases while reversing or
  driving in circles.
- Require a completed lap for any metric that can be minimized by parking,
  including brake use, throttle use, steering, g-load, and speed variation.
- Calculate lap trophies from per-lap telemetry. Final-run damage and contact
  totals cannot identify which individual lap was clean or accumulated damage.
- Treat a clean lap as one with no damage increase and no wall-contact time.
  Off-track position does not affect clean-lap eligibility.
- Use forward speed and a rolling window for top-speed metrics. An
  instantaneous absolute maximum can reward reverse motion or a one-tick
  collision spike.
- For start-offset aggregation, prefer the worst-offset value for reliability
  trophies and the mean of each offset's best eligible lap for lap trophies.
  This tests whether the controller can begin on different parts of the same
  circuit; it is not evidence of generalization to other tracks. A lap trophy
  should be omitted from that submission's sortable leaderboard values if any
  required starting offset has no eligible lap. The student-facing report
  should still display the trophy as `N/A`.
- Record enough precision to break legitimate ties, but round the displayed
  Gradescope value to a readable number of decimal places.

## Complete metric catalog

| Category | Trophy | Measurement | Order | Primary tension or guardrail |
| --- | --- | --- | --- | --- |
| Performance | All Spawns, No Crumbs | Worst-starting-offset official forward progress after marshal penalties, expressed in partial laps | Higher | Rewards robust distance from any starting point on the circuit |
| Performance | Clock It | Fastest completed lap with no damage increase or wall contact; off-track position is ignored | Lower | Speed versus contact-free safety |
| Performance | Speedmaxxing | Highest rolling one-second, time-weighted mean of positive forward speed during a completed lap | Higher | Rewards usable speed instead of an instantaneous spike |
| Performance | Opening Quarter | Fastest time to reach 25% of a lap from the seeded spawn | Lower | Gives non-lapping controllers a short sprint target |
| Performance | Sector Star | Fastest individual quarter-lap sector | Lower | Rewards specializing in one section rather than the whole circuit |
| Performance | Rolling Charge | Greatest official progress in any rolling three-second window | Higher | Rewards bursts of speed without requiring a completed lap |
| Performance | Launch Control | Fastest acceleration from 5 m/s to 15 m/s | Lower | Acceleration versus energy use and later braking |
| Damage | Hits Different | Sum of final accumulated damage across all full qualifying runs | Higher | Rewards surviving accumulated damage across every starting offset |
| Damage | Damage Comeback | Most forward progress made after the first damage event | Higher | Rewards recovery after a mistake or deliberate impact |
| Comfort | Sips Tea | Lowest accumulated absolute horizontal g-load during a completed lap | Lower | Comfort versus cornering speed; use magnitude so signed forces cannot cancel |
| Comfort | Gs Going Crazy | Greatest accumulated absolute horizontal g-load during a completed lap with no wall contact | Higher | Directly opposes Sips Tea without rewarding collision impulses |
| Comfort | Grandma's China | Fastest completed lap that never exceeds a configured horizontal g limit | Lower | Forces deliberate speed management |
| Comfort | Cruise Control | Lowest forward-speed variance during a completed lap | Lower | Constant speed versus optimal acceleration and braking |
| Comfort | High-Speed Alignment | Lowest mean absolute heading error while above a minimum speed | Lower | Stable tracking versus aggressive rotation into corners |
| Efficiency | Gas Locked In | Fastest completed lap with no actual brake application | Lower | Coasting and anticipation versus late braking |
| Efficiency | Fuel Sipper | Least propulsive throttle or engine impulse during a completed lap | Lower | Energy efficiency versus acceleration |
| Efficiency | Shortest Route | Least physical odometer distance required to gain one lap of official progress | Lower | Rewards discovering a compact racing line |
| Efficiency | Minimalist | Fewest meaningful throttle or steering changes during a completed lap | Lower | Simple control versus precise control |
| Efficiency | No Hands | Least accumulated absolute steering input during a completed lap | Lower | Steering economy versus corner-following accuracy |
| Efficiency | Keyboard Warrior | Fastest lap using only quantized commands such as `-1`, `0`, and `1` | Lower | Coarse control versus lap speed |
| Precision | On Rails | Lowest distance-weighted absolute centerline error during a completed lap | Lower | Centerline fidelity versus cutting apexes |
| Precision | Metronome | Lowest best-lap-time variation across starting offsets | Lower | Repeatability from different starting points on the same circuit |
| Precision | Valet Parking | After completing a lap, finish the 30-second run closest to the target point and at the lowest speed | Lower | Endpoint precision versus continuing to maximize distance |
| Precision | Beat the Clock | Completed lap time closest to a fixed target such as 20.000 seconds | Lower | Timing precision rather than raw speed |
| Precision | Last of the Late Brakers | Latest braking point before a designated corner without contact or leaving the track | Higher | Bravery versus sufficient stopping distance |
| Precision | Apex Predator | Highest forward speed at a designated corner apex | Higher | Corner-entry commitment versus maintaining control |
| Proximity | Wall Whisperer | Most positive forward progress accumulated within 0.35 m of a wall without contact | Higher | Precision at the boundary versus clean survival |
| Proximity | Wall Rider | Longest continuous wall-contact interval while still gaining forward progress and surviving | Higher | Intentionally opposes Wall Whisperer and Clock It |
| Style | Serving Sideways | Most positive forward distance above 8 m/s with an absolute slip angle between 12 and 45 degrees | Higher | Controlled oversteer versus smoothness and grip |
| Style | Figure Skater | Greatest accumulated absolute yaw rotation while still meeting a minimum forward-progress threshold | Higher | Rotation and spectacle versus efficient distance |
| Style | Reverse Commute | Most reverse distance followed by completion of a forward lap | Higher | Deliberately spends scarce run time going backward |
| Style | Lawnmower | Most off-track time while surviving and completing a lap | Higher | Exploration versus clean, fast driving |
| Style | Scenic Route | Greatest physical distance traveled while completing exactly one official lap | Higher | Directly opposes Shortest Route |
| Partial run | Contact-Free Progress | Greatest official progress reached before the first contact | Higher | Useful clean-driving measure without requiring a lap |
| Partial run | Forward Streak | Longest uninterrupted interval of positive official progress | Higher | Rewards continuous motion and penalizes stalls or reversals |

## Recommended initial leaderboard

The following eight trophies are the recommended starting set. Together they
cover speed, robustness, safety, braking strategy, g-load, and style, and they
create several deliberate conflicts between optimization targets.

| Gradescope field | Definition | Eligibility | Start-offset aggregation | Order |
| --- | --- | --- | --- | --- |
| All Spawns, No Crumbs (Laps) | Minimum across starting offsets of `max(0, best_forward_progress_m - marshal_penalty_m) / track_length_m` | Standard full-run qualification at every starting offset | Worst starting offset | Higher |
| Clock It (s) | Shortest lap with zero damage increase and zero wall contact; off-track position is ignored | At least one clean lap at every starting offset | Mean of each offset's fastest clean lap | Lower |
| Hits Different (damage %) | Sum of each run's final accumulated damage percentage | Every run must finish the full duration with positive progress and a surviving car below 100% damage | Sum across all five starting offsets | Higher |
| Sips Tea (g-s) | Integral over a lap of `hypot(forward_acceleration, lateral_acceleration) / 9.80665` | At least one completed lap at every starting offset | Mean of each offset's lowest-g-load lap | Lower |
| Gs Going Crazy (g-s) | Integral over a lap of `hypot(forward_acceleration, lateral_acceleration) / 9.80665` | At least one completed lap with zero wall contact at every starting offset | Mean of each offset's highest-g-load contact-free lap | Higher |
| Gas Locked In (s) | Shortest completed lap with zero actual brake application | At least one brake-free lap at every starting offset | Mean of each offset's fastest brake-free lap | Lower |
| Serving Sideways (m) | Positive official progress made while on track, contact-free, above 8 m/s, and between 12 and 45 degrees of absolute slip angle | At least one completed lap at every starting offset | Mean of each offset's greatest qualifying drift distance | Higher |
| Speedmaxxing (mph) | Highest rolling one-second, time-weighted mean of positive forward speed contained within a completed lap, converted to miles per hour | At least one completed lap at every starting offset | Mean of each offset's highest sustained speed | Higher |

## Intentional conflicts in the initial set

- Clock It and Hits Different reward opposite clean-lap and full-run damage strategies.
- Sips Tea and Gs Going Crazy are direct opposites, with wall contact excluded
  from Gs Going Crazy so collisions cannot manufacture the score.
- Gas Locked In rewards carrying and managing speed without braking, which can
  conflict with both Clock It and conventional fastest-lap strategies.
- Serving Sideways rewards sustained slip, which generally works against Sips
  Tea and the fastest conventional racing line.
- Speedmaxxing rewards maintaining maximum speed, which can conflict with
  Clock It and Sips Tea when the next corner arrives.

## Telemetry needed by the initial set

At each fixed 60 Hz physics tick, retain the applied command, actual brake
state, forward speed, forward and lateral acceleration, official progress
delta, projected track position, wall-contact state, damage, and vehicle world
velocity and heading. Record the cumulative values at each lap boundary and
reset the lap-local accumulators.

Slip angle should be calculated as the wrapped angle between the vehicle's
world-space horizontal velocity vector and its heading. Ignore slip samples at
low speed, where the velocity direction becomes numerically unstable.

Prefer the actual brake force when the trusted physics runtime exposes it. If a
command-based approximation is necessary, classify a throttle request opposing
the current direction of travel as braking. Use a small numerical epsilon when
testing the trusted brake force for zero so floating-point residue does not
disqualify an otherwise brake-free lap.
