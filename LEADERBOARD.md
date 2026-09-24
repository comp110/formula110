# Formula 110 Leaderboard

Your controller races for 30 simulated seconds from five different starting
offsets on the same track. The leaderboard rewards different styles, so there
is no single controller that should dominate every trophy.

## Qualification

To appear on the leaderboard, your controller must do all of the following at
every starting offset:

- Run for the full 30 simulated seconds.
- Make positive official forward track progress.
- Survive and finish below 100% damage.

All Spawns, No Crumbs does not require a completed lap. Every other trophy requires at
least one eligible completed lap at all five starting offsets. If the overall
qualification passes but one offset has no eligible lap for a trophy, that
trophy displays `N/A` in the submission report and is omitted from that
submission's sortable leaderboard values. The controller still appears through
All Spawns, No Crumbs and any other available run-level metrics.

## Metrics

| Name | Short and fun English description | Definition | Eligibility | Aggregation | Order |
| --- | --- | --- | --- | --- | --- |
| All Spawns, No Crumbs (Laps) | Every spawn pulls up, every run eats, and the track gets left with zero crumbs. | Penalty-adjusted official forward progress divided by the fixed track length. Each marshal recovery subtracts 5 meters, with the result floored at zero. | Overall leaderboard qualification; no completed lap required. | Take the minimum scored partial-lap progress across the five starting offsets. | Higher is better. |
| Clock It (s) | Clock the fastest squeaky-clean lap: no damage, no wall drama, just receipts. | Elapsed time for a completed lap with zero damage increase and zero wall-contact time. Going off track does not affect this trophy. | At least one clean completed lap at every starting offset. | Find the fastest clean lap at each offset, then average those five times. | Lower is better. |
| Hits Different (damage %) | Finish every full run alive while the combined damage meter enters its absolutely unhinged era. | Sum of the final accumulated damage from every full qualifying run, expressed in percentage points. | Every run must finish the full duration with positive progress and a surviving car below 100% damage. | Add the final damage percentages from all five starting offsets. | Higher is better. |
| Sips Tea (g-s) | Serve the calmest completed lap while the car stays unbothered and sips tea. | The lap integral of absolute horizontal g-load: `hypot(forward acceleration, lateral acceleration) / 9.80665`, accumulated once per physics step. | At least one completed lap at every starting offset. | Find the lowest-g-load lap at each offset, then average those five values. | Lower is better. |
| Gs Going Crazy (g-s) | Send the G-meter into its main-character era without touching a wall. | The same accumulated absolute horizontal g-load used by Sips Tea. | At least one completed lap with zero wall-contact time at every starting offset. | Find the highest-g-load contact-free lap at each offset, then average those five values. | Higher is better. |
| Gas Locked In (s) | Fastest brake-free lap: gas committed, panic pedal deleted, absolutely locked in. | Elapsed time for a completed lap during which the trusted physics engine never applies brake force. Coasting does not count as braking. | At least one brake-free completed lap at every starting offset. | Find the fastest brake-free lap at each offset, then average those five times. | Lower is better. |
| Serving Sideways (m) | Serve maximum sideways energy while still making actual forward progress. | Positive official progress accumulated while forward speed is above 8 m/s, the car is on track and out of wall contact, and absolute slip angle is from 12° through 45°. | At least one completed lap at every starting offset; an eligible lap may score zero drift distance. | Find the completed lap with the greatest drift distance at each offset, then average those five distances. | Higher is better. |
| Speedmaxxing (mph) | Hold peak speed for a full second and make the speedometer question reality. | Highest rolling one-second, time-weighted mean of positive forward speed contained within a completed lap, converted to miles per hour. | At least one completed lap at every starting offset. | Find the completed lap with the highest sustained speed at each offset, then average those five speeds. | Higher is better. |

## Measurement notes

- The five seeds only change the starting offset. Track shape and lap distance
  are identical in every run.
- A car that remains stuck for 2 seconds is marshaled back onto the track. The
  reset preserves damage, subtracts 5 meters from All Spawns, No Crumbs progress, and has a
  2-second cooldown. A car beyond the track's outer recovery boundary can be
  marshaled immediately.
- Official progress follows the track centerline. Raw odometry is not used
  because it also increases while reversing or driving in circles.
- Wall contact, damage, brake force, g-load, and slip angle are measured by the
  trusted simulator rather than reported by student code.
- A negative throttle request while moving forward normally resolves to brake
  force and therefore disqualifies that lap from Gas Locked In. Passive slowing
  and coasting are allowed.
- Horizontal g-load uses magnitude, so left/right turns and acceleration/
  deceleration cannot cancel one another.
- Drift distance counts only positive progress made during qualifying drift
  samples; spinning in place does not add distance.
- Speedmaxxing averages each one-second window, so a one-tick collision spike does
  not win the trophy.
