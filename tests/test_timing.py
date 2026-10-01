from __future__ import annotations

import math

import pytest

from racing.race.timing import RaceTiming, TimingSample, TimingStanding


def _samples(**distances: float) -> tuple[TimingSample, ...]:
    return tuple(TimingSample(car_id, distance) for car_id, distance in distances.items())


def _by_id(rows: tuple[TimingStanding, ...]) -> dict[str, TimingStanding]:
    return {row.car_id: row for row in rows}


def test_gap_uses_leader_crossing_history_instead_of_current_speed() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(leader=0.0, follower=0.0))
    timing.update(1.0, _samples(leader=10.0, follower=1.0))

    rows = timing.update(3.0, _samples(leader=50.0, follower=5.0))

    assert rows[0] == TimingStanding(1, "leader", 50.0, 0.0)
    assert rows[1].gap_seconds == pytest.approx(2.5)
    assert timing.rows == rows


def test_interval_uses_the_car_immediately_ahead_and_updates_after_a_pass() -> None:
    timing = RaceTiming()
    timing.update(0, _samples(leader=0, second=0, third=0))
    timing.update(1, _samples(leader=20, second=10, third=4))
    rows = _by_id(timing.update(2, _samples(leader=60, second=40, third=12)))
    assert rows["leader"].interval_seconds is None
    assert rows["second"].interval_seconds == pytest.approx(0.5)
    assert rows["third"].interval_seconds == pytest.approx(1 - 2 / 30)
    assert rows["third"].gap_seconds == pytest.approx(1.4)

    rows = _by_id(timing.update(3, _samples(leader=80, second=42, third=50)))
    assert rows["third"].rank == 2
    assert rows["second"].interval_seconds == pytest.approx(1 - 30 / 38)


def test_interval_is_unknown_after_a_marshal_until_the_ahead_car_rebuilds_history() -> None:
    timing = RaceTiming()
    timing.update(0, _samples(leader=0, second=0, third=0))
    timing.update(1, _samples(leader=30, second=20, third=10))
    rows = timing.update(2, (TimingSample("leader", 60), TimingSample("second", 40, discontinuity=True),
                             TimingSample("third", 20)))
    assert rows[2].gap_seconds is not None
    assert rows[2].interval_seconds is None
    rows = timing.update(3, _samples(leader=90, second=60, third=50))
    assert rows[2].interval_seconds == pytest.approx(0.5)


def test_stationary_follower_gap_keeps_growing_after_leader_passes() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(leader=0.0, follower=0.0))
    timing.update(1.0, _samples(leader=10.0, follower=5.0))

    first = _by_id(timing.update(10.0, _samples(leader=10.0, follower=5.0)))
    second = _by_id(timing.update(20.0, _samples(leader=10.0, follower=5.0)))

    assert first["follower"].gap_seconds == pytest.approx(9.5)
    assert second["follower"].gap_seconds == pytest.approx(19.5)


def test_plateau_retains_first_crossing_but_interpolates_restart_from_latest_sample() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(leader=0.0, at_stop=0.0, past_stop=0.0))
    timing.update(1.0, _samples(leader=10.0, at_stop=0.0, past_stop=0.0))
    timing.update(5.0, _samples(leader=10.0, at_stop=1.0, past_stop=2.0))

    rows = _by_id(timing.update(6.0, _samples(leader=20.0, at_stop=10.0, past_stop=15.0)))

    assert rows["at_stop"].gap_seconds == pytest.approx(5.0)
    assert rows["past_stop"].gap_seconds == pytest.approx(0.5)


def test_overtake_changes_gap_source_to_new_leaders_own_history() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(1.0, _samples(a=10.0, b=8.0))

    overtaken = timing.update(2.0, _samples(a=12.0, b=20.0))
    regained = timing.update(3.0, _samples(a=30.0, b=25.0))

    assert [row.car_id for row in overtaken] == ["b", "a"]
    assert overtaken[1].gap_seconds == pytest.approx(2.0 / 3.0)
    assert [row.car_id for row in regained] == ["a", "b"]
    assert regained[1].gap_seconds == pytest.approx(5.0 / 18.0)


def test_equal_distances_use_stable_ids_and_zero_gap_even_after_long_stop() -> None:
    timing = RaceTiming()
    first = timing.update(0.0, _samples(z=0.0, a=0.0, m=0.0))
    later = timing.update(100.0, _samples(m=0.0, z=0.0, a=0.0))

    assert [row.car_id for row in first] == ["a", "m", "z"]
    assert [row.car_id for row in later] == ["a", "m", "z"]
    assert [row.rank for row in later] == [1, 2, 3]
    assert all(row.gap_seconds == 0.0 for row in later)


def test_gap_remains_cumulative_when_separated_by_more_than_a_lap() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(leader=0.0, follower=0.0))
    timing.update(10.0, _samples(leader=100.0, follower=50.0))

    rows = timing.update(20.0, _samples(leader=250.0, follower=70.0))

    assert rows[1].distance_m == 70.0
    assert rows[1].gap_seconds == pytest.approx(13.0)


def test_missing_history_stays_unknown_until_new_samples_cover_follower_score() -> None:
    timing = RaceTiming()
    unknown = timing.update(10.0, _samples(a=100.0, b=20.0))
    covered = timing.update(11.0, _samples(a=110.0, b=105.0))

    assert unknown[0].gap_seconds == 0.0
    assert unknown[1].gap_seconds is None
    assert covered[1].gap_seconds == pytest.approx(0.5)


def test_score_decrease_discards_old_crossings_before_rebuilding_gap() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(5.0, _samples(a=50.0, b=40.0))
    penalized = timing.update(6.0, _samples(a=45.0, b=41.0))
    rebuilt = timing.update(7.0, _samples(a=55.0, b=50.0))

    assert penalized[1].gap_seconds is None
    assert rebuilt[1].gap_seconds == pytest.approx(0.5)


@pytest.mark.parametrize("new_distance", [50.0, 80.0])
def test_explicit_marshal_discontinuity_drops_history_even_without_score_decrease(new_distance: float) -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(5.0, _samples(a=50.0, b=20.0))

    rows = timing.update(6.0, (TimingSample("a", new_distance, discontinuity=True), TimingSample("b", 30.0)))

    assert rows[0].car_id == "a"
    assert rows[1].gap_seconds is None


def test_trailing_car_history_is_also_reset_before_it_takes_the_lead() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(5.0, _samples(a=35.0, b=40.0))
    timing.update(6.0, _samples(a=36.0, b=30.0))

    rows = timing.update(7.0, _samples(a=38.0, b=70.0))

    assert rows[0].car_id == "b"
    assert rows[1].gap_seconds == pytest.approx(0.8)


def test_bounded_history_does_not_extrapolate_into_discarded_crossings() -> None:
    timing = RaceTiming(max_history_points=3)
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(1.0, _samples(a=10.0, b=1.0))
    timing.update(2.0, _samples(a=20.0, b=2.0))
    rows = timing.update(3.0, _samples(a=30.0, b=5.0))

    assert rows[1].gap_seconds is None
    timing.update(100.0, _samples(a=30.0, b=10.0))
    rows = timing.update(101.0, _samples(a=40.0, b=30.0))
    assert rows[1].gap_seconds == pytest.approx(98.0)
    rows = timing.update(102.0, _samples(a=50.0, b=35.0))
    assert rows[1].gap_seconds == pytest.approx(1.5)


def test_reappearing_car_cannot_reuse_history_from_unobserved_interval() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(5.0, _samples(a=50.0, b=40.0))
    timing.update(6.0, _samples(b=60.0))

    rows = timing.update(7.0, _samples(a=100.0, b=80.0))

    assert rows[0].car_id == "a"
    assert rows[1].gap_seconds is None


def test_eliminated_leader_moves_below_active_cars_and_promotes_new_leader() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))

    rows = timing.update(5.0, (TimingSample("a", 50.0, eliminated=True), TimingSample("b", 40.0)))

    assert rows[0].car_id == "b"
    assert not rows[0].eliminated
    assert rows[0].gap_seconds == 0.0
    assert rows[1].car_id == "a"
    assert rows[1].rank == 2
    assert rows[1].eliminated
    assert rows[1].gap_seconds is None


def test_retirees_follow_active_cars_in_reverse_retirement_order() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=50.0, b=40.0, c=30.0, d=20.0))
    timing.update(1.0, (TimingSample("a", 100.0, eliminated=True), *_samples(b=40.0, c=30.0, d=20.0)))
    rows = timing.update(
        2.0,
        (
            TimingSample("a", 100.0, eliminated=True),
            TimingSample("b", 40.0),
            TimingSample("c", 30.0, eliminated=True),
            TimingSample("d", 20.0),
        ),
    )
    assert [row.car_id for row in rows] == ["b", "d", "c", "a"]
    assert [row.rank for row in rows] == [1, 2, 3, 4]
    all_out = timing.update(3.0, tuple(TimingSample(row.car_id, row.distance_m, eliminated=True) for row in rows))
    # Same-tick retirements tie by stable ID; both precede older retirees.
    assert [row.car_id for row in all_out] == ["b", "d", "c", "a"]
    assert all(row.gap_seconds is None for row in all_out)
    assert timing.update(10.0, tuple(TimingSample(row.car_id, 1000.0, eliminated=True) for row in all_out)) == tuple(
        TimingStanding(index, row.car_id, 1000.0, None, eliminated=True) for index, row in enumerate(all_out, start=1)
    )
    timing.reset()
    restarted = timing.update(0.0, _samples(a=-4.0, b=-8.0, c=-12.0, d=-16.0))
    assert [row.car_id for row in restarted] == ["a", "b", "c", "d"]
    assert all(not row.eliminated for row in restarted)


def test_reset_allows_new_race_time_and_discards_previous_standings() -> None:
    timing = RaceTiming()
    timing.update(10.0, _samples(a=100.0, b=50.0))

    timing.reset()

    assert timing.rows == ()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    rows = timing.update(1.0, _samples(a=50.0, b=30.0))
    assert rows[1].gap_seconds == pytest.approx(0.4)


def test_empty_samples_remove_standings_and_old_histories() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    assert timing.update(1.0, ()) == ()

    rows = timing.update(2.0, _samples(a=20.0, b=10.0))

    assert rows[1].gap_seconds is None


def test_repeated_physics_snapshot_does_not_shift_first_crossing_timestamp() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    first = timing.update(1.0, _samples(a=10.0, b=5.0))
    assert timing.update(1.0, _samples(a=10.0, b=5.0)) == first

    rows = timing.update(2.0, _samples(a=20.0, b=10.0))

    assert rows[1].gap_seconds == pytest.approx(1.0)


def test_changed_score_at_same_timestamp_is_not_interpolated_as_motion() -> None:
    timing = RaceTiming()
    timing.update(0.0, _samples(a=0.0, b=0.0))
    timing.update(1.0, _samples(a=10.0, b=5.0))

    rows = timing.update(1.0, _samples(a=20.0, b=6.0))

    assert rows[1].gap_seconds is None


@pytest.mark.parametrize("invalid_time", [-1.0, math.inf, math.nan])
def test_invalid_timestamps_leave_existing_rows_unchanged(invalid_time: float) -> None:
    timing = RaceTiming()
    rows = timing.update(0.0, _samples(a=0.0))
    with pytest.raises(ValueError, match="elapsed_seconds"):
        timing.update(invalid_time, _samples(a=10.0))
    assert timing.rows == rows


@pytest.mark.parametrize("invalid_distance", [math.inf, -math.inf, math.nan])
def test_invalid_distances_leave_existing_rows_unchanged(invalid_distance: float) -> None:
    timing = RaceTiming()
    rows = timing.update(0.0, _samples(a=0.0))
    with pytest.raises(ValueError, match="distance_m"):
        timing.update(1.0, _samples(a=invalid_distance))
    assert timing.rows == rows


def test_backward_time_duplicate_ids_and_empty_ids_are_rejected() -> None:
    timing = RaceTiming()
    timing.update(1.0, _samples(a=0.0))
    with pytest.raises(ValueError, match="reset timing"):
        timing.update(0.0, _samples(a=0.0))
    with pytest.raises(ValueError, match="duplicate"):
        timing.update(2.0, (TimingSample("a", 0.0), TimingSample("a", 1.0)))
    with pytest.raises(ValueError, match="nonempty"):
        timing.update(2.0, (TimingSample("", 0.0),))


def test_history_requires_enough_points_for_interpolation() -> None:
    with pytest.raises(ValueError, match="at least two"):
        RaceTiming(max_history_points=1)


def test_grid_positions_before_the_line_keep_order_until_an_actual_pass() -> None:
    timing = RaceTiming()
    grid = timing.update(0.0, _samples(pole=-10.0, second=-15.0, third=-20.0, fourth=-25.0))
    assert [row.car_id for row in grid] == ["pole", "second", "third", "fourth"]
    # Second travels farther, but has not made up its grid deficit.
    rows = timing.update(1.0, _samples(pole=-9.0, second=-11.0, third=-15.0, fourth=-20.0))
    assert rows[0].car_id == "pole"
    rows = timing.update(2.0, _samples(pole=-8.0, second=-7.0, third=-10.0, fourth=-15.0))
    assert rows[0].car_id == "second"
    assert rows[1].gap_seconds == pytest.approx(0.25)
