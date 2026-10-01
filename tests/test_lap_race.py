from __future__ import annotations

import pytest

from racing.race.laps import LapRace, validate_round_laps
from racing.race.timing import TimingSample


def _update(race: LapRace, time: float, *distances: float, out: tuple[int, ...] = ()) -> None:
    race.update(
        elapsed_seconds=time,
        samples=tuple(TimingSample(str(index), distance, index in out) for index, distance in enumerate(distances)),
    )


def test_start_line_crossing_does_not_count_as_a_completed_lap() -> None:
    race = LapRace(round_laps=2, track_length_m=100)
    _update(race, 0, -5, -10)
    _update(race, 1, 5, 0)
    assert race.clock_text == "LAP 1/2"
    _update(race, 11, 105, 100)
    assert race.clock_text == "LAP 2/2"
    assert race.finish_times == {}
    _update(race, 20, 195, 190)
    _update(race, 21, 205, 200)
    assert race.finish_times == {"0": 20.5, "1": 21}
    assert race.complete
    assert race.clock_text == "FINISH"


def test_finisher_lap_count_does_not_round_down_on_fractional_track_lengths() -> None:
    race = LapRace(round_laps=5, track_length_m=0.1)
    _update(race, 0, 0.49)
    _update(race, 1, 0.51)
    assert race.completed_laps("0") == 5


def test_same_tick_finish_order_uses_crossing_time_not_loop_order_or_overshoot() -> None:
    race = LapRace(round_laps=1, track_length_m=100)
    _update(race, 0, -10, -5, -15)
    _update(race, 10, 90, 99, 80)
    _update(race, 11, 110, 102, 90)
    assert race.finish_position("1") == 1
    assert race.finish_position("0") == 2
    assert race.finish_position("2") is None
    assert race.deadline_seconds == pytest.approx(30 + 1 / 3)
    assert [row.car_id for row in race.rows] == ["1", "0", "2"]
    # A finisher's subsequent crash or reversing cannot change their result.
    _update(race, 12, 50, 0, 99, out=(1,))
    assert not race.complete
    assert [row.car_id for row in race.rows] == ["1", "0", "2"]
    assert not race.rows[0].eliminated
    assert race.rows[1].gap_seconds == pytest.approx(1 / 6)
    _update(race, 13, 60, 0, 101, out=(1,))
    assert race.complete
    assert race.finish_position("2") == 3
    assert race.rows[2].interval_seconds == pytest.approx(2.0)


def test_exact_same_time_finishes_have_deterministic_ordinal_positions() -> None:
    race = LapRace(round_laps=1, track_length_m=100)
    _update(race, 0, 90, 90)
    race.update(elapsed_seconds=1, samples=(TimingSample("1", 110), TimingSample("0", 110)))
    assert race.finish_position("0") == 1
    assert race.finish_position("1") == 2


def test_unfinished_interval_ignores_a_finisher_reversing_between_running_cars() -> None:
    race = LapRace(round_laps=1, track_length_m=100)
    _update(race, 0, 90, 60, 50)
    _update(race, 1, 110, 80, 70)
    _update(race, 2, 75, 90, 80)
    assert [row.car_id for row in race.rows] == ["0", "1", "2"]
    assert race.rows[1].interval_seconds is None
    assert race.rows[2].interval_seconds == pytest.approx(1.0)


@pytest.mark.parametrize("timeout", [5.5, 20.0, 40.0])
def test_timeout_uses_winner_crossing_and_accepts_only_crossings_by_deadline(timeout: float) -> None:
    race = LapRace(round_laps=1, track_length_m=100, finish_timeout_seconds=timeout)
    _update(race, 0, -5, -10, -15, -20)
    _update(race, 10, 95, 10, 10, 10)
    _update(race, 11, 105, 20, 20, 20)
    deadline = 10.5 + timeout
    assert race.deadline_seconds == deadline
    assert race.clock_text == f"{timeout - 0.5:04.1f}s"
    _update(race, deadline - 0.5, 200, 95, 94, 0)
    assert not race.complete
    # Both cars cross within this tick, but only one meets the exact deadline.
    _update(race, deadline + 0.5, 210, 105, 104, 0)
    assert race.finish_times == {"0": 10.5, "1": deadline}
    assert race.dnf == {"2", "3"}
    assert race.complete
    _update(race, deadline + 1.5, 220, 115, 114, 110)
    assert race.finish_times == {"0": 10.5, "1": deadline}


def test_zero_timeout_ends_at_p1_and_excludes_later_crossings_in_the_same_tick() -> None:
    race = LapRace(round_laps=1, track_length_m=100, finish_timeout_seconds=0)
    _update(race, 0, 90, 95)
    _update(race, 1, 105, 110)
    assert race.complete
    assert race.finish_times == {"1": pytest.approx(1 / 3)}
    assert race.dnf == {"0"}
    assert race.deadline_seconds == pytest.approx(1 / 3)


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), float("-inf")])
def test_finish_timeout_requires_a_finite_nonnegative_number(timeout: float) -> None:
    with pytest.raises(ValueError, match="finish_timeout_seconds must be finite and nonnegative"):
        LapRace(round_laps=1, track_length_m=100, finish_timeout_seconds=timeout)


def test_eliminations_do_not_take_a_finishing_place_or_delay_round_end() -> None:
    race = LapRace(round_laps=1, track_length_m=100, finish_timeout_seconds=60)
    _update(race, 0, -5, -10, -15)
    _update(race, 10, 95, 90, 85)
    _update(race, 11, 105, 100, 95, out=(0,))
    assert race.finish_position("1") == 1
    assert race.finish_position("0") is None
    _update(race, 12, 105, 110, 105, out=(0,))
    assert race.complete
    assert race.deadline_seconds is not None and race.elapsed_seconds < race.deadline_seconds
    assert race.finish_position("2") == 2
    assert race.dnf == {"0"}


def test_all_dnf_ends_round_without_a_winner_or_timeout() -> None:
    race = LapRace(round_laps=1, track_length_m=100)
    _update(race, 0, -5, -10)
    _update(race, 1, 0, -5, out=(0, 1))
    assert race.complete
    assert race.finish_times == {}
    assert race.deadline_seconds is None


def test_no_time_limit_before_first_finisher_and_lapped_cars_need_full_distance() -> None:
    race = LapRace(round_laps=2, track_length_m=100)
    _update(race, 0, -5, -10)
    _update(race, 1000, 195, 95)
    assert not race.complete
    assert race.deadline_seconds is None
    _update(race, 1001, 205, 105)
    assert race.finish_position("0") == 1
    assert race.finish_position("1") is None
    _update(race, 1021, 300, 199)
    assert race.complete
    assert race.dnf == {"1"}


def test_teleports_and_backward_crossings_do_not_finish_a_race() -> None:
    race = LapRace(round_laps=1, track_length_m=100)
    _update(race, 0, 95)
    race.update(elapsed_seconds=1, samples=(TimingSample("0", 105, discontinuity=True),))
    assert not race.complete
    _update(race, 2, 95)
    assert not race.complete
    _update(race, 3, 105)
    assert race.finish_times == {"0": 2.5}


@pytest.mark.parametrize("laps", [0, -1, True, 1.5])
def test_lap_limit_requires_a_positive_integer(laps: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        validate_round_laps(laps)
