from __future__ import annotations

import pytest

from racing.race.heat import HeatRaceSnapshot
from racing.race.interest import InterestWeights, RaceInterestTracker, score_interest
from racing.race.timing import TimingSample


def snapshot(
    time: float,
    distances: tuple[float, ...],
    *,
    eliminated: str = "",
    recovered: str = "",
    finished: tuple[str, ...] = (),
) -> HeatRaceSnapshot:
    return HeatRaceSnapshot(
        race_index=1,
        elapsed_seconds=time,
        samples=tuple(
            TimingSample(name, distance, eliminated=name == eliminated, discontinuity=name == recovered)
            for name, distance in zip("ABCD", distances, strict=True)
        ),
        finished_car_ids=frozenset(finished),
    )


def test_sustained_pass_for_lead_counts_one_pass_and_one_lead_change() -> None:
    tracker = RaceInterestTracker()
    tracker.update(snapshot(0, (10, 8, 4, 2)))
    tracker.update(snapshot(0.1, (11, 12, 5, 3)))
    assert not tracker.events
    tracker.update(snapshot(0.4, (12, 14, 6, 4)))
    assert tracker.overtakes == tracker.lead_changes == 1
    assert all(event.overtaker == "B" and event.passed == "A" for event in tracker.events)
    assert all(event.time_seconds == 0.1 and event.confirmed_at_seconds == 0.4 for event in tracker.events)
    tracker.update(snapshot(1, (15, 17, 10, 11)))
    tracker.update(snapshot(1.3, (16, 18, 11, 13)))
    assert tracker.overtakes == 2
    assert tracker.lead_changes == 1


def test_flicker_and_short_swaps_are_not_overtakes() -> None:
    tracker = RaceInterestTracker()
    tracker.update(snapshot(0, (10, 9.9, 5, 1)))
    tracker.update(snapshot(0.1, (10, 10.2, 5, 1)))
    tracker.update(snapshot(0.5, (10, 10.2, 5, 1)))
    tracker.update(snapshot(0.6, (10, 11, 5, 1)))
    tracker.update(snapshot(0.8, (12, 11, 5, 1)))
    tracker.update(snapshot(1.2, (15, 14, 5, 1)))
    assert not tracker.events


@pytest.mark.parametrize("change", ["eliminated", "recovered", "finished"])
def test_retirement_recovery_and_finish_promotions_do_not_count(change: str) -> None:
    tracker = RaceInterestTracker()
    tracker.update(snapshot(0, (10, 8, 4, 2)))
    for time in (0.1, 0.5, 1, 3, 3.5):
        tracker.update(
            snapshot(
                time,
                (0, 20, 15, 10),
                eliminated="A" if change == "eliminated" else "",
                recovered="A" if change == "recovered" and time == 0.1 else "",
                finished=("A",) if change == "finished" else (),
            )
        )
    assert not tracker.events


def test_unwrapped_lap_boundary_and_lapping_do_not_create_passes() -> None:
    tracker = RaceInterestTracker()
    tracker.update(snapshot(0, (99, 98, 1, 0)))
    tracker.update(snapshot(0.5, (102, 101, 2, 1)))
    tracker.update(snapshot(1, (105, 103, 3, 2)))
    assert not tracker.events


def test_backward_lapped_car_cannot_take_lead_based_on_modulo_position() -> None:
    tracker = RaceInterestTracker()
    tracker.update(snapshot(0, (110, 80, 50, 30)))
    tracker.update(snapshot(0.5, (115, 90, 51, 31)))
    tracker.update(snapshot(1, (120, 100, 52, 32)))
    assert not tracker.events


def test_tracker_rejects_reuse_between_races() -> None:
    tracker = RaceInterestTracker()
    first = snapshot(0, (10, 8, 4, 2))
    tracker.update(first)
    with pytest.raises(ValueError, match="fresh"):
        tracker.update(HeatRaceSnapshot(2, 0, first.samples, frozenset()))


def test_score_weights_rates_and_finish_component() -> None:
    result = score_interest(
        lead_changes=5,
        overtakes=35,
        round_laps=5,
        entrant_count=8,
        finish_times=(100, 100, 100),
    )
    assert result.lead_points == result.overtake_points == 15
    assert result.finish_points == 40
    assert result.score == 70
    assert result.gap_p2_seconds == result.gap_p3_seconds == 0


def test_more_battles_and_closer_finishes_raise_score_without_unbounded_growth() -> None:
    quiet = score_interest(lead_changes=0, overtakes=0, round_laps=5, entrant_count=8, finish_times=(100, 105, 110))
    busy = score_interest(lead_changes=5, overtakes=35, round_laps=5, entrant_count=8, finish_times=(100, 105, 110))
    close = score_interest(
        lead_changes=5, overtakes=35, round_laps=5, entrant_count=8, finish_times=(100, 100.2, 100.5)
    )
    assert 0 < quiet.score < busy.score < close.score < 100
    extreme = score_interest(
        lead_changes=10000, overtakes=10000, round_laps=5, entrant_count=8, finish_times=(100, 100, 100)
    )
    assert close.score < extreme.score < 100


def test_equal_event_rates_score_equally_across_lengths_and_field_sizes() -> None:
    first = score_interest(lead_changes=3, overtakes=9, round_laps=3, entrant_count=4, finish_times=(100, 101, 102))
    second = score_interest(lead_changes=6, overtakes=42, round_laps=6, entrant_count=8, finish_times=(200, 201, 202))
    assert first.score == second.score


@pytest.mark.parametrize("times", [(), (100,), (100, 100.01)])
def test_missing_third_finisher_earns_no_finish_credit(times: tuple[float, ...]) -> None:
    score = score_interest(lead_changes=0, overtakes=0, round_laps=5, entrant_count=8, finish_times=times)
    assert score.finish_points == score.score == 0
    assert score.gap_p3_seconds is None


def test_two_car_exhibition_scores_passes_without_inventing_a_third_finisher() -> None:
    score = score_interest(lead_changes=1, overtakes=2, round_laps=3, entrant_count=2, finish_times=(10, 10.1))
    assert score.score > 0 and score.finish_points == 0 and score.gap_p3_seconds is None


def test_weight_changes_are_normalized_and_only_affect_the_requested_terms() -> None:
    finish_only = score_interest(
        lead_changes=10,
        overtakes=10,
        round_laps=5,
        entrant_count=8,
        finish_times=(100, 100, 100),
        weights=InterestWeights(0, 0, 2),
    )
    assert finish_only.score == finish_only.finish_points == 100
    assert finish_only.lead_points == finish_only.overtake_points == 0
    with pytest.raises(ValueError):
        InterestWeights(0, 0, 0)
    with pytest.raises(ValueError):
        InterestWeights(float("nan"), 30, 40)
