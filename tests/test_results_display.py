from __future__ import annotations

import pytest

from racing.graphics.results import format_result_time, head_to_head_results_table, heat_results_table
from racing.race.head_to_head import HeadToHeadRaceResult, HeadToHeadResult, HeadToHeadTeamRaceStats
from racing.race.heat import DEFAULT_HEAT_COLORS, HeatRaceEntry, HeatRaceResult, HeatResult, HeatStanding


@pytest.mark.parametrize(("seconds", "expected"), [
    (0, "0:00.000"), (83.819, "1:23.819"), (59.9996, "1:00.000"),
    (3600.001, "60:00.001"), (None, "—"), (-1, "—"), (float("nan"), "—"), (float("inf"), "—"),
])
def test_result_time_formats_milliseconds_and_rollover(seconds: float | None, expected: str) -> None:
    assert format_result_time(seconds) == expected


def _standing(index: int, position: int | None, time: float | None, distance: float = 100) -> HeatStanding:
    return HeatStanding(index, f"Car {index}", distance, 1 if time is not None else 0, 0, 0,
                        finish_position=position, finish_time_seconds=time, dnf=time is None)


def _heat(*races: tuple[HeatStanding, ...], laps: int | None = 1) -> HeatResult:
    return HeatResult(tuple(item.name for item in races[0]), 90,
                      tuple(HeatRaceResult(index, rows) for index, rows in enumerate(races, start=1)), round_laps=laps)


def _colors(count: int) -> dict[str, tuple[float, float, float, float]]:
    return {HeatRaceEntry(index).car_id: DEFAULT_HEAT_COLORS[index] for index in range(count)}


def test_lap_results_use_recorded_finish_order_and_gap_to_winner() -> None:
    result = _heat((_standing(0, 3, 85), _standing(1, 1, 83.125), _standing(2, 2, 83.819), _standing(3, None, None)))
    table = heat_results_table(result, _colors(4), {"heat-2:0": 99})
    assert [row.car_id for row in table.rows] == ["heat-1:0", "heat-2:0", "heat-0:0", "heat-3:0"]
    assert table.details["heat-1:0"] == "1:23.125 · WINNER"
    assert table.details["heat-2:0"] == "1:23.819 · +0.694s"
    assert table.rows[-1].dnf
    assert table.details["heat-3:0"] == "DNF · 0/1 laps"


def test_all_dnf_does_not_award_a_winner_or_invent_times() -> None:
    table = heat_results_table(_heat((_standing(0, None, None), _standing(1, None, None))), _colors(2), {})
    assert all(row.dnf for row in table.rows)
    assert all(detail.startswith("DNF") for detail in table.details.values())


def test_series_uses_placement_totals_and_never_reports_partial_time_as_total() -> None:
    result = _heat(
        (_standing(0, 1, 83), _standing(1, 2, 85), _standing(2, None, None)),
        (_standing(0, 2, 85), _standing(1, 1, 83), _standing(2, 3, 86)),
    )
    table = heat_results_table(result, _colors(3), {})
    assert [row.rank for row in table.rows] == [1, 1, 3]
    assert table.details["heat-0:0"] == "3 place total · 2:48.000 total"
    assert table.details["heat-2:0"] == "7 place total · 1 DNF"


def test_timed_results_preserve_scored_distance_ties_and_only_use_known_gaps() -> None:
    result = _heat((_standing(0, None, None, 120), _standing(1, None, None, 120),
                    _standing(2, None, None, 110)), laps=None)
    table = heat_results_table(result, _colors(3), {"heat-2:0": 0.694})
    assert [row.rank for row in table.rows] == [1, 1, 3]
    assert not any(row.dnf for row in table.rows)
    assert table.details["heat-0:0"] == "120.0 m · 1:30.000 elapsed"
    assert table.details["heat-2:0"] == "110.0 m · Gap +0.694s"


def test_head_to_head_ranks_by_official_wins_and_keeps_best_lap_time() -> None:
    first = HeadToHeadTeamRaceStats((100,), (1,), (0,), (0,), best_lap_times_seconds=(65.123,))
    second = HeadToHeadTeamRaceStats((120,), (1,), (0,), (0,), best_lap_times_seconds=(60.222,))
    result = HeadToHeadResult("AA", "BB", 90, 1, (HeadToHeadRaceResult(1, "incumbent", first, second),))
    table = head_to_head_results_table(result, {"challenger": DEFAULT_HEAT_COLORS[0],
                                               "incumbent": DEFAULT_HEAT_COLORS[1]})
    assert [row.car_id for row in table.rows] == ["incumbent", "challenger"]
    assert table.details["incumbent"] == "120.0 m · Best lap 1:00.222"
