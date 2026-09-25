from __future__ import annotations

from racing.graphics.timing_tower import TimingTowerRow, format_race_countdown, format_timing_gap


def test_timing_labels_distinguish_leader_unknown_gap_and_elimination() -> None:
    color = (0.3, 0.6, 0.8, 1.0)

    assert format_timing_gap(TimingTowerRow("leader", "Leader", color, 1, None)) == "LEADER"
    assert format_timing_gap(TimingTowerRow("second", "Second", color, 2, 1.23449)) == "+1.234"
    assert format_timing_gap(TimingTowerRow("unknown", "Unknown", color, 3, None)) == "—"
    assert format_timing_gap(TimingTowerRow("out", "Retired", color, 1, 0.0, eliminated=True)) == "OUT"


def test_timing_gap_never_displays_negative_or_non_finite_values() -> None:
    color = (0.3, 0.6, 0.8, 1.0)

    assert format_timing_gap(TimingTowerRow("close", "Close", color, 2, -0.001)) == "+0.000"
    assert format_timing_gap(TimingTowerRow("invalid", "Invalid", color, 3, float("nan"))) == "—"
    assert format_timing_gap(TimingTowerRow("invalid", "Invalid", color, 3, float("inf"))) == "—"


def test_countdown_preserves_full_seconds_and_stops_at_zero() -> None:
    assert format_race_countdown(90) == "01:30"
    assert format_race_countdown(89.99) == "01:30"
    assert format_race_countdown(89) == "01:29"
    assert format_race_countdown(0.01) == "00:01"
    assert format_race_countdown(-0.01) == "00:00"
