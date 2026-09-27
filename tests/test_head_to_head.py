from __future__ import annotations

import json

import pytest

from racing.race.head_to_head import (
    HeadToHeadRaceResult,
    HeadToHeadResult,
    HeadToHeadTeamRaceStats,
    classify_head_to_head_winner,
    format_head_to_head_result,
    format_head_to_head_result_banner,
    head_to_head_race_entries,
    head_to_head_race_margin,
)


def test_head_to_head_stats_compute_team_distances_and_contacts() -> None:
    stats = HeadToHeadTeamRaceStats(
        distances_m=(10.0, 6.0),
        lap_counts=(1, 0),
        wall_contact_seconds=(0.5, 0.25),
        car_contact_seconds=(0.0, 0.75),
    )

    assert stats.best_distance_m == 10.0
    assert stats.team_sum_distance_m == 16.0
    assert stats.average_distance_m == 8.0
    assert stats.team_sum_raw_distance_m == 16.0
    assert stats.max_speed_mps == 0.0
    assert stats.best_lap_time_seconds is None
    assert stats.total_contact_seconds == 1.5


def test_head_to_head_stats_compute_extended_aggregates() -> None:
    stats = HeadToHeadTeamRaceStats(
        distances_m=(10.0, 6.0),
        lap_counts=(1, 0),
        wall_contact_seconds=(0.5, 0.25),
        car_contact_seconds=(0.0, 0.75),
        damages=(0.25, 1.0),
        raw_distances_m=(12.0, 7.0),
        max_speeds_mps=(8.0, 11.0),
        best_lap_times_seconds=(42.0, None),
        penalized_distances_m=(1.5, 0.5),
        marshal_counts=(1, 2),
        marshal_penalties_m=(5.0, 10.0),
    )

    assert stats.best_raw_distance_m == 12.0
    assert stats.team_sum_raw_distance_m == 19.0
    assert stats.max_speed_mps == 11.0
    assert stats.best_lap_time_seconds == 42.0
    assert stats.total_lap_count == 1
    assert stats.average_damage == pytest.approx(0.625)
    assert stats.elimination_count == 1
    assert stats.total_penalized_distance_m == 2.0
    assert stats.total_marshal_count == 3
    assert stats.total_marshal_penalty_m == 15.0


def test_head_to_head_stats_reject_mismatched_extended_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        HeadToHeadTeamRaceStats(
            distances_m=(10.0,),
            lap_counts=(1,),
            wall_contact_seconds=(0.0,),
            car_contact_seconds=(0.0,),
            max_speeds_mps=(8.0, 9.0),
        )


def test_head_to_head_margin_supports_scoring_modes() -> None:
    challenger = HeadToHeadTeamRaceStats(
        distances_m=(9.0, 4.0),
        lap_counts=(0, 0),
        wall_contact_seconds=(0.0, 0.0),
        car_contact_seconds=(0.0, 0.0),
    )
    incumbent = HeadToHeadTeamRaceStats(
        distances_m=(8.0, 7.0),
        lap_counts=(0, 0),
        wall_contact_seconds=(0.0, 0.0),
        car_contact_seconds=(0.0, 0.0),
    )

    assert head_to_head_race_margin(challenger=challenger, incumbent=incumbent, scoring="best-copy") == 1.0
    assert head_to_head_race_margin(challenger=challenger, incumbent=incumbent, scoring="team-sum") == -2.0


def test_classify_head_to_head_winner_uses_margin_threshold() -> None:
    assert classify_head_to_head_winner(margin_m=1.1, win_margin_m=1.0) == "challenger"
    assert classify_head_to_head_winner(margin_m=-1.1, win_margin_m=1.0) == "incumbent"
    assert classify_head_to_head_winner(margin_m=1.0, win_margin_m=1.0) == "tie"


def test_head_to_head_entries_are_deterministic_and_include_each_role() -> None:
    first = head_to_head_race_entries(race_index=2, random_seed=110, challenger_copies=2, incumbent_copies=3)
    second = head_to_head_race_entries(race_index=2, random_seed=110, challenger_copies=2, incumbent_copies=3)

    assert first == second
    assert sum(1 for entry in first if entry.role == "challenger") == 2
    assert sum(1 for entry in first if entry.role == "incumbent") == 3


def test_format_head_to_head_result_outputs_winner_first_table() -> None:
    race_one_challenger = HeadToHeadTeamRaceStats(
        distances_m=(12.0,),
        raw_distances_m=(14.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.5,),
        car_contact_seconds=(0.25,),
        damages=(0.10,),
        max_speeds_mps=(5.0,),
        low_progress_seconds=(1.0,),
        off_track_seconds=(2.0,),
        marshal_counts=(1,),
        marshal_penalties_m=(5.0,),
    )
    race_one_incumbent = HeadToHeadTeamRaceStats(
        distances_m=(20.0,),
        raw_distances_m=(22.0,),
        lap_counts=(1,),
        wall_contact_seconds=(0.0,),
        car_contact_seconds=(0.0,),
        damages=(0.20,),
        max_speeds_mps=(7.0,),
        best_lap_times_seconds=(58.0,),
    )
    race_two_challenger = HeadToHeadTeamRaceStats(
        distances_m=(4.0,),
        raw_distances_m=(5.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.0,),
        car_contact_seconds=(0.0,),
        damages=(1.0,),
        max_speeds_mps=(8.0,),
    )
    race_two_incumbent = HeadToHeadTeamRaceStats(
        distances_m=(11.0,),
        raw_distances_m=(13.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.25,),
        car_contact_seconds=(0.5,),
        damages=(0.0,),
        max_speeds_mps=(6.0,),
    )
    result = HeadToHeadResult(
        challenger_name="driver_a",
        incumbent_name="driver_b",
        round_seconds=60.0,
        win_margin_m=1.0,
        races=(
            HeadToHeadRaceResult(
                race_index=1,
                winner="incumbent",
                challenger=race_one_challenger,
                incumbent=race_one_incumbent,
            ),
            HeadToHeadRaceResult(
                race_index=2,
                winner="incumbent",
                challenger=race_two_challenger,
                incumbent=race_two_incumbent,
            ),
        ),
    )

    summary = format_head_to_head_result(result)
    lines = summary.splitlines()
    header = next(line for line in lines if line.startswith("Metric"))
    best_lap_row = next(line for line in lines if line.startswith("Best lap"))

    assert lines[0] == "Winner: driver_b"
    assert "Record: driver_a 0, driver_b 2, ties 0" in summary
    assert header.index("driver_b") < header.index("driver_a")
    assert "Scored distance" in summary
    assert "Raw distance" in summary
    assert "Max speed" in summary
    assert "Average damage" in summary
    assert "Eliminations" in summary
    assert "Marshal resets" in summary
    assert "Off-track time" in summary
    assert "31.0 m" in summary
    assert "58.00 s" in best_lap_row
    assert "--" in best_lap_row
    assert "race 01" in summary
    assert "race 02" in summary


def test_format_head_to_head_result_preserves_challenger_order_for_tie() -> None:
    challenger = HeadToHeadTeamRaceStats(
        distances_m=(10.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.0,),
        car_contact_seconds=(0.0,),
    )
    incumbent = HeadToHeadTeamRaceStats(
        distances_m=(10.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.0,),
        car_contact_seconds=(0.0,),
    )
    result = HeadToHeadResult(
        challenger_name="challenger",
        incumbent_name="incumbent",
        round_seconds=60.0,
        win_margin_m=1.0,
        races=(
            HeadToHeadRaceResult(
                race_index=1,
                winner="tie",
                challenger=challenger,
                incumbent=incumbent,
            ),
        ),
    )

    summary = format_head_to_head_result(result)
    header = next(line for line in summary.splitlines() if line.startswith("Metric"))

    assert summary.splitlines()[0] == "Winner: tie"
    assert header.index("challenger") < header.index("incumbent")


def test_format_head_to_head_result_banner_names_winner_and_distances() -> None:
    result = HeadToHeadResult(
        challenger_name="candidate",
        incumbent_name="baseline",
        round_seconds=30.0,
        win_margin_m=1.0,
        races=(
            HeadToHeadRaceResult(
                race_index=1,
                winner="challenger",
                challenger=HeadToHeadTeamRaceStats(
                    distances_m=(18.25,),
                    lap_counts=(0,),
                    wall_contact_seconds=(0.0,),
                    car_contact_seconds=(0.0,),
                ),
                incumbent=HeadToHeadTeamRaceStats(
                    distances_m=(12.75,),
                    lap_counts=(0,),
                    wall_contact_seconds=(0.0,),
                    car_contact_seconds=(0.0,),
                ),
            ),
        ),
    )

    banner = format_head_to_head_result_banner(result)

    assert banner == (
        "WINNER: candidate\ncandidate: 18.2 m  |  0.0% total damage\nbaseline: 12.8 m  |  0.0% total damage"
    )


def test_head_to_head_result_has_versioned_json_compatible_record() -> None:
    challenger = HeadToHeadTeamRaceStats(
        distances_m=(12.0,),
        lap_counts=(1,),
        wall_contact_seconds=(0.0,),
        car_contact_seconds=(0.0,),
    )
    incumbent = HeadToHeadTeamRaceStats(
        distances_m=(8.0,),
        lap_counts=(0,),
        wall_contact_seconds=(0.5,),
        car_contact_seconds=(0.0,),
    )
    result = HeadToHeadResult(
        challenger_name="learned",
        incumbent_name="baseline",
        round_seconds=30.0,
        win_margin_m=1.0,
        races=(
            HeadToHeadRaceResult(
                race_index=1,
                winner="challenger",
                challenger=challenger,
                incumbent=incumbent,
            ),
        ),
        random_seed=271,
        track_id="procedural-110",
        track_seed=110,
        fixed_delta_seconds=1 / 60,
    )

    record = result.to_dict()
    encoded = json.dumps(record, allow_nan=False)

    assert record["schema_version"] == 1
    assert record["fixed_delta_seconds"] == pytest.approx(1 / 60)
    assert record["track_id"] == "procedural-110"
    assert record["track_seed"] == 110
    assert record["summary"] == {
        "winner": "challenger",
        "race_count": 1,
        "challenger_wins": 1,
        "incumbent_wins": 0,
        "ties": 0,
        "aggregate_margin_m": 4.0,
    }
    assert '"raw_distances_m": [12.0]' in encoded
