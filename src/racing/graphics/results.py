"""Presentation of official race results, independent of the renderer."""

from __future__ import annotations

from dataclasses import dataclass
from math import isclose, isfinite

from racing.graphics.colors import ColorRGBA
from racing.graphics.timing_tower import TimingTowerRow
from racing.race.head_to_head import HeadToHeadResult
from racing.race.heat import HeatRaceEntry, HeatResult


@dataclass(frozen=True, slots=True)
class ResultsTable:
    rows: tuple[TimingTowerRow, ...]
    details: dict[str, str]


def format_result_time(seconds: float | None) -> str:
    """Round to milliseconds before splitting, including minute rollovers."""
    if seconds is None or not isfinite(seconds) or seconds < 0:
        return "—"
    minutes, milliseconds = divmod(round(seconds * 1000), 60_000)
    return f"{minutes}:{milliseconds // 1000:02}.{milliseconds % 1000:03}"


def heat_results_table(
    result: HeatResult, colors: dict[str, ColorRGBA], gaps: dict[str, float | None],
) -> ResultsTable:
    """Keep official scoring order, including ties and unfinished lap races."""
    rows: list[TimingTowerRow] = []
    details: dict[str, str] = {}
    standings = result.standings
    winner_time = standings[0].finish_time_seconds if standings else None
    previous_score: float | None = None
    rank = 0
    for index, standing in enumerate(standings, start=1):
        car_id = HeatRaceEntry(standing.entrant_index).car_id
        score = standing.placement_total if result.round_laps is not None else standing.distance_m
        if previous_score is None or score is None or not isclose(score, previous_score, rel_tol=0, abs_tol=1e-6):
            rank = index
        previous_score = score
        dnf = result.round_laps is not None and result.race_count == 1 and standing.dnf
        rows.append(TimingTowerRow(car_id, standing.name, colors[car_id], rank, None, dnf=dnf))
        if result.round_laps is not None and result.race_count == 1:
            if dnf:
                detail = f"DNF · {standing.lap_count}/{result.round_laps} laps"
            else:
                time = standing.finish_time_seconds
                gap = max(0.0, time - winner_time) if time is not None and winner_time is not None else None
                suffix = "WINNER" if rank == 1 else (f"+{gap:.3f}s" if gap is not None else "—")
                detail = f"{format_result_time(time)} · {suffix}"
        elif result.round_laps is not None:
            # Placements decide a series; a partial sum is never a finishing time.
            times = [
                item.finish_time_seconds for race in result.races for item in race.standings
                if item.entrant_index == standing.entrant_index and item.finish_time_seconds is not None
            ]
            timing = (
                f"{standing.dnf_count} DNF" if standing.dnf_count else f"{format_result_time(sum(times))} total"
            )
            detail = f"{standing.placement_total} place total · {timing}"
        else:
            gap = gaps.get(car_id) if result.race_count == 1 else None
            timing = f"{format_result_time(result.round_seconds * result.race_count)} elapsed"
            if rank > 1 and gap is not None and isfinite(gap) and gap >= 0:
                timing = f"Gap +{gap:.3f}s"
            detail = f"{standing.distance_m:.1f} m · {timing}"
        details[car_id] = detail
    return ResultsTable(tuple(rows), details)


def head_to_head_results_table(result: HeadToHeadResult, colors: dict[str, ColorRGBA]) -> ResultsTable:
    """Rank the two teams by the same wins used by the official suite result."""
    rows: list[TimingTowerRow] = []
    details: dict[str, str] = {}
    roles = ("incumbent", "challenger") if result.winner == "incumbent" else ("challenger", "incumbent")
    for index, role in enumerate(roles, start=1):
        challenger = role == "challenger"
        stats = [race.challenger if challenger else race.incumbent for race in result.races]
        distance = sum(
            team.team_sum_distance_m if race.scoring == "team-sum" else team.best_distance_m
            for race, team in zip(result.races, stats, strict=True)
        )
        laps = [team.best_lap_time_seconds for team in stats if team.best_lap_time_seconds is not None]
        timing = (
            f"Best lap {format_result_time(min(laps))}" if laps
            else f"{format_result_time(result.round_seconds * result.race_count)} elapsed"
        )
        wins = result.challenger_wins if challenger else result.incumbent_wins
        wins_text = f"{wins} wins · " if result.race_count > 1 else ""
        details[role] = f"{wins_text}{distance:.1f} m · {timing}"
        rows.append(TimingTowerRow(
            role, result.challenger_name if challenger else result.incumbent_name,
            colors[role], 1 if result.winner == "tie" else index, None,
        ))
    return ResultsTable(tuple(rows), details)
