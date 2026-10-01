from __future__ import annotations

import pytest

from racing.game.cli import build_argument_parser
from racing.graphics.track_mesh import clean_offset_path, offset_path
from racing.graphics.track_rendering import TRACK_EDGE_BUFFER, TRACK_WALL_THICKNESS
from racing.race.progress import resolve_track, track_pose_at_distance
from racing.track.world import TRACK_WIDTH


@pytest.mark.parametrize(
    "arguments",
    [
        ["--track", "bahrain"],
        ["h2h", "--track", "bahrain"],
        ["--track", "bahrain", "h2h"],
        ["heat", "--track", "bahrain", "--module", "controllers.level_3"],
        ["--track", "bahrain", "heat", "--module", "controllers.level_3"],
    ],
)
def test_bahrain_is_selectable_in_every_race_mode(arguments: list[str]) -> None:
    assert build_argument_parser().parse_args(arguments).track == "bahrain"


def test_bahrain_lap_starts_on_main_straight_toward_opening_hairpin() -> None:
    track = resolve_track("bahrain")
    start = track_pose_at_distance(track.model, 0.0)
    ahead = track_pose_at_distance(track.model, 20.0)
    wrapped = track_pose_at_distance(track.model, track.model.total_length_m)

    assert start.position.x == pytest.approx(track.layout.start_position.x)
    assert start.position.z == pytest.approx(track.layout.start_position.z)
    assert start.heading_degrees == pytest.approx(-90.0)
    assert ahead.position.x < start.position.x
    assert ahead.position.z == pytest.approx(start.position.z)
    assert wrapped.position == start.position
    assert min(track.model.segment_lengths) > 0.0


@pytest.mark.parametrize(
    "offset",
    [
        -TRACK_WIDTH / 2,
        TRACK_WIDTH / 2,
        -(TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS),
        TRACK_WIDTH / 2 + TRACK_EDGE_BUFFER + TRACK_WALL_THICKNESS,
    ],
)
def test_bahrain_road_and_barriers_do_not_need_intersection_trimming(offset: float) -> None:
    samples = resolve_track("bahrain").samples

    assert clean_offset_path(samples, offset, 0.0) == offset_path(samples, offset, 0.0)
