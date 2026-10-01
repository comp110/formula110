"""Shared competitive race rules for head-to-head evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M = 1.0
HEAD_TO_HEAD_DEFAULT_SCORING = "team-sum"
HEAD_TO_HEAD_DEFAULT_SEED_SUITE = (42, 110, 271, 997, 2027)

HeadToHeadScoring = Literal["best-copy", "team-sum"]


@dataclass(frozen=True, slots=True)
class HeadToHeadRaceRules:
    """Authoritative rules for competitive head-to-head races."""

    scoring: HeadToHeadScoring = HEAD_TO_HEAD_DEFAULT_SCORING
    win_margin_m: float = HEAD_TO_HEAD_DEFAULT_WIN_MARGIN_M
    marshal_enabled: bool = True
    marshal_stuck_seconds: float = 1.5
    marshal_penalty_m: float = 5.0
    marshal_cooldown_seconds: float = 2.0
    damage_enabled: bool = True

    def __post_init__(self) -> None:
        if self.scoring not in ("best-copy", "team-sum"):
            raise ValueError(f"unknown head-to-head scoring rule: {self.scoring}")
        if self.win_margin_m < 0.0:
            raise ValueError("win_margin_m cannot be negative")
        if self.marshal_stuck_seconds < 0.0:
            raise ValueError("marshal_stuck_seconds cannot be negative")
        if self.marshal_penalty_m < 0.0:
            raise ValueError("marshal_penalty_m cannot be negative")
        if self.marshal_cooldown_seconds < 0.0:
            raise ValueError("marshal_cooldown_seconds cannot be negative")

    def to_dict(self) -> dict[str, object]:
        """Convert race rules into simple values that can be logged or saved."""
        return {
            "scoring": self.scoring,
            "win_margin_m": self.win_margin_m,
            "marshal_enabled": self.marshal_enabled,
            "marshal_stuck_seconds": self.marshal_stuck_seconds,
            "marshal_penalty_m": self.marshal_penalty_m,
            "marshal_cooldown_seconds": self.marshal_cooldown_seconds,
            "damage_enabled": self.damage_enabled,
        }

    def to_json(self) -> dict[str, object]:
        """Backward-compatible alias for :meth:`to_dict`."""
        return self.to_dict()
