from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


EventType = Literal["shot", "turnover", "foul", "rebound", "substitution", "period_start", "period_end"]


@dataclass
class BaseEvent:
    game_id: str
    period: int
    clock: float  # seconds remaining in period
    team_id: Optional[int] = None


@dataclass
class ShotEvent(BaseEvent):
    player_id: int
    shot_value: int
    x: float
    y: float
    distance_ft: float
    is_made: bool


@dataclass
class TurnoverEvent(BaseEvent):
    player_id: Optional[int] = None
    subtype: str | None = None


@dataclass
class FoulEvent(BaseEvent):
    committed_by: Optional[int] = None
    drawn_by: Optional[int] = None
    is_shooting: bool = False


@dataclass
class ReboundEvent(BaseEvent):
    player_id: int | None = None
    is_offensive: bool = False


@dataclass
class SubstitutionEvent(BaseEvent):
    player_out: int
    player_in: int


@dataclass
class PeriodBoundaryEvent(BaseEvent):
    kind: Literal["start", "end"] = "start"
