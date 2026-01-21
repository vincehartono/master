from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlayerState:
    player_id: int
    team_id: int
    name: str | None = None
    minutes_played: float = 0.0
    fouls: int = 0
    fatigue: float = 0.0  # 0 fresh, 1 exhausted
    on_court: bool = False


@dataclass
class TeamState:
    team_id: int
    name: str | None = None
    score: int = 0
    fouls_team: int = 0
    timeouts: int = 7
    lineup: List[int] = field(default_factory=list)  # player_ids on court


@dataclass
class GameState:
    game_id: str
    period: int = 1
    clock: float = 12 * 60.0  # seconds remaining in period
    home_team_id: Optional[int] = None
    away_team_id: Optional[int] = None
    possession_team_id: Optional[int] = None
    players: Dict[int, PlayerState] = field(default_factory=dict)
    teams: Dict[int, TeamState] = field(default_factory=dict)
    events: List[dict] = field(default_factory=list)  # simple log for now

    def log_event(self, event: dict) -> None:
        self.events.append(event)
