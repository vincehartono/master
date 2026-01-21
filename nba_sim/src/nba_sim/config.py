from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class CourtConfig:
    length_ft: float = 94.0
    width_ft: float = 50.0
    hoop_x_ft: float = 5.25
    hoop_y_ft: float = 25.0


@dataclass
class SimConfig:
    n_possessions: int = 1000
    seed: Optional[int] = 42
