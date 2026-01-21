from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class PossessionModel:
    """
    Very simple model:
    - For each player, probability of taking a shot when on court.
    - For each player, field goal % by shot value (2/3).
    """

    usage: Dict[int, float]
    fg2: Dict[int, float]
    fg3: Dict[int, float]

    @classmethod
    def from_nbastatsv3(cls, df: pd.DataFrame) -> "PossessionModel":
        """
        Build from nbastatsv3_2025.csv-like data.
        Expects:
          - 'personId', 'shotResult' ('Made'/'Missed'), 'shotValue' (2/3), 'isFieldGoal' (0/1)
        """
        shots = df[df["isFieldGoal"] == 1].copy()

        # Usage: relative share of shots among players
        usage_counts = shots["personId"].value_counts()
        usage = (usage_counts / usage_counts.sum()).to_dict()

        def fg_for_value(value: int) -> Dict[int, float]:
            sub = shots[shots["shotValue"] == value]
            makes = sub[sub["shotResult"] == "Made"].groupby("personId")["shotResult"].count()
            attempts = sub.groupby("personId")["shotResult"].count()
            pct = (makes / attempts).fillna(0.0)
            return pct.to_dict()

        fg2 = fg_for_value(2)
        fg3 = fg_for_value(3)

        return cls(usage=usage, fg2=fg2, fg3=fg3)

    def sample_shooter(self, rng: np.random.Generator) -> int:
        players = np.fromiter(self.usage.keys(), dtype=int)
        probs = np.fromiter(self.usage.values(), dtype=float)
        probs = probs / probs.sum()
        return int(rng.choice(players, p=probs))

    def shot_prob(self, player_id: int, shot_value: int) -> float:
        if shot_value == 3:
            return float(self.fg3.get(player_id, 0.33))
        return float(self.fg2.get(player_id, 0.5))
