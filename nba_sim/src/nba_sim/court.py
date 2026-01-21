from __future__ import annotations

import numpy as np
import pandas as pd

from .config import CourtConfig


def normalize_coords(df: pd.DataFrame, court: CourtConfig | None = None) -> pd.DataFrame:
    """
    Normalize shot coordinates into [0, 1] x [0, 1] court space.
    Assumes LOC_X / LOC_Y are in feet relative to one baseline.
    """
    court = court or CourtConfig()
    out = df.copy()
    out["x_norm"] = (out["LOC_X"]) / court.length_ft
    out["y_norm"] = (out["LOC_Y"]) / court.width_ft
    return out


def distance_to_hoop(df: pd.DataFrame, court: CourtConfig | None = None) -> np.ndarray:
    court = court or CourtConfig()
    dx = df["LOC_X"].to_numpy() - court.hoop_x_ft
    dy = df["LOC_Y"].to_numpy() - court.hoop_y_ft
    return np.sqrt(dx * dx + dy * dy)
