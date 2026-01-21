from __future__ import annotations

import numpy as np
import pandas as pd

from .court import distance_to_hoop


def fit_baseline_model(df: pd.DataFrame) -> dict:
    """
    Toy model: make prob as logistic function of distance.
    Expects a boolean/int column 'SHOT_MADE_FLAG' (1 = make).
    """
    dist = distance_to_hoop(df)
    y = df["SHOT_MADE_FLAG"].to_numpy()

    if dist.size == 0:
        raise ValueError("No shots provided")

    # Simple heuristic parameters; later replace with real fit.
    beta0 = 1.0
    beta1 = -0.25

    return {"beta0": beta0, "beta1": beta1}


def make_prob(dist_ft: np.ndarray, model: dict) -> np.ndarray:
    z = model["beta0"] + model["beta1"] * dist_ft
    return 1.0 / (1.0 + np.exp(-z))
