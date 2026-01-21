from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd


def load_nbastatsv3(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load nbastatsv3_* CSV (play-by-play like) and normalize a few useful columns.
    """
    path = Path(path)
    df = pd.read_csv(path)

    # Normalize booleans and types
    if "isFieldGoal" in df.columns:
        df["isFieldGoal"] = df["isFieldGoal"].astype(int)

    if "shotValue" in df.columns:
        df["shotValue"] = df["shotValue"].fillna(0).astype(int)

    if "shotResult" in df.columns:
        df["shotResult"] = df["shotResult"].fillna("")

    return df
