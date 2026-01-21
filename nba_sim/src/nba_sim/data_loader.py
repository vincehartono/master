from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Union

import pandas as pd


def load_shots(path: Union[str, Path], kind: Optional[Literal["csv", "parquet"]] = None) -> pd.DataFrame:
    """
    Load shot data with at least LOC_X, LOC_Y, made/miss, player/team columns.
    """
    path = Path(path)
    if kind is None:
        if path.suffix.lower() == ".csv":
            kind = "csv"
        elif path.suffix.lower() in {".parquet", ".pq"}:
            kind = "parquet"
        else:
            raise ValueError(f"Could not infer file type from {path}")

    if kind == "csv":
        df = pd.read_csv(path)
    elif kind == "parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported kind: {kind}")

    required = {"LOC_X", "LOC_Y"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    return df
