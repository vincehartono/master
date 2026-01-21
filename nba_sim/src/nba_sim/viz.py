from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import pandas as pd


def plot_shot_scatter(
    df: pd.DataFrame,
    *,
    made_col: str = "SHOT_MADE_FLAG",
    save_path: Optional[Union[str, Path]] = None,
) -> None:
    """
    Basic sanity-check scatter of shots on the court using LOC_X / LOC_Y.
    """
    made = df[df[made_col] == 1]
    miss = df[df[made_col] == 0]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(miss["LOC_X"], miss["LOC_Y"], s=5, c="lightcoral", alpha=0.4, label="Miss")
    ax.scatter(made["LOC_X"], made["LOC_Y"], s=8, c="seagreen", alpha=0.7, label="Make")

    ax.set_xlabel("LOC_X (ft)")
    ax.set_ylabel("LOC_Y (ft)")
    ax.set_title("Shot chart (raw coords)")
    ax.legend(loc="upper right")
    ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    else:
        plt.show()

    plt.close(fig)
