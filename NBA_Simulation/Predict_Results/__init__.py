"""
Predict_Results package for NBA_Simulation.

Re-exports key functions so existing imports like
`import Predict_Results as PR` continue to work
when this is used as a package (e.g., in AWS Lambda).
"""

from .Predict_Results import (  # noqa: F401
    _read_history,
    _normalize_history,
    todays_games,
    _exp_weights,
    predict_from_history,
    backtest_from_history,
    _season_start_for_date,
)

