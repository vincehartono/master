"""
Predict_Results package for NBA_Simulation.

Re-exports key functions so existing imports like
`import Predict_Results as PR` continue to work
when this is used as a package (e.g., in AWS Lambda).
"""

# Use absolute import so it also works when the package is
# loaded as top-level (avoids relative import issues in Lambda).
from NBA_Simulation.Predict_Results.Predict_Results import (
    _read_history,
    _normalize_history,
    todays_games,
    _exp_weights,
    predict_from_history,
    backtest_from_history,
    _season_start_for_date,
)