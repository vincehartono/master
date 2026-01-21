from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import SimConfig
from .court import distance_to_hoop
from .models.possession_model import PossessionModel
from .state import GameState, TeamState


def simulate_possessions(shots: pd.DataFrame, model: dict, config: Optional[SimConfig] = None) -> pd.DataFrame:
    """
    Legacy possession simulator kept for quick experiments with shotdetail data.
    """
    config = config or SimConfig()
    rng = np.random.default_rng(config.seed)

    if shots.empty:
        raise ValueError("No shot data to simulate from")

    idx = rng.integers(0, len(shots), size=config.n_possessions)
    sampled = shots.iloc[idx].reset_index(drop=True)

    dist = distance_to_hoop(sampled)
    # dummy: uniform 0.5 for now if no 'model' structure is provided
    p = np.full_like(dist, fill_value=0.5, dtype=float)
    makes = rng.binomial(1, p)

    out = sampled.copy()
    out["make_prob"] = p
    out["sim_make"] = makes
    return out


def simulate_single_game_from_nbastatsv3(
    df: pd.DataFrame,
    game_id: str,
    *,
    seed: Optional[int] = 42,
) -> GameState:
    """
    Very simple game-level simulator using nbastatsv3 data:
    - Build a PossessionModel from all shots in the file.
    - For the specified game_id, simulate a sequence of shot events
      with basic 2/3 point shots and track score.

    This is intentionally minimal and does not yet model fouls,
    rebounds, substitutions, or clock in detail.
    """
    # gameId in the file may be stored as int; normalize to string for comparison
    game_id_str = str(game_id)
    game_rows = df[df["gameId"].astype(str) == game_id_str].copy()
    if game_rows.empty:
        raise ValueError(f"No rows found for game_id={game_id}")

    # Use home/away scores from data to infer final period/clock span
    first = game_rows.iloc[0]
    # We do not explicitly know home/away team IDs here; just treat team ids that appear
    team_ids = sorted(t for t in game_rows["teamId"].unique() if t > 0)
    home_team_id = team_ids[0] if team_ids else None
    away_team_id = team_ids[1] if len(team_ids) > 1 else None

    game_state = GameState(
        game_id=game_id,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        possession_team_id=home_team_id,
    )

    model = PossessionModel.from_nbastatsv3(df)
    rng = np.random.default_rng(seed)

    # Simulate N possessions approximately equal to number of real FGA in this game
    n_real_shots = int((game_rows["isFieldGoal"] == 1).sum())
    n_possessions = max(n_real_shots, 100)

    for _ in range(n_possessions):
        shooter_id = model.sample_shooter(rng)
        shot_value = 3 if rng.random() < 0.35 else 2
        p_make = model.shot_prob(shooter_id, shot_value)
        made = bool(rng.random() < p_make)

        # Update score for the team in possession
        team_id = game_state.possession_team_id
        if team_id is not None:
            if team_id not in game_state.teams:
                game_state.teams[team_id] = TeamState(team_id=team_id)

            if made:
                game_state.teams[team_id].score += shot_value

        game_state.log_event(
            {
                "event_type": "shot",
                "team_id": team_id,
                "player_id": shooter_id,
                "shot_value": shot_value,
                "made": made,
                "p_make": p_make,
            }
        )

        # Swap possession after each shot
        if home_team_id is not None and away_team_id is not None:
            game_state.possession_team_id = away_team_id if team_id == home_team_id else home_team_id

    return game_state


def simulate_matchup_from_nbastatsv3(
    df: pd.DataFrame,
    home_tricode: str,
    away_tricode: str,
    *,
    seed: Optional[int] = 42,
    points_scale: float = 1.0,
) -> GameState:
    """
    Simulate a game between two teams identified by teamTricode
    (e.g. 'LAC' vs 'HOU') using season play-by-play data.

    This ignores the original gameId and instead treats all
    season shots for those two teams as the basis for the model.
    """
    mask = df["teamTricode"].isin([home_tricode, away_tricode])
    teams_df = df[mask].copy()
    if teams_df.empty:
        raise ValueError(f"No rows found for teams {home_tricode} and {away_tricode}")

    team_ids = teams_df["teamId"].dropna().unique()
    if len(team_ids) < 2:
        raise ValueError(f"Need two distinct teamIds for matchup {home_tricode} vs {away_tricode}")

    # Map tricode -> first teamId seen
    tricode_to_id = (
        teams_df.dropna(subset=["teamTricode", "teamId"])
        .groupby("teamTricode")["teamId"]
        .first()
        .to_dict()
    )
    home_team_id = int(tricode_to_id.get(home_tricode))
    away_team_id = int(tricode_to_id.get(away_tricode))

    game_state = GameState(
        game_id=f"{home_tricode}_vs_{away_tricode}",
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        possession_team_id=home_team_id,
    )

    model = PossessionModel.from_nbastatsv3(teams_df)
    rng = np.random.default_rng(seed)

    # Use a fixed number of possessions to keep scores in a realistic range.
    # Rough NBA average is ~95–105 possessions per team per game.
    n_possessions = 100

    for _ in range(n_possessions):
        shooter_id = model.sample_shooter(rng)
        shot_value = 3 if rng.random() < 0.35 else 2
        p_make = model.shot_prob(shooter_id, shot_value)
        made = bool(rng.random() < p_make)

        team_id = game_state.possession_team_id
        if team_id is not None:
            if team_id not in game_state.teams:
                game_state.teams[team_id] = TeamState(team_id=team_id)
            if made:
                game_state.teams[team_id].score += int(round(shot_value * points_scale))

        game_state.log_event(
            {
                "event_type": "shot",
                "team_id": team_id,
                "player_id": shooter_id,
                "shot_value": shot_value,
                "made": made,
                "p_make": p_make,
            }
        )

        if home_team_id is not None and away_team_id is not None:
            game_state.possession_team_id = (
                away_team_id if team_id == home_team_id else home_team_id
            )

    return game_state
