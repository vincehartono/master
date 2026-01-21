import os
import argparse
import pandas as pd
import numpy as np


def learn_usage_caps(box_path: str, out_path: str, min_minutes: float = 6.0, pctl: float = 0.90):
    df = pd.read_csv(box_path)
    # Required columns: team.code, game.id, player.firstname, player.lastname, min, fga
    required = ["team.code", "game.id", "player.firstname", "player.lastname", "min", "fga"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {box_path}: {missing}")

    # Normalize types
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    df["fga"] = pd.to_numeric(df["fga"], errors="coerce").fillna(0.0)

    # Filter DNPs
    df = df[df["min"] >= min_minutes].copy()

    # Rank by minutes within team-game
    df["name"] = (df["player.firstname"].astype(str).str.strip() + " " + df["player.lastname"].astype(str).str.strip()).str.strip()
    df["rank"] = (
        df.sort_values(["team.code", "game.id", "min"], ascending=[True, True, False])
          .groupby(["team.code", "game.id"])  # type: ignore
          .cumcount()
          + 1
    )

    # Team FGAs per game
    team_fga = df.groupby(["team.code", "game.id"], as_index=False).agg(team_fga=("fga", "sum"))
    df = df.merge(team_fga, on=["team.code", "game.id"], how="left")
    df["usage_share"] = df["fga"] / df["team_fga"].replace(0, np.nan)

    # Remove invalids
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["usage_share"]) 

    # Compute percentile caps per rank (limit to reasonable ranks, e.g., 1..10)
    caps = (
        df[df["rank"] <= 10]
          .groupby("rank")["usage_share"]
          .quantile(pctl)
          .reset_index()
          .rename(columns={"usage_share": "cap"})
    )
    # Clamp caps to sane bounds
    caps["cap"] = caps["cap"].clip(lower=0.03, upper=0.40)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    caps.to_csv(out_path, index=False)
    return out_path


def main():
    p = argparse.ArgumentParser(description="Learn usage caps from historical box scores")
    p.add_argument("--box", default=os.path.join(os.path.dirname(__file__), "player_scores.csv"), help="CSV with historical player box including fga")
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "usage_caps.csv"), help="Output CSV path for learned caps")
    p.add_argument("--pctl", type=float, default=0.90, help="Percentile for cap (0..1)")
    p.add_argument("--min-minutes", type=float, default=6.0, help="Minimum minutes to include a player-game")
    args = p.parse_args()
    learn_usage_caps(args.box, args.out, min_minutes=args.min_minutes, pctl=args.pctl)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

