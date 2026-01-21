import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo

# Ensure parent (NBA_Simulation) is importable when run from subfolder
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Reuse helper to get today's games list
try:
    from simulation_engine import load_today_context
except Exception:
    load_today_context = None


def _read_history(data_dir: str) -> pd.DataFrame:
    # Prefer consolidated history if present
    candidates = [
        os.path.join(data_dir, "nba_game_scores.csv"),
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    # Fallback: stitch from backtest cache games files
    cache_dir = os.path.join(data_dir, "Backtest_cache")
    rows = []
    if os.path.isdir(cache_dir):
        for f in os.listdir(cache_dir):
            if f.startswith("sim_backtest_games_") and f.endswith(".csv"):
                try:
                    df = pd.read_csv(os.path.join(cache_dir, f))
                    rows.append(df)
                except Exception:
                    continue
    if rows:
        return pd.concat(rows, ignore_index=True)
    return pd.DataFrame()


def _normalize_history(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    # Try to locate team code columns
    home_col = next((c for c in [
        "home_code", "home", "home_team", "homeTeam", "HOME"
    ] if c in d.columns), None)
    away_col = next((c for c in [
        "visitor_code", "away", "away_team", "awayTeam", "AWAY", "visitor"
    ] if c in d.columns), None)
    # Locate points (actual results)
    hpts_col = next((c for c in [
        "home_points", "home_pts", "home_score", "home_total", "home_actual"
    ] if c in d.columns), None)
    apts_col = next((c for c in [
        "away_points", "away_pts", "away_score", "away_total", "away_actual", "visitor_score"
    ] if c in d.columns), None)
    # Locate date
    date_col = next((c for c in ["date", "date_pst", "game_date", "gameDate"] if c in d.columns), None)

    if not home_col or not away_col or not hpts_col or not apts_col:
        # Try to infer from known shapes
        # If backtest cache frame: has home, away, home_actual, away_actual
        if {"home", "away", "home_actual", "away_actual"}.issubset(set(d.columns)):
            home_col, away_col, hpts_col, apts_col = "home", "away", "home_actual", "away_actual"
        elif {"home_code", "visitor_code", "home_score", "visitor_score"}.issubset(set(d.columns)):
            home_col, away_col, hpts_col, apts_col = "home_code", "visitor_code", "home_score", "visitor_score"
        else:
            raise ValueError(f"Unable to locate team/points columns in history. Columns found: {list(d.columns)}")

    d = d[[c for c in [date_col, home_col, away_col, hpts_col, apts_col] if c]].copy()
    if date_col:
        d["date"] = pd.to_datetime(d[date_col], errors="coerce")
    else:
        d["date"] = pd.NaT
    d.rename(columns={home_col: "home", away_col: "away", hpts_col: "home_pts", apts_col: "away_pts"}, inplace=True)
    d["home"] = d["home"].astype(str)
    d["away"] = d["away"].astype(str)
    d["home_pts"] = pd.to_numeric(d["home_pts"], errors="coerce")
    d["away_pts"] = pd.to_numeric(d["away_pts"], errors="coerce")
    d = d.dropna(subset=["home", "away", "home_pts", "away_pts"]).reset_index(drop=True)
    return d


def todays_games(data_dir: str) -> pd.DataFrame:
    if load_today_context is not None:
        try:
            _, todays, _ = load_today_context()
            if todays is not None and not todays.empty:
                return todays[["home_code", "visitor_code"]].rename(columns={"home_code": "home", "visitor_code": "away"}).astype(str)
        except Exception:
            pass
    # Fallback: try filtered_game_scores.csv for today’s list (not guaranteed)
    return pd.DataFrame(columns=["home", "away"])


def _exp_weights(n: int, half_life: float) -> np.ndarray:
    # Newer observations get higher weight; index 0 oldest, n-1 newest
    if half_life is None or half_life <= 0:
        return np.ones(n) / max(n, 1)
    lam = np.log(2.0) / float(half_life)
    idx = np.arange(n)
    w = np.exp(lam * (idx - (n - 1)))
    w = w / w.sum() if w.sum() else np.ones(n) / n
    return w


def predict_from_history(hist: pd.DataFrame, games_today: pd.DataFrame, last_n: int = 10, home_adv: float = 1.5, w_for: float = 0.5, use_decay: bool = False, half_life: float = 0.0) -> pd.DataFrame:
    # Build per-team rolling averages
    df = hist.copy()
    # Melt into team perspective
    home_rows = df[["date", "home", "home_pts", "away" ]].rename(columns={"home": "team", "away": "opp", "home_pts": "pts"})
    away_rows = df[["date", "away", "away_pts", "home" ]].rename(columns={"away": "team", "home": "opp", "away_pts": "pts"})
    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Compute team-for and team-against averages over last N or with decay
    # Join opponent same-game points to get against (vectorized via merge on game pairs)
    # Simpler: compute mean against from the other side per game
    merge_home = df[["home", "away", "away_pts"]].rename(columns={"home": "team", "away": "opp", "away_pts": "pts_against"})
    merge_away = df[["away", "home", "home_pts"]].rename(columns={"away": "team", "home": "opp", "home_pts": "pts_against"})
    against_long = pd.concat([merge_home, merge_away], ignore_index=True)
    long2 = long.merge(against_long, on=["team", "opp"], how="left")
    # Normalize merged column name
    if "pts_against" not in long2.columns:
        if "pts_against_y" in long2.columns:
            long2["pts_against"] = long2["pts_against_y"]
        elif "pts_against_x" in long2.columns:
            long2["pts_against"] = long2["pts_against_x"]

    def _roll_avg(g: pd.DataFrame, col: str) -> pd.Series:
        if not use_decay:
            return g[col].rolling(last_n, min_periods=1).mean()
        # Apply exponential weights over last_n window (or full length if smaller)
        vals = g[col].to_numpy(dtype=float)
        out = np.empty(len(vals))
        for i in range(len(vals)):
            start = 0 if i + 1 < last_n else i + 1 - last_n
            window = vals[start:i+1]
            w = _exp_weights(len(window), half_life if half_life > 0 else max(3, last_n//2))
            out[i] = float(np.dot(window, w))
        return pd.Series(out, index=g.index)

    avgs = long2.groupby("team").apply(lambda g: pd.DataFrame({
        "for_avg": _roll_avg(g, "pts"),
        "against_avg": _roll_avg(g, "pts_against"),
    })).reset_index(level=0, drop=True)
    long2 = pd.concat([long2, avgs], axis=1)

    # Last known averages per team
    team_last = long2.groupby("team").tail(1)[["team", "for_avg", "against_avg"]].set_index("team")

    # Predict for today's games
    out_rows = []
    for _, r in games_today.iterrows():
        home = str(r["home"]) ; away = str(r["away"]) 
        h_for = float(team_last.loc[home, "for_avg"]) if home in team_last.index else np.nan
        h_opp_def = float(team_last.loc[away, "against_avg"]) if away in team_last.index else np.nan
        a_for = float(team_last.loc[away, "for_avg"]) if away in team_last.index else np.nan
        a_opp_def = float(team_last.loc[home, "against_avg"]) if home in team_last.index else np.nan

        # Blend weights for offense vs opponent defense
        wf = float(w_for)
        wa = 1.0 - wf
        home_pred = (wf * h_for + wa * a_opp_def) + home_adv
        away_pred = (wf * a_for + wa * h_opp_def)
        out_rows.append({"home": home, "away": away, "home_pred": home_pred, "away_pred": away_pred})

    preds = pd.DataFrame(out_rows)
    # Try to apply learned adjustments if available
    try:
        here = os.path.dirname(__file__)
        adj = pd.read_csv(os.path.join(here, "team_adjustments.csv"))
        params = pd.read_csv(os.path.join(here, "model_adjustments.csv"))
        off_map = dict(zip(adj["team"].astype(str), pd.to_numeric(adj["off_adj"], errors="coerce")))
        def_map = dict(zip(adj["team"].astype(str), pd.to_numeric(adj["def_adj"], errors="coerce")))
        homeadv_map = dict(zip(adj["team"].astype(str), pd.to_numeric(adj.get("home_adv_adj", pd.Series([0]*len(adj))), errors="coerce")))
        bias_home = float(pd.to_numeric(params.loc[params["param"]=="bias_home","value"], errors="coerce").fillna(0.0).iloc[0])
        bias_away = float(pd.to_numeric(params.loc[params["param"]=="bias_away","value"], errors="coerce").fillna(0.0).iloc[0])
        # If tuned w_for/home_adv exist, override current call context for future runs
        try:
            wf_row = params.loc[params["param"]=="w_for","value"]
            ha_row = params.loc[params["param"]=="home_adv","value"]
            if not wf_row.empty:
                w_for = float(wf_row.iloc[0])
            if not ha_row.empty:
                home_adv = float(ha_row.iloc[0])
        except Exception:
            pass
        preds["home_final"] = preds["home_pred"] + bias_home + preds["home"].map(homeadv_map).fillna(0.0) + preds["home"].map(off_map).fillna(0.0) - preds["away"].map(def_map).fillna(0.0)
        preds["away_final"] = preds["away_pred"] + bias_away + preds["away"].map(off_map).fillna(0.0) - preds["home"].map(def_map).fillna(0.0)
        # Replace base predictions with adjusted
        preds["home_pred"] = preds["home_final"]
        preds["away_pred"] = preds["away_final"]
        preds.drop(columns=["home_final","away_final"], inplace=True)
    except Exception:
        pass
    return preds


def _season_start_for_date(dt_pst: datetime) -> datetime:
    year = dt_pst.year
    if dt_pst.month < 10 or (dt_pst.month == 10 and dt_pst.day < 23):
        year -= 1
    return datetime(year, 10, 23, tzinfo=ZoneInfo("America/Los_Angeles"))


def backtest_from_history(hist: pd.DataFrame, start_dt: datetime, last_n: int = 10, home_adv: float = 1.5, w_for: float = 0.5, use_decay: bool = False, half_life: float = 0.0) -> pd.DataFrame:
    # Ensure date is datetime and normalized to naive local timestamps
    df = hist.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # If timezone-aware, convert to America/Los_Angeles then drop tz
    try:
        df["date"] = df["date"].dt.tz_convert("America/Los_Angeles")
    except Exception:
        # If not tz-aware, ignore
        pass
    try:
        df["date"] = df["date"].dt.tz_localize(None)
    except Exception:
        pass
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    results = []
    all_dates = sorted(pd.to_datetime(df["date"]).dt.normalize().unique())
    # Normalize start date to Timestamp (no time, local naive)
    start_ts = pd.Timestamp(start_dt)
    try:
        start_ts = start_ts.tz_convert("America/Los_Angeles")
    except Exception:
        pass
    try:
        start_ts = start_ts.tz_localize(None)
    except Exception:
        pass
    start_ts = start_ts.normalize()
    for d in all_dates:
        d_ts = pd.Timestamp(d).tz_localize(None).normalize()
        if d_ts < start_ts:
            continue
        # History strictly before this date
        prior = df[df["date"].dt.normalize() < d_ts]
        if prior.empty:
            continue
        # Games on this date
        games_d = df[df["date"].dt.normalize() == d_ts][["home","away","home_pts","away_pts"]]
        preds = predict_from_history(prior, games_d[["home","away"]], last_n=last_n, home_adv=home_adv, w_for=w_for, use_decay=use_decay, half_life=half_life)
        if preds is None or preds.empty:
            continue
        merged = games_d.merge(preds, on=["home","away"], how="left")
        merged["date"] = d_ts
        merged["home_error"] = merged["home_pred"] - merged["home_pts"]
        merged["away_error"] = merged["away_pred"] - merged["away_pts"]
        results.append(merged)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame(columns=["date","home","away","home_pts","away_pts","home_pred","away_pred","home_error","away_error"])


def main():
    # Paths
    data_dir = PARENT_DIR
    out_dir = os.path.dirname(__file__)  # Predict_Results folder
    hist_raw = _read_history(data_dir)
    if hist_raw is None or hist_raw.empty:
        print("No historical game results found. Expected nba_game_scores.csv or Backtest_cache.")
        sys.exit(1)
    hist = _normalize_history(hist_raw)

    # Get today's games
    tg = todays_games(data_dir)
    if tg.empty:
        print("No games found for today.")
        sys.exit(0)

    preds = predict_from_history(hist, tg, last_n=int(os.environ.get("PR_LAST_N", 10)), home_adv=float(os.environ.get("PR_HOME_ADV", 1.5)))
    out_path = os.path.join(out_dir, "predicted_game_scores_today.csv")
    preds.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")

    # Backtest from season start (Oct 23) to today
    try:
        today_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
        season_start = _season_start_for_date(today_pst)
        bt = backtest_from_history(hist, season_start, last_n=int(os.environ.get("PR_LAST_N", 10)), home_adv=float(os.environ.get("PR_HOME_ADV", 1.5)))
        bt_path = os.path.join(out_dir, "backtest_results.csv")
        bt.to_csv(bt_path, index=False)
        if not bt.empty:
            home_mae = float(bt["home_error"].abs().mean())
            away_mae = float(bt["away_error"].abs().mean())
            summary = pd.DataFrame([
                {"metric": "home_mae", "value": home_mae},
                {"metric": "away_mae", "value": away_mae},
                {"metric": "games", "value": int(len(bt))},
            ])
            summary_path = os.path.join(out_dir, "backtest_summary.csv")
            summary.to_csv(summary_path, index=False)
            print(f"Backtest summary: home_mae={home_mae:.2f}, away_mae={away_mae:.2f} (n={len(bt)})")
            print(f"Wrote {bt_path} and {summary_path}")
            # Learn adjustments from backtest and write to folder
            try:
                from Predict_Results.Learn_Adjustments import learn_team_adjustments
            except Exception:
                from Learn_Adjustments import learn_team_adjustments  # when run inside folder
            adj_path = learn_team_adjustments(bt_path, out_dir)
            print(f"Learned and saved adjustments to {adj_path}")
        else:
            print("Backtest produced no rows (insufficient prior data).")
    except Exception as e:
        print(f"Backtest failed: {e}")


if __name__ == "__main__":
    main()
