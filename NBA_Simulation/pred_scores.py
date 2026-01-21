import os
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

# Backtest toggle: set True to run backtest on the last N game days automatically
BACKTEST_TOGGLE = True
BACKTEST_LAST_N_GAME_DAYS = 5

# Default model hyperparameters (can be overridden by auto-tune)
DEFAULT_HPARAMS = {
    # alpha no longer used (PPM from season to date only)
    "alpha": 0.0,
    "k": 5.0,                # optional shrinkage toward league_ppm for low samples
    "sigma": 0.25,           # lognormal volatility for Poisson rate
    "pace_clip": (0.9, 1.1), # clamp for pace factor
    "opp_def_clip": (0.9, 1.1),
    "ml_alpha": 0.5,         # blend weight for ML PPM vs season PPM
    "ridge_lambda": 0.5,     # ridge regularization for linear model
    # Role-based scaling
    "role_start_min": 28.0,  # minutes threshold to be considered starter
    "role_rot_min": 18.0,    # minutes threshold to be considered rotation
    "starter_min_scale": 1.04,
    "rotation_min_scale": 1.02,
    "bench_min_scale": 0.98,
    "starter_ppm_scale": 1.03,
    "rotation_ppm_scale": 1.01,
    "bench_ppm_scale": 0.99,
}


def _season_start_for_date(dt_pst: datetime) -> datetime:
    # Season start on Oct 23 (PST). If month < Oct, use Oct 23 of previous year.
    year = dt_pst.year
    if dt_pst.month < 10 or (dt_pst.month == 10 and dt_pst.day < 23):
        year -= 1
    return datetime(year, 10, 23, tzinfo=ZoneInfo("America/Los_Angeles"))


def season_player_ppm(ps_hist: pd.DataFrame, season_start_pst: datetime) -> pd.DataFrame:
    df = ps_hist.copy()
    if "game_date_pst" in df.columns:
        df = df[df["game_date_pst"] >= pd.Timestamp(season_start_pst)]
    # numeric safety
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    agg = df.groupby("player.id").agg(points_sum=("points", "sum"), minutes_sum=("min", "sum"), games_played=("game.id", "nunique")).reset_index()
    agg["ppm_season"] = agg["points_sum"] / agg["minutes_sum"].replace(0, pd.NA)
    return agg[["player.id", "ppm_season", "games_played"]]


def season_player_per_minute(ps_hist: pd.DataFrame, season_start_pst: datetime, stat_cols: list) -> pd.DataFrame:
    df = ps_hist.copy()
    if "game_date_pst" in df.columns:
        df = df[df["game_date_pst"] >= pd.Timestamp(season_start_pst)]
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    stat_col = next((c for c in stat_cols if c in df.columns), None)
    if stat_col is None:
        return pd.DataFrame(columns=["player.id", "per_min", "games_played"]).assign(per_min=np.nan)
    df[stat_col] = pd.to_numeric(df[stat_col], errors="coerce")
    agg = df.groupby("player.id").agg(stat_sum=(stat_col, "sum"), minutes_sum=("min", "sum"), games_played=("game.id", "nunique")).reset_index()
    agg["per_min"] = agg["stat_sum"] / agg["minutes_sum"].replace(0, pd.NA)
    return agg[["player.id", "per_min", "games_played"]]


def _select_rating_features(df: pd.DataFrame) -> list:
    # Prefer offensive z-score columns if present; fall back to any *_z
    preferred = [
        "Outside Scoring_z", "Three-Point Shot_z", "Mid-Range Shot_z", "Close Shot_z",
        "Free Throw_z", "Playmaking_z", "Pass Accuracy_z", "Ball Handle_z",
        "Speed_z", "Acceleration_z", "Offensive Consistency_z", "Shot IQ_z"
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = [c for c in df.columns if c.endswith("_z")]
    return cols[:20]


def train_ppm_model(ratings_clean: pd.DataFrame, ps_hist: pd.DataFrame, season_start_pst: datetime, ridge_lambda: float = 0.5):
    # Prepare training frame: season PPM target merged with ratings on player_name_key
    df = ps_hist.copy()
    df = df[df["game_date_pst"] >= pd.Timestamp(season_start_pst)] if "game_date_pst" in df.columns else df
    # ensure numeric
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")
    agg = df.groupby(["player.id", "player_name_key"], as_index=False).agg(points_sum=("points","sum"), minutes_sum=("min","sum"))
    agg = agg[agg["minutes_sum"] > 0]
    agg["ppm"] = agg["points_sum"] / agg["minutes_sum"]

    if ratings_clean is None or ratings_clean.empty or agg.empty:
        return None

    feats = _select_rating_features(ratings_clean)
    tr = agg.merge(ratings_clean[["player_name_key"] + feats], on="player_name_key", how="left").dropna(subset=feats)
    if tr.empty:
        return None

    # Build X, y with ridge closed form: (X^T W X + lambda I)^{-1} X^T W y
    import numpy as np
    X = tr[feats].astype(float).to_numpy()
    y = tr["ppm"].astype(float).to_numpy()
    w = tr["minutes_sum"].astype(float).to_numpy()
    w = w / (w.mean() if w.mean() else 1.0)
    # add bias term
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    W = np.diag(w)
    lam = float(max(0.0, ridge_lambda))
    I = np.eye(Xb.shape[1]); I[0,0] = 0.0  # don't regularize bias
    try:
        beta = np.linalg.solve(Xb.T @ W @ Xb + lam * I, Xb.T @ W @ y)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(Xb, y, rcond=None)[0]
    model = {"beta": beta, "features": feats}
    return model


def predict_ppm_from_ratings(model: dict, ratings_frame: pd.DataFrame, default_ppm: float) -> pd.Series:
    if not model:
        return pd.Series(default_ppm, index=ratings_frame.index)
    feats = model["features"]
    beta = model["beta"]
    X = ratings_frame[feats].astype(float).fillna(0.0).to_numpy()
    Xb = np.hstack([np.ones((X.shape[0],1)), X])
    yhat = Xb @ beta
    return pd.Series(np.clip(yhat, 0.2, 2.5), index=ratings_frame.index)


DATA_DIR = os.path.dirname(__file__)


def read_inputs():
    ratings_path = os.path.join(DATA_DIR, "2KRatings_Data.xlsx")
    games_filtered_path = os.path.join(DATA_DIR, "filtered_game_scores.csv")
    games_all_path = os.path.join(DATA_DIR, "nba_game_scores.csv")
    player_stats_path = os.path.join(DATA_DIR, "player_scores.csv")

    ratings = pd.read_excel(ratings_path)
    games_filtered = pd.read_csv(games_filtered_path)
    games_all = pd.read_csv(games_all_path)
    player_stats = pd.read_csv(player_stats_path)
    return ratings, games_filtered, games_all, player_stats


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    df = ratings.copy()
    if "Player" not in df.columns:
        return pd.DataFrame(columns=["player_name_key"])  # fallback
    repl = {"--": np.nan, "-": np.nan, "N/A": np.nan, "n/a": np.nan, "": np.nan}
    for c in df.columns:
        if c == "Player":
            continue
        df[c] = df[c].replace(repl)
        df[c] = pd.to_numeric(df[c], errors="ignore")
    num_cols = [c for c in df.columns if c != "Player" and np.issubdtype(df[c].dtype, np.number)]
    z_df = pd.DataFrame(index=df.index)
    for c in num_cols:
        col = df[c].astype(float)
        mu, sd = col.mean(skipna=True), col.std(skipna=True)
        z_df[c + "_z"] = 0.0 if (pd.isna(sd) or sd == 0) else (col - mu) / sd
    if z_df.empty:
        z_df["z_all_mean"] = 0.0
    else:
        z_df["z_all_mean"] = z_df.mean(axis=1, skipna=True)
    out = pd.concat([df[["Player"]], z_df], axis=1)
    out["player_name_key"] = out["Player"].str.strip().str.lower()
    return out.drop(columns=["Player"], errors="ignore")


def enrich_player_stats_with_game_dates(player_stats: pd.DataFrame, games_all: pd.DataFrame) -> pd.DataFrame:
    g = games_all[["game.id", "game_date", "visitor_code", "home_code"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"], utc=True, errors="coerce")
    g["game_date_pst"] = g["game_date"].dt.tz_convert("America/Los_Angeles")
    ps = player_stats.copy()
    ps["player_name_key"] = (
        ps["player.firstname"].astype(str).str.strip()
        + " "
        + ps["player.lastname"].astype(str).str.strip()
    ).str.lower()
    ps = ps.merge(g, how="left", on="game.id")
    ps["opponent_code"] = np.where(ps["team.code"] == ps["home_code"], ps["visitor_code"], ps["home_code"])
    return ps


def get_todays_games_pst(games_all: pd.DataFrame, today_pst: datetime) -> pd.DataFrame:
    df = games_all.copy()
    df["game_date_utc"] = pd.to_datetime(df["game_date"], utc=True, errors="coerce")
    df["game_date_pst"] = df["game_date_utc"].dt.tz_convert("America/Los_Angeles")
    df["game_date_pst_date"] = df["game_date_pst"].dt.date
    return df[df["game_date_pst_date"] == today_pst.date()].copy()


def build_today_rosters(ps: pd.DataFrame, todays_games: pd.DataFrame, start_from_pst: datetime = None) -> pd.DataFrame:
    df = ps.copy()
    # For backtest cutoff, only use games BEFORE the target date to infer rosters
    if start_from_pst is not None and "game_date_pst" in df.columns:
        df = df[df["game_date_pst"] < pd.Timestamp(start_from_pst)].copy()
    df["team.code"] = df["team.code"].astype(str).str.strip().str.upper()
    todays_codes = pd.unique(pd.concat([
        todays_games["home_code"].astype(str).str.strip().str.upper(),
        todays_games["visitor_code"].astype(str).str.strip().str.upper()
    ]))
    df = df[df["team.code"].isin(todays_codes)]
    df = df.sort_values(["player.id", "game_date"]).copy()
    mm5 = df.groupby("player.id")["min"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    df["min_mean5"] = mm5.values
    last = df.groupby("player.id").tail(1)[["player.id", "team.code", "player.firstname", "player.lastname", "min_mean5"]]
    last["min_mean5"].fillna(0.0, inplace=True)
    rosters = last.sort_values(["team.code", "min_mean5"], ascending=[True, False])
    rosters = rosters.groupby("team.code").head(10).reset_index(drop=True)
    rosters["player_name_key"] = (
        rosters["player.firstname"].astype(str).str.strip()
        + " "
        + rosters["player.lastname"].astype(str).str.strip()
    ).str.lower()
    return rosters


def minutes_from_recent_and_ratings(ps: pd.DataFrame, rosters: pd.DataFrame, ratings_clean: pd.DataFrame = None) -> pd.DataFrame:
    df = ps.sort_values("game_date").copy()
    df = df.groupby("player.id").apply(
        lambda g: g.assign(min_mean5=g["min"].shift(1).rolling(5, min_periods=1).mean())
    ).reset_index(drop=True)

    def _trend_last5(g: pd.DataFrame):
        g = g.dropna(subset=["min"]).copy()
        g5 = g.tail(5)
        n = len(g5)
        if n >= 2:
            x = np.arange(1, n + 1, dtype=float)
            y = g5["min"].astype(float).values
            try:
                b, a = np.polyfit(x, y, 1)
                return a + b * (n + 1)
            except Exception:
                return np.nan
        return np.nan

    trend = df.groupby("player.id").apply(_trend_last5).rename("min_trend_pred").reset_index()

    def _ema_last5(g: pd.DataFrame):
        s = g["min"].shift(1).astype(float)
        return s.ewm(span=5, adjust=False, min_periods=1).mean().iloc[-1]

    ema = df.groupby("player.id").apply(_ema_last5).rename("min_ema5").reset_index()

    latest = (
        df.groupby("player.id").tail(1)[["player.id", "team.code", "player.firstname", "player.lastname", "min_mean5"]]
        .merge(trend, on="player.id", how="left")
        .merge(ema, on="player.id", how="left")
    )
    latest["player_name_key"] = (
        latest["player.firstname"].astype(str).str.strip()
        + " "
        + latest["player.lastname"].astype(str).str.strip()
    ).str.lower()

    feat_df = rosters.merge(latest, on=["player_name_key"], how="left")
    if ratings_clean is not None and not ratings_clean.empty:
        feat_df = feat_df.merge(ratings_clean, on="player_name_key", how="left")

    # Ensure team.code exists after merges
    if "team.code" not in feat_df.columns:
        if "team.code_x" in feat_df.columns:
            feat_df["team.code"] = feat_df["team.code_x"]
        elif "team.code_y" in feat_df.columns:
            feat_df["team.code"] = feat_df["team.code_y"]
        else:
            feat_df = feat_df.merge(rosters[["player_name_key","team.code"]], on="player_name_key", how="left")
    dedup_key = "player.id" if "player.id" in feat_df.columns else "player_name_key"
    dedup_key = "player.id" if "player.id" in feat_df.columns else "player_name_key"
    if "min_mean5" in feat_df.columns:
        feat_df = feat_df.sort_values([dedup_key, "min_mean5"], ascending=[True, False]).drop_duplicates(subset=[dedup_key], keep="first")

    # Guard against accidental dup rows after merges
    feat_df = feat_df.drop_duplicates(subset=["team.code", dedup_key], keep="first")
    if "min_mean5" not in feat_df.columns:
        feat_df["min_mean5"] = np.nan
    trend_or_mean = feat_df["min_trend_pred"].where(feat_df["min_trend_pred"].notna(), feat_df["min_mean5"])
    ema_or_mean = feat_df["min_ema5"].where(feat_df["min_ema5"].notna(), feat_df["min_mean5"])
    base = 0.6 * trend_or_mean.fillna(0) + 0.4 * ema_or_mean.fillna(0)
    base = base.where(base.notna() & (base > 0), feat_df["min_mean5"]).fillna(22.0).astype(float)

    # ratings minutes boost
    st = feat_df.get("Stamina_z", pd.Series(0.0, index=feat_df.index)).fillna(0.0).astype(float)
    hu = feat_df.get("Hustle_z", pd.Series(0.0, index=feat_df.index)).fillna(0.0).astype(float)
    dc = feat_df.get("Defensive Consistency_z", pd.Series(0.0, index=feat_df.index)).fillna(0.0).astype(float)
    boost = (1.0 + 0.04 * st + 0.02 * hu + 0.02 * dc).clip(0.9, 1.15)
    base = base * boost

    feat_df["pred_minutes_raw"] = base.values

    def norm(group: pd.DataFrame) -> pd.DataFrame:
        lower = 0.0
        alloc = group["pred_minutes_raw"].astype(float).values
        s = alloc.sum()
        alloc = 240.0 * alloc / s if s > 0 else np.full(len(alloc), 240.0 / max(len(alloc), 1))
        alloc = np.maximum(alloc, lower)
        order = np.argsort(-alloc)
        core = min(8, len(order))
        for idx in order[:core]:
            if alloc[idx] < 22.0:
                alloc[idx] = 22.0
        for idx in order[core:]:
            if alloc[idx] < 6.0:
                alloc[idx] = 6.0
        s = alloc.sum()
        alloc = 240.0 * alloc / s if s > 0 else np.full(len(alloc), 240.0 / max(len(alloc), 1))
        alloc = np.maximum(alloc, lower)
        for _ in range(12):
            diff = 240.0 - float(alloc.sum())
            if abs(diff) < 1e-6:
                break
            mask = np.ones_like(alloc, dtype=bool) if diff > 0 else (alloc > (lower + 1e-9))
            if not np.any(mask):
                break
            alloc[mask] = np.maximum(alloc[mask] + diff / mask.sum(), lower)
        group["pred_minutes"] = alloc
        return group

    feat_df = feat_df.groupby("team.code", group_keys=False).apply(norm)
    for basecol in ["player.id", "player.firstname", "player.lastname"]:
        if basecol not in feat_df.columns:
            cand = [c for c in feat_df.columns if c.startswith(basecol)]
            feat_df[basecol] = feat_df[cand[0]] if cand else np.nan
    return feat_df[[
        "team.code", "player.id", "player.firstname", "player.lastname",
        "pred_minutes", "pred_minutes_raw", "player_name_key"
    ]]


def ratings_to_ppm(ratings_clean: pd.DataFrame, roster_df: pd.DataFrame, base_ppm: float = 0.47) -> pd.Series:
    df = roster_df.merge(ratings_clean, on="player_name_key", how="left") if ratings_clean is not None and not ratings_clean.empty else roster_df.copy()
    comps = []
    for c in [
        "Outside Scoring_z", "Three-Point Shot_z", "Mid-Range Shot_z", "Close Shot_z",
        "Free Throw_z", "Playmaking_z", "Pass Accuracy_z", "Ball Handle_z"
    ]:
        if c in df:
            comps.append(df[c])
    z_off = pd.concat(comps, axis=1).mean(axis=1, skipna=True).fillna(0.0) if comps else pd.Series(0.0, index=df.index)
    k = 0.12
    ppm = base_ppm * np.exp(k * z_off)
    return pd.Series(np.clip(ppm, 0.3, 1.8), index=df.index)


def recent_player_ppm(ps: pd.DataFrame) -> pd.DataFrame:
    df = ps.sort_values("game_date").copy()
    df["ppm_game"] = df["points"] / df["min"].replace(0, np.nan)
    roll = df.groupby("player.id")["ppm_game"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).reset_index(level=0, drop=True)
    df["ppm_recent5"] = roll.values
    last = df.groupby("player.id").tail(1)[["player.id", "team.code", "ppm_recent5"]].copy()
    counts = df.groupby("player.id").size().rename("games_played").reset_index()
    return last.merge(counts, on="player.id", how="left")


def team_recent_factors(games_filtered: pd.DataFrame) -> dict:
    g = games_filtered.copy()
    g["home_score"] = pd.to_numeric(g["home_score"], errors="coerce")
    g["visitor_score"] = pd.to_numeric(g["visitor_score"], errors="coerce")
    g["total"] = g["home_score"] + g["visitor_score"]
    league_total = g["total"].mean()
    long = []
    for _, r in g.iterrows():
        if pd.isna(r.get("home_score")) or pd.isna(r.get("visitor_score")):
            continue
        long.append({"team": r["home_code"], "points": r["home_score"], "allowed": r["visitor_score"], "date": r.get("game_date_pacific")})
        long.append({"team": r["visitor_code"], "points": r["visitor_score"], "allowed": r["home_score"], "date": r.get("game_date_pacific")})
    ldf = pd.DataFrame(long)
    if ldf.empty:
        return {"league_total": league_total, "off": {}, "def": {}}
    ldf = ldf.sort_values("date")
    off = ldf.groupby("team")["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).groupby(level=0).tail(1)
    deff = ldf.groupby("team")["allowed"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).groupby(level=0).tail(1)
    return {"league_total": float(league_total) if league_total else 220.0,
            "off": off.rename("off5").to_dict(),
            "def": deff.rename("def5").to_dict()}


def simulate_games_from_ratings(todays_games: pd.DataFrame, ps: pd.DataFrame, ratings_clean: pd.DataFrame, sims: int = 1000, start_from_pst: datetime = None, hparams: dict = None):
    # Limit historical data used for features to games strictly before start_from_pst
    ps_hist = ps.copy()
    if start_from_pst is not None and "game_date_pst" in ps_hist.columns:
        ps_hist = ps_hist[ps_hist["game_date_pst"] < pd.Timestamp(start_from_pst)]

    rosters = build_today_rosters(ps, todays_games, start_from_pst=start_from_pst)
    if rosters.empty:
        return pd.DataFrame(), pd.DataFrame()
    minutes_today = minutes_from_recent_and_ratings(ps_hist, rosters, ratings_clean)

    # Build PPM purely from this season to date (no ratings)
    # Determine season start relative to the target date
    target_dt = pd.Timestamp(start_from_pst) if start_from_pst is not None else pd.Timestamp(datetime.now(ZoneInfo("America/Los_Angeles")))
    season_start = _season_start_for_date(target_dt)
    season_ppm_df = season_player_ppm(ps_hist, season_start)
    minutes_today = minutes_today.merge(season_ppm_df, on="player.id", how="left")

    # Hyperparameters
    hp = DEFAULT_HPARAMS.copy()
    if hparams:
        hp.update({k: v for k, v in hparams.items() if v is not None})

    # Dynamic league_ppm from ps_hist if available
    try:
        tmp = ps_hist.copy()
        tmp["min"] = pd.to_numeric(tmp["min"], errors="coerce")
        tmp["points"] = pd.to_numeric(tmp["points"], errors="coerce")
        league_ppm = (tmp["points"].sum() / tmp["min"].sum()) if tmp["min"].sum() and tmp["min"].sum() > 0 else 0.47
    except Exception:
        league_ppm = 0.47
    # Optional shrink toward league_ppm for low minutes/games
    k = float(hp.get("k", 5.0))
    n = minutes_today["games_played"].fillna(0).astype(float)
    base_ppm = minutes_today["ppm_season"].fillna(league_ppm)
    ppm_season_shrunk = (n / (n + k)) * base_ppm + (k / (n + k)) * league_ppm
    # ML model from ratings → ppm
    ml_model = train_ppm_model(ratings_clean, ps_hist, season_start, ridge_lambda=float(hp.get("ridge_lambda", 0.5)))
    # Prepare features for the players we're predicting today
    feats_cols = _select_rating_features(ratings_clean) if ratings_clean is not None and not ratings_clean.empty else []
    if feats_cols:
        ratings_feat = ratings_clean[["player_name_key"] + feats_cols].drop_duplicates("player_name_key")
        feat_frame = minutes_today.merge(ratings_feat, on="player_name_key", how="left")
    else:
        feat_frame = minutes_today.copy()
    ppm_ml = predict_ppm_from_ratings(ml_model, feat_frame if feats_cols else feat_frame.assign(), default_ppm=float(league_ppm))
    # Ensure alignment with minutes_today rows
    if hasattr(ppm_ml, 'reindex'):
        ppm_ml = ppm_ml.reindex(minutes_today.index)
    if len(ppm_ml) != len(minutes_today):
        # As a fallback, create a default vector
        ppm_ml = pd.Series(np.full(len(minutes_today), float(league_ppm)), index=minutes_today.index)
    ml_alpha = float(hp.get("ml_alpha", 0.5))
    ppm_blend = ((1.0 - ml_alpha) * ppm_season_shrunk.values) + (ml_alpha * ppm_ml.values)

    factors_df = pd.read_csv(os.path.join(DATA_DIR, "filtered_game_scores.csv"))
    if start_from_pst is not None and "game_date_pacific" in factors_df.columns:
        with pd.option_context('mode.use_inf_as_na', True):
            try:
                factors_df["game_date_pacific"] = pd.to_datetime(factors_df["game_date_pacific"], errors="coerce")
                factors_df = factors_df[factors_df["game_date_pacific"] < pd.Timestamp(start_from_pst)]
            except Exception:
                pass
    factors = team_recent_factors(factors_df)
    league_total = factors.get("league_total", 220.0)
    def team_total(team):
        off = factors["off"].get(team)
        deff = factors["def"].get(team)
        vals = [v for v in [off, deff] if v is not None]
        return np.mean(vals) if vals else league_total
    team_series = minutes_today["team.code"].astype(str)
    opp_series = minutes_today.get("opponent_code", pd.Series([None] * len(minutes_today)))
    pace_team = team_series.apply(lambda t: team_total(t))
    pace_opp = opp_series.astype(str).apply(lambda t: team_total(t) if t and t != "None" else league_total)
    pace_factor = ((pace_team + pace_opp) / 2.0) / max(league_total, 1e-6)
    lo, hi = hp.get("pace_clip", (0.9, 1.1))
    pace_factor = pace_factor.clip(lo, hi)
    opp_def = opp_series.astype(str).apply(lambda t: factors["def"].get(t, league_total))
    dlo, dhi = hp.get("opp_def_clip", (0.9, 1.1))
    opp_def_factor = (league_total / opp_def.replace(0, league_total)).clip(dlo, dhi)
    ppm_adj = ppm_blend * pace_factor.values * opp_def_factor.values

    # Role-based minutes scaling and re-normalization per team
    role_start_min = float(hp.get("role_start_min", 28.0))
    role_rot_min = float(hp.get("role_rot_min", 18.0))
    s_min_scale = float(hp.get("starter_min_scale", 1.04))
    r_min_scale = float(hp.get("rotation_min_scale", 1.02))
    b_min_scale = float(hp.get("bench_min_scale", 0.98))
    pm_arr = minutes_today["pred_minutes"].astype(float).values
    roles = np.where(pm_arr >= role_start_min, "starter", np.where(pm_arr >= role_rot_min, "rotation", "bench"))
    min_scale = np.where(roles == "starter", s_min_scale, np.where(roles == "rotation", r_min_scale, b_min_scale))
    minutes_today["pred_minutes"] = minutes_today["pred_minutes"].astype(float) * min_scale
    def _renorm(group: pd.DataFrame) -> pd.DataFrame:
        vals = group["pred_minutes"].astype(float).values
        s = vals.sum()
        if s > 0:
            group["pred_minutes"] = 240.0 * vals / s
        return group
    minutes_today = minutes_today.groupby("team.code", group_keys=False).apply(_renorm)

    # Role-based ppm scaling
    s_ppm_scale = float(hp.get("starter_ppm_scale", 1.03))
    r_ppm_scale = float(hp.get("rotation_ppm_scale", 1.01))
    b_ppm_scale = float(hp.get("bench_ppm_scale", 0.99))
    ppm_scale = np.where(roles == "starter", s_ppm_scale, np.where(roles == "rotation", r_ppm_scale, b_ppm_scale))
    minutes_today["ppm_final"] = ppm_adj * ppm_scale

    mu = minutes_today["pred_minutes"].astype(float).values * minutes_today["ppm_final"].astype(float).values
    rng = np.random.default_rng(42)
    sigma = float(hp.get("sigma", 0.25))
    multipliers = rng.lognormal(mean=0.0, sigma=sigma, size=(sims, len(minutes_today)))
    rate = multipliers * mu[None, :]
    points_sim = rng.poisson(rate)

    team_index = {t: np.where(minutes_today["team.code"].values == t)[0] for t in minutes_today["team.code"].unique()}
    team_stats = {}
    for t, idx in team_index.items():
        totals = points_sim[:, idx].sum(axis=1)
        team_stats[t] = {
            "mean": float(np.mean(totals)),
            "p05": float(np.percentile(totals, 5)),
            "p50": float(np.percentile(totals, 50)),
            "p95": float(np.percentile(totals, 95)),
        }

    rows = []
    for _, r in todays_games.iterrows():
        home = r["home_code"]; away = r["visitor_code"]
        rows.append({
            "game.id": r["game.id"],
            "game_date_pst": r.get("game_date_pst", r.get("game_date", "")),
            "home": home,
            "away": away,
            "home_pred_points": team_stats.get(home, {}).get("mean"),
            "away_pred_points": team_stats.get(away, {}).get("mean"),
            "home_p05": team_stats.get(home, {}).get("p05"),
            "home_p50": team_stats.get(home, {}).get("p50"),
            "home_p95": team_stats.get(home, {}).get("p95"),
            "away_p05": team_stats.get(away, {}).get("p05"),
            "away_p50": team_stats.get(away, {}).get("p50"),
            "away_p95": team_stats.get(away, {}).get("p95"),
        })
    games_pred = pd.DataFrame(rows)
    games_pred["proj_spread_home_minus_away"] = games_pred["home_pred_points"].astype(float) - games_pred["away_pred_points"].astype(float)
    games_pred["model_total"] = games_pred["home_pred_points"].astype(float) + games_pred["away_pred_points"].astype(float)

    player_means = points_sim.mean(axis=0)

    # Also estimate rebounds and assists using season per-minute with shrinkage
    season_start_pst = _season_start_for_date(pd.Timestamp(start_from_pst) if start_from_pst is not None else datetime.now(ZoneInfo("America/Los_Angeles")))
    reb_hist = season_player_per_minute(ps_hist, season_start_pst, ["totReb", "rebounds.total", "reb", "Rebounds"])
    ast_hist = season_player_per_minute(ps_hist, season_start_pst, ["assists", "totAst", "ast", "Assists"])
    # League priors from ps_hist
    try:
        total_min = pd.to_numeric(ps_hist.get("min"), errors="coerce").sum()
        reb_sum = pd.to_numeric(ps_hist.get("totReb", ps_hist.get("rebounds.total", pd.Series([]))), errors="coerce").sum()
        ast_sum = pd.to_numeric(ps_hist.get("assists", ps_hist.get("totAst", pd.Series([]))), errors="coerce").sum()
        league_reb_per_min = (reb_sum / total_min) if total_min and total_min > 0 else 0.2
        league_ast_per_min = (ast_sum / total_min) if total_min and total_min > 0 else 0.12
    except Exception:
        league_reb_per_min = 0.2
        league_ast_per_min = 0.12

    def _shrink_per_min(hist_df: pd.DataFrame, prior: float, k: float = 5.0) -> pd.Series:
        if hist_df is None or hist_df.empty:
            return pd.Series(prior, index=minutes_today.index)
        m = minutes_today[["player.id"]].merge(hist_df, on="player.id", how="left")
        g = pd.to_numeric(m.get("games_played"), errors="coerce").fillna(0.0)
        s = pd.to_numeric(m.get("per_min"), errors="coerce").fillna(prior)
        w = g / (g + k)
        return (w * s + (1 - w) * prior).astype(float)

    rpm_season_shrunk = _shrink_per_min(reb_hist, league_reb_per_min, k=float(hp.get("k", 5.0)))
    apm_season_shrunk = _shrink_per_min(ast_hist, league_ast_per_min, k=float(hp.get("k", 5.0)))

    # Simple context adjustment: scale by pace_factor similar to points
    rpm_adj = rpm_season_shrunk * pace_factor.values
    apm_adj = apm_season_shrunk * pace_factor.values

    pred_reb = minutes_today["pred_minutes"].astype(float).values * rpm_adj.values
    pred_ast = minutes_today["pred_minutes"].astype(float).values * apm_adj.values

    minutes_today_out = minutes_today.copy()
    minutes_today_out["pred_points"] = player_means
    minutes_today_out["pred_reb"] = pred_reb
    minutes_today_out["pred_ast"] = pred_ast
    minutes_today_out = minutes_today_out.sort_values(["player.id", "pred_points"], ascending=[True, False]).drop_duplicates(subset=["player.id"], keep="first")
    return games_pred, minutes_today_out[["team.code", "player.id", "player.firstname", "player.lastname", "pred_minutes", "ppm_final", "pred_points", "pred_reb", "pred_ast"]]


def backtest_player_points(games_all: pd.DataFrame, player_stats: pd.DataFrame, ratings_clean: pd.DataFrame,
                           start_date: str = None, end_date: str = None, sims: int = 1000,
                           dates_list=None, hparams: dict = None) -> pd.DataFrame:
    # Prepare enriched stats and calendar of PST dates
    ps = enrich_player_stats_with_game_dates(player_stats, games_all)
    g = games_all.copy()
    g["game_date_utc"] = pd.to_datetime(g["game_date"], utc=True, errors="coerce")
    g["game_date_pst"] = g["game_date_utc"].dt.tz_convert("America/Los_Angeles")
    g["pst_date"] = g["game_date_pst"].dt.date

    if dates_list is not None and len(dates_list) > 0:
        unique_dates = sorted(set(pd.to_datetime(pd.Series(dates_list)).dt.date.tolist()))
    else:
        if start_date:
            start_d = pd.to_datetime(start_date).date()
            g = g[g["pst_date"] >= start_d]
        if end_date:
            end_d = pd.to_datetime(end_date).date()
            g = g[g["pst_date"] <= end_d]
        unique_dates = sorted(g["pst_date"].dropna().unique().tolist())

    # Build actuals and last-game baseline helper frames
    ps = ps.sort_values(["player.id", "game_date_pst"]) if "game_date_pst" in ps.columns else ps.sort_values(["player.id", "game_date"]) 
    ps["pst_date"] = ps["game_date_pst"].dt.date if "game_date_pst" in ps.columns else pd.to_datetime(ps["game_date"], errors="coerce").dt.date
    ps["last_game_points"] = ps.groupby("player.id")["points"].shift(1)
    results = []
    empty_dates_info = []

    for d in unique_dates:
        dt = pd.Timestamp(d, tz=ZoneInfo("America/Los_Angeles"))
        todays_games = get_todays_games_pst(games_all, dt)
        if todays_games.empty:
            continue
        games_pred, player_points_today = simulate_games_from_ratings(todays_games, ps, ratings_clean, sims=sims, start_from_pst=dt, hparams=hparams)
        if player_points_today.empty:
            empty_dates_info.append((d, 0, 0, 0, 'no_predictions'))
            continue

        # Actuals for that date
        played_today = ps[ps["pst_date"] == d].copy()
        # Build actuals including rebounds/assists if present
        actuals = played_today[["player.id", "team.code", "player.firstname", "player.lastname"]].copy()
        # Map actual stats with fallback column names
        def _pick_col(df, candidates):
            for c in candidates:
                if c in df.columns:
                    return c
            return None
        pts_col = _pick_col(played_today, ["points"]) or "points"
        reb_col = _pick_col(played_today, ["totReb", "rebounds.total", "reb", "Rebounds"])  # optional
        ast_col = _pick_col(played_today, ["assists", "totAst", "ast", "Assists"])          # optional
        actuals["actual_points"] = pd.to_numeric(played_today.get(pts_col), errors="coerce")
        if reb_col:
            actuals["actual_reb"] = pd.to_numeric(played_today.get(reb_col), errors="coerce")
        if ast_col:
            actuals["actual_ast"] = pd.to_numeric(played_today.get(ast_col), errors="coerce")
        # Normalize ids for robust merge
        player_points_today["player.id"] = pd.to_numeric(player_points_today["player.id"], errors="coerce")
        actuals["player.id"] = pd.to_numeric(actuals["player.id"], errors="coerce")
        player_points_today = player_points_today.dropna(subset=["player.id"]).copy()
        actuals = actuals.dropna(subset=["player.id"]).copy()
        baseline_today = played_today[["player.id", "last_game_points"]].drop_duplicates("player.id")

        # Merge on player id only to avoid mismatches from team code casing/changes
        merged = player_points_today.merge(actuals.drop(columns=["team.code"]), on=["player.id"], how="inner", suffixes=("_pred", "_act"))
        merged = merged.merge(baseline_today, on="player.id", how="left")
        # Ensure name columns exist for output (prefer actuals -> base -> preds)
        if "player.firstname" not in merged.columns:
            merged["player.firstname"] = merged.get("player.firstname_act", merged.get("player.firstname", merged.get("player.firstname_pred", np.nan)))
        if "player.lastname" not in merged.columns:
            merged["player.lastname"] = merged.get("player.lastname_act", merged.get("player.lastname", merged.get("player.lastname_pred", np.nan)))
        # Combined player name for easier reading
        try:
            merged["player_name"] = (
                merged["player.firstname"].fillna("").astype(str).str.strip() + " " +
                merged["player.lastname"].fillna("").astype(str).str.strip()
            ).str.strip()
        except Exception:
            merged["player_name"] = np.nan
        merged["date"] = pd.to_datetime(dt).date()
        # Prediction errors
        merged["pred_error"] = merged["pred_points"].astype(float) - merged["actual_points"].astype(float)
        if "pred_reb" in merged.columns and "actual_reb" in merged.columns:
            merged["reb_error"] = pd.to_numeric(merged["pred_reb"], errors="coerce") - pd.to_numeric(merged["actual_reb"], errors="coerce")
        if "pred_ast" in merged.columns and "actual_ast" in merged.columns:
            merged["ast_error"] = pd.to_numeric(merged["pred_ast"], errors="coerce") - pd.to_numeric(merged["actual_ast"], errors="coerce")
        merged["baseline_error"] = merged["last_game_points"].astype(float) - merged["actual_points"].astype(float)
        if merged.empty:
            empty_dates_info.append((d, len(player_points_today), len(actuals), 0, 'no_overlap'))
            continue

        cols_out = [
            "date", "team.code", "player.id", "player.firstname", "player.lastname", "player_name",
            "pred_points", "actual_points", "last_game_points", "pred_error", "baseline_error"
        ]
        if "reb_error" in merged.columns:
            cols_out += ["pred_reb", "actual_reb", "reb_error"]
        if "ast_error" in merged.columns:
            cols_out += ["pred_ast", "actual_ast", "ast_error"]
        results.append(merged[cols_out])

    if not results:
        # Print small diagnostic to help identify why no rows
        if empty_dates_info:
            try:
                diag = pd.DataFrame(empty_dates_info, columns=["date", "n_pred", "n_act", "n_merge", "reason"])
                print("Backtest diagnostics (first 10 rows):")
                print(diag.head(10).to_string(index=False))
            except Exception:
                pass
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest", action="store_true", help="Run backtest; if no dates provided, uses last 5 PST game days")
    parser.add_argument("--backtest-start", dest="backtest_start", default=None)
    parser.add_argument("--backtest-end", dest="backtest_end", default=None)
    parser.add_argument("--auto-tune", action="store_true", help="Grid-search hyperparameters on the backtest range")
    parser.add_argument("--backtest-today-only", action="store_true", help="Restrict backtest to players who play today (PST)")
    try:
        args = parser.parse_args()
    except SystemExit:
        args = parser.parse_args([])

    ratings, games_filtered, games_all, player_stats = read_inputs()
    ratings_clean = clean_ratings(ratings)

    # Backtest mode (optional) – CLI args or environment variables
    env_bt_start = os.environ.get("NBA_BT_START")
    env_bt_end = os.environ.get("NBA_BT_END")
    bt_start = args.backtest_start or env_bt_start
    bt_end = args.backtest_end or env_bt_end

    # If in-code toggle or --backtest without dates, use last N games per team (union of PST dates)
    auto_dates = None
    # Today-only backtest takes priority
    if args.backtest_today_only:
        now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
        auto_dates = [now_pst.date()]
        print(f"Backtesting for today only (PST): {auto_dates[0]}")
    elif BACKTEST_TOGGLE or (args.backtest and not bt_start and not bt_end):
        gtmp = games_all.copy()
        gtmp["game_date_utc"] = pd.to_datetime(gtmp["game_date"], utc=True, errors="coerce")
        gtmp["game_date_pst"] = gtmp["game_date_utc"].dt.tz_convert("America/Los_Angeles")
        gtmp["pst_date"] = gtmp["game_date_pst"].dt.date
        # Only use completed games (actuals available) for per-team last-N selection
        gtmp["home_score"] = pd.to_numeric(gtmp.get("home_score"), errors="coerce")
        gtmp["visitor_score"] = pd.to_numeric(gtmp.get("visitor_score"), errors="coerce")
        gdone = gtmp.dropna(subset=["home_score", "visitor_score"]) if "home_score" in gtmp.columns and "visitor_score" in gtmp.columns else gtmp
        auto_set = set()
        for side in ["home_code", "visitor_code"]:
            tdf = gdone[[side, "pst_date"]].dropna()
            tdf = tdf.rename(columns={side: "team"}).sort_values(["team", "pst_date"])
            lastn = tdf.groupby("team").tail(BACKTEST_LAST_N_GAME_DAYS)
            auto_set.update(lastn["pst_date"].tolist())
        auto_dates = sorted(auto_set)
        # Also intersect with player_stats dates to ensure actuals for players exist
        ps_dates = None
        try:
            tmp_ps = enrich_player_stats_with_game_dates(player_stats, games_all)
            tmp_ps["pst_date"] = tmp_ps["game_date_pst"].dt.date
            ps_dates = set(tmp_ps["pst_date"].dropna().unique().tolist())
        except Exception:
            ps_dates = None
        if ps_dates:
            auto_dates = [d for d in auto_dates if d in ps_dates]
        if auto_dates:
            print(f"Backtesting last {BACKTEST_LAST_N_GAME_DAYS} games per team across {len(auto_dates)} dates")
    # If still no dates or explicit start, default backtest start to season start (Oct 23 PST)
    if (args.backtest or BACKTEST_TOGGLE) and not bt_start and not auto_dates and not args.backtest_today_only:
        now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
        season_start = _season_start_for_date(now_pst)
        bt_start = str(season_start.date())
    best_hparams = DEFAULT_HPARAMS.copy()
    if (bt_start or bt_end or auto_dates) and args.auto_tune:
        # Simple grid search for alpha, k, sigma, and clips
        grid_alpha = [0.0]  # alpha not used in season-only PPM
        grid_k = [2.0, 3.5, 5.0, 8.0]
        grid_sigma = [0.15, 0.2, 0.25]
        grid_clip = [(0.95, 1.05), (0.92, 1.08), (0.90, 1.12)]
        grid_ml_alpha = [0.3, 0.5, 0.7, 0.85]
        grid_ridge = [0.1, 0.5, 1.0]
        # Role-based grids (modest ranges)
        grid_role_min = [(28.0, 18.0)]
        grid_min_scale = [(1.06, 1.02, 0.98), (1.04, 1.02, 0.99)]
        grid_ppm_scale = [(1.04, 1.02, 0.99), (1.03, 1.01, 0.99)]
        best_mae = float("inf")
        for a in grid_alpha:
            for k in grid_k:
                for sg in grid_sigma:
                    for clip in grid_clip:
                        for mla in grid_ml_alpha:
                            for rl in grid_ridge:
                                for (start_min, rot_min) in grid_role_min:
                                    for (s_ms, r_ms, b_ms) in grid_min_scale:
                                        for (s_ps, r_ps, b_ps) in grid_ppm_scale:
                                            hps = {
                                                "alpha": a,
                                                "k": k,
                                                "sigma": sg,
                                                "pace_clip": clip,
                                                "opp_def_clip": clip,
                                                "ml_alpha": mla,
                                                "ridge_lambda": rl,
                                                "role_start_min": start_min,
                                                "role_rot_min": rot_min,
                                                "starter_min_scale": s_ms,
                                                "rotation_min_scale": r_ms,
                                                "bench_min_scale": b_ms,
                                                "starter_ppm_scale": s_ps,
                                                "rotation_ppm_scale": r_ps,
                                                "bench_ppm_scale": b_ps,
                                            }
                                            bt_df_try = backtest_player_points(
                                                games_all, player_stats, ratings_clean,
                                                start_date=bt_start, end_date=bt_end, sims=350, dates_list=auto_dates, hparams=hps
                                            )
                                            if bt_df_try is None or bt_df_try.empty:
                                                continue
                                            mae = bt_df_try["pred_error"].abs().mean()
                                            if mae < best_mae:
                                                best_mae = mae
                                                best_hparams = hps
        print(f"Auto-tune best MAE: {best_mae:.3f} with {best_hparams}")

    if bt_start or bt_end or auto_dates:
        bt_df = backtest_player_points(
            games_all, player_stats, ratings_clean,
            start_date=bt_start, end_date=bt_end, sims=500,
            dates_list=auto_dates, hparams=best_hparams
        )
        out_bt_path = os.path.join(DATA_DIR, "player_points_backtest.csv")
        bt_df.to_csv(out_bt_path, index=False)
        if not bt_df.empty:
            mae_pred = (bt_df["pred_error"].abs().mean())
            mae_base = (bt_df["baseline_error"].abs().mean())
            print(f"Backtest rows: {len(bt_df)} | MAE pred: {mae_pred:.2f} | MAE last-game baseline: {mae_base:.2f}")
            print(f"Saved backtest to: {out_bt_path}")
        else:
            print("Backtest produced no rows — check date range.")

    # Daily prediction
    now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
    ps = enrich_player_stats_with_game_dates(player_stats, games_all)
    todays_games = get_todays_games_pst(games_all, now_pst)
    if todays_games.empty:
        print("No games found for today (PST). Skipping daily predictions.")
        return

    games_pred, player_points_today = simulate_games_from_ratings(todays_games, ps, ratings_clean, sims=1000)

    out_games_path = os.path.join(DATA_DIR, "predicted_game_scores_today.csv")
    out_players_path = os.path.join(DATA_DIR, "predicted_player_points_today.csv")

    try:
        hist_l5 = games_filtered.copy()
        hist_l5["home_score"] = pd.to_numeric(hist_l5["home_score"], errors="coerce")
        hist_l5["visitor_score"] = pd.to_numeric(hist_l5["visitor_score"], errors="coerce")
        hist_l5["date"] = pd.to_datetime(hist_l5["game_date_pacific"], errors="coerce")
        long_rows = []
        for _, r in hist_l5.iterrows():
            if pd.isna(r.get("home_score")) or pd.isna(r.get("visitor_score")):
                continue
            long_rows.append({"team": r["home_code"], "points": r["home_score"], "date": r["date"]})
            long_rows.append({"team": r["visitor_code"], "points": r["visitor_score"], "date": r["date"]})
        ldf = pd.DataFrame(long_rows)
        if not ldf.empty:
            ldf = ldf.sort_values(["team", "date"]).assign(avg_pts_last5=ldf.groupby("team")["points"].apply(lambda s: s.shift(1).rolling(5, min_periods=1).mean()).values)
            latest_l5 = ldf.groupby("team").tail(1)[["team", "avg_pts_last5"]]
            games_pred = games_pred.merge(latest_l5.rename(columns={"team": "home"}), on="home", how="left").rename(columns={"avg_pts_last5": "home_avg_pts_last5"})
            games_pred = games_pred.merge(latest_l5.rename(columns={"team": "away"}), on="away", how="left").rename(columns={"avg_pts_last5": "away_avg_pts_last5"})
    except Exception:
        pass

    cols = [
        "game.id", "game_date_pst", "home", "away", "home_pred_points", "away_pred_points",
        "proj_spread_home_minus_away", "model_total", "home_p05", "home_p50", "home_p95",
        "away_p05", "away_p50", "away_p95", "home_avg_pts_last5", "away_avg_pts_last5"
    ]
    # Global calibration: actual ≈ a + b * pred (fit on recent backtests)
    try:
        player_points_today['pred_points'] = pd.to_numeric(player_points_today['pred_points'], errors='coerce')
        if 'pred_points_raw' not in player_points_today.columns:
            player_points_today['pred_points_raw'] = player_points_today['pred_points']
        player_points_today['pred_points_calibrated'] = player_points_today['pred_points']
        if 'bt_df' in locals() and bt_df is not None and not bt_df.empty:
            cal = bt_df.copy()
            cal['pred_points'] = pd.to_numeric(cal.get('pred_points'), errors='coerce')
            cal['actual_points'] = pd.to_numeric(cal.get('actual_points'), errors='coerce')
            cal = cal.dropna(subset=['pred_points','actual_points'])
            if 'pst_date' in cal.columns:
                try:
                    cal['pst_date'] = pd.to_datetime(cal['pst_date'], errors='coerce')
                    last_dates = sorted([d for d in cal['pst_date'].dt.date.dropna().unique()])[-10:]
                    if last_dates:
                        cal = cal[cal['pst_date'].dt.date.isin(last_dates)]
                except Exception:
                    pass
            if len(cal) >= 30 and cal['pred_points'].std(skipna=True) > 1e-6:
                try:
                    b, a = np.polyfit(cal['pred_points'].astype(float), cal['actual_points'].astype(float), 1)
                    b = float(np.clip(b, 0.6, 1.4))
                    a = float(np.clip(a, -6.0, 6.0))
                    player_points_today['pred_points_calibrated'] = a + b * player_points_today['pred_points_raw']
                except Exception:
                    pass
    except Exception:
        pass

    # Apply per-player bias (intercept) from recent backtest errors if available
    try:
        if 'bt_df' in locals() and bt_df is not None and not bt_df.empty and 'player.id' in bt_df.columns:
            tmp = bt_df.copy()
            tmp['pred_error'] = pd.to_numeric(tmp.get('pred_error'), errors='coerce')
            tmp['player.id'] = pd.to_numeric(tmp['player.id'], errors='coerce')
            tmp = tmp.dropna(subset=['player.id', 'pred_error'])
            if 'pst_date' in tmp.columns:
                try:
                    tmp['pst_date'] = pd.to_datetime(tmp['pst_date'], errors='coerce')
                except Exception:
                    pass
                tmp = tmp.sort_values(['player.id', 'pst_date'])
            else:
                tmp = tmp.sort_values(['player.id'])

            K = 10; halflife = 5.0; c = 10.0; bmax = 3.0

            def _recent_bias(g: pd.DataFrame) -> float:
                g = g.tail(K)
                n = len(g)
                if n == 0:
                    return 0.0
                w = np.array([0.5 ** ((n - 1 - i) / max(halflife, 1e-6)) for i in range(n)], dtype=float)
                num = float(np.nansum(w * g['pred_error'].astype(float).values))
                den = float(np.nansum(w)) if np.nansum(w) > 0 else 1.0
                mean_err = num / den
                shrink = n / (n + c)
                return float(np.clip(shrink * mean_err, -bmax, bmax))

            bias_series = tmp.groupby('player.id').apply(_recent_bias).rename('bias')
            bias_df = bias_series.reset_index()

            player_points_today['player.id'] = pd.to_numeric(player_points_today['player.id'], errors='coerce')
            # Start from calibrated predictions if available
            base_col = 'pred_points_calibrated' if 'pred_points_calibrated' in player_points_today.columns else 'pred_points'
            player_points_today[base_col] = pd.to_numeric(player_points_today[base_col], errors='coerce')
            player_points_today = player_points_today.merge(bias_df, on='player.id', how='left')
            player_points_today['bias'] = player_points_today['bias'].fillna(0.0)
            player_points_today['pred_points'] = (player_points_today[base_col] + player_points_today['bias']).astype(float)
        else:
            player_points_today['bias'] = 0.0
            if 'pred_points_raw' not in player_points_today.columns:
                player_points_today['pred_points_raw'] = player_points_today['pred_points']
            if 'pred_points_calibrated' not in player_points_today.columns:
                player_points_today['pred_points_calibrated'] = player_points_today['pred_points']
    except Exception:
        # Fail-safe: continue without biasing
        player_points_today['bias'] = 0.0
        if 'pred_points_raw' not in player_points_today.columns:
            player_points_today['pred_points_raw'] = player_points_today['pred_points']
        if 'pred_points_calibrated' not in player_points_today.columns:
            player_points_today['pred_points_calibrated'] = player_points_today['pred_points']

    games_pred[cols].to_csv(out_games_path, index=False)
    player_points_today.sort_values(["team.code", "pred_points"], ascending=[True, False]).to_csv(out_players_path, index=False)
    print(f"Saved game predictions to: {out_games_path}")
    print(f"Saved player predictions to: {out_players_path}")

    # Create today's player summary with cumulative pred_error if backtest ran
    try:
        todays_df = player_points_today.copy()
        todays_df["player_name"] = (
            todays_df["player.firstname"].fillna("").astype(str).str.strip() + " " +
            todays_df["player.lastname"].fillna("").astype(str).str.strip()
        ).str.strip()
        summary_cols = ["player_name", "team.code", "pred_points", "player.id", "bias", "pred_points_raw", "pred_points_calibrated"]
        summary_df = todays_df[summary_cols].rename(columns={"team.code": "team"})

        # Attach sum of pred_error from backtest if available
        if 'bt_df' in locals() and bt_df is not None and not bt_df.empty:
            err_agg = (
                bt_df.assign(abs_pred_error=bt_df["pred_error"].abs())
                  .groupby("player.id")
                  .agg(sum_abs_pred_error=("abs_pred_error", "sum"),
                       games=("abs_pred_error", "size"))
                  .reset_index()
            )
            err_agg["avg_abs_pred_error"] = err_agg["sum_abs_pred_error"] / err_agg["games"].replace(0, pd.NA)
            summary_df = summary_df.merge(err_agg, on="player.id", how="left")
            # Optional: rebounds/assists error aggregates if present in backtest
            if "reb_error" in bt_df.columns:
                reb_agg = (
                    bt_df.assign(abs_reb_error=bt_df["reb_error"].abs())
                      .groupby("player.id")
                      .agg(sum_abs_reb_error=("abs_reb_error", "sum"),
                           games_reb=("abs_reb_error", "size"))
                      .reset_index()
                )
                reb_agg["avg_abs_reb_error"] = reb_agg["sum_abs_reb_error"] / reb_agg["games_reb"].replace(0, pd.NA)
                summary_df = summary_df.merge(reb_agg, on="player.id", how="left")
            if "ast_error" in bt_df.columns:
                ast_agg = (
                    bt_df.assign(abs_ast_error=bt_df["ast_error"].abs())
                      .groupby("player.id")
                      .agg(sum_abs_ast_error=("abs_ast_error", "sum"),
                           games_ast=("abs_ast_error", "size"))
                      .reset_index()
                )
                ast_agg["avg_abs_ast_error"] = ast_agg["sum_abs_ast_error"] / ast_agg["games_ast"].replace(0, pd.NA)
                summary_df = summary_df.merge(ast_agg, on="player.id", how="left")
        else:
            summary_df["sum_abs_pred_error"] = np.nan
            summary_df["games"] = np.nan
            summary_df["avg_abs_pred_error"] = np.nan
            summary_df["sum_abs_reb_error"] = np.nan
            summary_df["games_reb"] = np.nan
            summary_df["avg_abs_reb_error"] = np.nan
            summary_df["sum_abs_ast_error"] = np.nan
            summary_df["games_ast"] = np.nan
            summary_df["avg_abs_ast_error"] = np.nan

        # Filter: require at least 3 backtest games
        try:
            summary_df = summary_df[summary_df["games"].astype(float) >= 3]
        except Exception:
            pass
        # Add relative error = avg_abs_pred_error / pred_points and sort ascending
        summary_df["pred_points"] = pd.to_numeric(summary_df["pred_points"], errors="coerce")
        summary_df["avg_abs_pred_error"] = pd.to_numeric(summary_df["avg_abs_pred_error"], errors="coerce")
        denom = summary_df["pred_points"].replace(0, np.nan)
        summary_df["rel_abs_error"] = (summary_df["avg_abs_pred_error"] / denom).replace([np.inf, -np.inf], np.nan)
        # Try to attach rebounds/assists predictions
        if "pred_reb" in todays_df.columns:
            summary_df = summary_df.merge(todays_df[["player.id", "pred_reb", "pred_ast"]], on="player.id", how="left")
        # Choose output columns including optional reb/ast backtest metrics
        base_cols = ["player_name", "team", "pred_points", "sum_abs_pred_error", "games", "avg_abs_pred_error", "rel_abs_error", "bias", "pred_points_raw", "pred_points_calibrated"]
        if "pred_reb" in todays_df.columns:
            base_cols += ["pred_reb", "pred_ast"]
        extra_bt_cols = ["sum_abs_reb_error", "games_reb", "avg_abs_reb_error", "sum_abs_ast_error", "games_ast", "avg_abs_ast_error"]
        keep_cols = [c for c in base_cols + extra_bt_cols if c in summary_df.columns]
        summary_df = summary_df[keep_cols]
        summary_df = summary_df.sort_values(["rel_abs_error", "avg_abs_pred_error"], ascending=[True, True], na_position='last')
        out_summary_path = os.path.join(DATA_DIR, "players_today_summary.csv")
        summary_df.to_csv(out_summary_path, index=False)
        print(f"Saved today summary to: {out_summary_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
