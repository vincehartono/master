import os
import math
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

# Reuse helpers from pred_scores where possible (same folder)
from pred_scores import (
    read_inputs,
    clean_ratings,
    enrich_player_stats_with_game_dates,
    get_todays_games_pst,
    build_today_rosters,
    minutes_from_recent_and_ratings,
    _season_start_for_date,
)


DATA_DIR = os.path.dirname(__file__)

# Load input overrides from model_inputs.txt (key=value)
def _load_input_overrides():
    cfg = {}
    cfg_path = os.path.join(DATA_DIR, "model_inputs.txt")
    if not os.path.exists(cfg_path):
        return cfg
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    cfg[k.strip()] = v.strip()
    except Exception:
        pass
    return cfg

_INPUT_OVERRIDES = _load_input_overrides()

# Backtest cache directory (per-day files)
BACKTEST_CACHE_DIR = _INPUT_OVERRIDES.get("BACKTEST_CACHE_DIR", os.path.join(DATA_DIR, "Backtest_cache"))
os.makedirs(BACKTEST_CACHE_DIR, exist_ok=True)
USAGE_CAPS_PATH = _INPUT_OVERRIDES.get("USAGE_CAPS_FILE", os.path.join(DATA_DIR, "usage_caps.csv"))

_USAGE_CAPS_CACHE = None

def load_usage_caps(path: str = None):
    global _USAGE_CAPS_CACHE
    if path is None:
        path = USAGE_CAPS_PATH
    if _USAGE_CAPS_CACHE is not None:
        return _USAGE_CAPS_CACHE
    if not os.path.exists(path):
        _USAGE_CAPS_CACHE = {}
        return _USAGE_CAPS_CACHE
    try:
        df = pd.read_csv(path)
        caps = {}
        for _, r in df.iterrows():
            try:
                rank = int(r.get("rank"))
                cap = float(r.get("cap"))
                if rank >= 1 and cap > 0:
                    caps[rank] = max(0.01, min(0.5, cap))
            except Exception:
                continue
        _USAGE_CAPS_CACHE = caps
        return caps
    except Exception:
        _USAGE_CAPS_CACHE = {}
        return _USAGE_CAPS_CACHE

# Default output folder for granular PBP
PBP_DIR = _INPUT_OVERRIDES.get("PBP_DIR", os.path.join(DATA_DIR, "Play_by_Play"))


def _sigmoid(x: float, k: float = 1.0):
    return 1.0 / (1.0 + math.exp(-k * x))


def _fmt_clock(seconds_left: float) -> str:
    s = max(0, int(round(seconds_left)))
    m, s = divmod(s, 60)
    return f"{m:02d}:{s:02d}"


def _ratings_aggregate(team_players: pd.DataFrame) -> dict:
    # Weighted by predicted minutes; fallback equal weights
    df = team_players.copy()
    if "pred_minutes" in df.columns:
        w = pd.to_numeric(df["pred_minutes"], errors="coerce").fillna(0.0)
        if w.sum() <= 0:
            w = pd.Series(np.ones(len(df)), index=df.index)
    else:
        w = pd.Series(np.ones(len(df)), index=df.index)

    def wz(col):
        return (pd.to_numeric(df.get(col), errors="coerce").fillna(0.0) * w).sum() / (w.sum() if w.sum() else 1.0)

    # Offense components
    off_3 = wz("Three-Point Shot_z")
    off_mid = wz("Mid-Range Shot_z")
    off_rim = wz("Close Shot_z")
    ft = wz("Free Throw_z")
    play = wz("Playmaking_z")
    handle = wz("Ball Handle_z")
    speed = wz("Speed_z")
    iq = wz("Shot IQ_z")

    # Defense components
    per_d = wz("Perimeter Defense_z")
    int_d = wz("Interior Defense_z")
    dcons = wz("Defensive Consistency_z")
    block = wz("Block_z")
    steal = wz("Steal_z")
    d_speed = wz("Lateral Quickness_z") if "Lateral Quickness_z" in df.columns else wz("Speed_z")

    # Rebounding proxies
    oreb = wz("Offensive Rebound_z") if "Offensive Rebound_z" in df.columns else wz("Offensive Rebound IQ_z") if "Offensive Rebound IQ_z" in df.columns else 0.0
    dreb = wz("Defensive Rebound_z") if "Defensive Rebound_z" in df.columns else wz("Defensive Rebound IQ_z") if "Defensive Rebound IQ_z" in df.columns else 0.0

    return {
        "off_3": off_3,
        "off_mid": off_mid,
        "off_rim": off_rim,
        "ft": ft,
        "play": play,
        "handle": handle,
        "speed": speed,
        "iq": iq,
        "per_d": per_d,
        "int_d": int_d,
        "dcons": dcons,
        "block": block,
        "steal": steal,
        "d_speed": d_speed,
        "oreb": oreb,
        "dreb": dreb,
    }


def _team_shot_profile(off: dict, deff: dict, rng=None) -> dict:
    if rng is None:
        rng = np.random.default_rng()
    # Shot mix influenced by offense strengths vs opponent defense
    rim_bias = _sigmoid(off["off_rim"] - deff["int_d"], 0.9)
    mid_bias = _sigmoid(off["off_mid"] - 0.5 * (deff["int_d"] + deff["per_d"]), 0.7)
    three_bias = _sigmoid(off["off_3"] - deff["per_d"], 0.9)
    base = np.array([rim_bias, mid_bias, three_bias], dtype=float)
    # Add small randomness to shot mix before normalization
    base = base * rng.uniform(0.95, 1.05, size=3)
    base = np.maximum(base, 1e-3)
    mix = base / base.sum()

    # Make/miss probabilities baseline adjusted by defense
    p2_rim = 0.62 + 0.05 * (off["off_rim"] - deff["int_d"]) + 0.01 * off["iq"]
    p2_mid = 0.42 + 0.04 * (off["off_mid"] - 0.5 * (deff["int_d"] + deff["per_d"])) + 0.005 * off["iq"]
    p3 = 0.35 + 0.05 * (off["off_3"] - deff["per_d"]) + 0.005 * off["iq"]
    # Jitter each probability slightly and clip
    p2_rim = float(np.clip(p2_rim + rng.uniform(-0.02, 0.02), 0.45, 0.80))
    p2_mid = float(np.clip(p2_mid + rng.uniform(-0.02, 0.02), 0.30, 0.60))
    p3 = float(np.clip(p3 + rng.uniform(-0.02, 0.02), 0.25, 0.45))

    # Turnover and foul tendencies
    tov = 0.12 - 0.025 * (off["handle"] + off["play"]) + 0.02 * deff["steal"]
    tov = float(np.clip(tov + rng.uniform(-0.01, 0.01), 0.07, 0.18))

    draw_foul = 0.12 + 0.015 * off["iq"] - 0.012 * deff["dcons"]
    draw_foul = float(np.clip(draw_foul + rng.uniform(-0.01, 0.01), 0.06, 0.20))

    # Rebounding chances for second chances
    oreb = 0.26 + 0.05 * (off["oreb"] - deff["dreb"]) - 0.01 * deff["block"]
    oreb = float(np.clip(oreb + rng.uniform(-0.02, 0.02), 0.15, 0.35))

    ft_pct = 0.76 + 0.04 * off["ft"]
    ft_pct = float(np.clip(ft_pct + rng.uniform(-0.02, 0.02), 0.65, 0.90))

    return {
        "mix": mix,  # [rim, mid, three]
        "p2_rim": p2_rim,
        "p2_mid": p2_mid,
        "p3": p3,
        "tov": tov,
        "draw_foul": draw_foul,
        "oreb": oreb,
        "ft_pct": ft_pct,
    }


def _pick_player_for_event(team_df: pd.DataFrame, kind: str, rng: np.random.Generator,
                           shots_taken=None, shot_targets=None, shot_caps=None,
                           exclude_player: str = None) -> str:
    # Choose a shooter or turnover committer based on relevant ratings and minutes
    df = team_df.copy()
    pm = pd.to_numeric(df.get("pred_minutes"), errors="coerce").fillna(0.0)
    if kind == "three":
        score = pd.to_numeric(df.get("Three-Point Shot_z"), errors="coerce").fillna(0.0)
    elif kind == "mid":
        score = pd.to_numeric(df.get("Mid-Range Shot_z"), errors="coerce").fillna(0.0)
    elif kind == "rim":
        score = pd.to_numeric(df.get("Close Shot_z"), errors="coerce").fillna(0.0)
    elif kind == "oreb":
        # favor offensive rebounding
        if "Offensive Rebound_z" in df.columns:
            score = pd.to_numeric(df.get("Offensive Rebound_z"), errors="coerce").fillna(0.0)
        else:
            score = pd.to_numeric(df.get("Strength_z", 0), errors="coerce").fillna(0.0)
    elif kind == "dreb":
        if "Defensive Rebound_z" in df.columns:
            score = pd.to_numeric(df.get("Defensive Rebound_z"), errors="coerce").fillna(0.0)
        else:
            score = pd.to_numeric(df.get("Defensive Consistency_z", 0), errors="coerce").fillna(0.0)
    elif kind == "tov":
        score = -pd.to_numeric(df.get("Ball Handle_z"), errors="coerce").fillna(0.0)
    else:
        score = pd.Series(np.zeros(len(df)), index=df.index)
    w = (pm.clip(lower=1.0) * (1.0 + 0.5 * score.clip(lower=-2, upper=2))).to_numpy()
    # Usage throttle: downweight players exceeding target shots; also cap by usage share
    if shots_taken is not None and shot_targets is not None:
        factors = []
        team_total = float(sum(shots_taken.values())) if isinstance(shots_taken, dict) else 0.0
        for i in range(len(df)):
            first = str(df.iloc[i].get("player.firstname", "")).strip()
            last = str(df.iloc[i].get("player.lastname", "")).strip()
            name = (first + " " + last).strip()
            taken = shots_taken.get(name, 0)
            target = max(1e-6, shot_targets.get(name, 0))
            over = max(0.0, (taken / target) - 1.0)
            # After exceeding target, decay weight quickly
            f = 1.0 / (1.0 + 2.0 * over)
            # Cap by usage share if provided
            if shot_caps is not None and team_total > 0:
                cap = max(0.05, float(shot_caps.get(name, 1.0)))
                share = taken / team_total if team_total > 0 else 0.0
                over_share = max(0.0, (share / cap) - 1.0)
                if over_share > 0:
                    f *= 1.0 / (1.0 + 8.0 * over_share)
            factors.append(f)
        w = w * np.array(factors)
    w = np.maximum(w, 1e-6)
    # exclude a specific player (e.g., shooter when picking assister)
    if exclude_player is not None:
        mask = []
        for i in range(len(df)):
            nm = (str(df.iloc[i].get("player.firstname", "")).strip() + " " + str(df.iloc[i].get("player.lastname", "")).strip()).strip()
            mask.append(nm != exclude_player)
        mask = np.array(mask, dtype=bool)
        if mask.any():
            w = np.where(mask, w, 0.0)
    if w.sum() <= 0:
        w = np.ones(len(df))
    idx = rng.choice(len(df), p=w / w.sum())
    first = str(df.iloc[idx].get("player.firstname", "")).strip()
    last = str(df.iloc[idx].get("player.lastname", "")).strip()
    name = (first + " " + last).strip()
    return name or "Unknown"


def _quarter_lineups(team_df: pd.DataFrame) -> dict:
    # Build simple quarter lineups: pick top 5 by predicted minutes each quarter
    df = team_df.copy()
    df["pred_minutes"] = pd.to_numeric(df.get("pred_minutes"), errors="coerce").fillna(0.0)
    df = df.sort_values("pred_minutes", ascending=False).reset_index(drop=True)
    starters = df.head(5).copy()
    bench = df.iloc[5:].copy()
    # Weight per quarter: starters play a bit more in Q1/Q4
    weights = np.array([0.27, 0.23, 0.23, 0.27])
    target_q_min = (df["pred_minutes"].sum() * weights) / 5.0  # average per-player on court per quarter
    # For simplicity, keep starters Q1 and Q4, mix in two bench in Q2/Q3
    q_lineups = {
        1: starters,
        2: pd.concat([starters.head(3), bench.head(2)]).reset_index(drop=True),
        3: pd.concat([starters.head(3), bench.tail(2) if len(bench) >= 2 else bench.head(min(2, len(bench))) ]).reset_index(drop=True),
        4: starters,
    }
    # Ensure exactly 5
    for k in [1,2,3,4]:
        q_lineups[k] = q_lineups[k].head(5)
        if len(q_lineups[k]) < 5:
            # fill from starters
            fill = starters.head(5 - len(q_lineups[k]))
            q_lineups[k] = pd.concat([q_lineups[k], fill]).head(5)
    return q_lineups


def simulate_game(home_team, away_team, team_frames, pace_total, rng, team_calib=None, make_nerf=0.10):
    # Setup team profiles
    home_df = team_frames[home_team]
    away_df = team_frames[away_team]
    home_agg = _ratings_aggregate(home_df)
    away_agg = _ratings_aggregate(away_df)
    home_prof = _team_shot_profile(home_agg, away_agg, rng)
    away_prof = _team_shot_profile(away_agg, home_agg, rng)

    # Estimate possessions from total and efficiency
    # Start with baseline points per possession ~ 1.10 and adjust by offense-defense balance
    ppp_home = 1.10 + 0.05 * (home_agg["iq"] + 0.5 * (home_agg["off_rim"] + home_agg["off_3"]) - 0.5 * (away_agg["per_d"] + away_agg["int_d"]))
    ppp_away = 1.10 + 0.05 * (away_agg["iq"] + 0.5 * (away_agg["off_rim"] + away_agg["off_3"]) - 0.5 * (home_agg["per_d"] + home_agg["int_d"]))
    # Apply optional team calibration (offense and defense scalers)
    if team_calib:
        hc = team_calib.get(home_team, {})
        ac = team_calib.get(away_team, {})
        off_scale_home = float(hc.get("off_ppp_scale", 1.0))
        def_scale_home = float(hc.get("def_ppp_scale", 1.0))
        off_scale_away = float(ac.get("off_ppp_scale", 1.0))
        def_scale_away = float(ac.get("def_ppp_scale", 1.0))
        ppp_home *= off_scale_home * (1.0 / max(1e-6, def_scale_away))
        ppp_away *= off_scale_away * (1.0 / max(1e-6, def_scale_home))
    ppp_home = float(np.clip(ppp_home, 0.98, 1.25))
    ppp_away = float(np.clip(ppp_away, 0.98, 1.25))
    poss_total = max(160, int(round(2 * pace_total / ((ppp_home + ppp_away) / 2.0))))

    # 48 minutes, start with away possession
    clock = 48 * 60
    quarter_end = [36 * 60, 24 * 60, 12 * 60, 0]
    q = 1
    score_h = 0
    score_a = 0
    events = []
    offense = "away"

    # Quarter-based lineups
    home_q_lineups = _quarter_lineups(home_df)
    away_q_lineups = _quarter_lineups(away_df)

    # Foul tracking and simple foul-trouble benching per quarter
    foul_counts = {"home": {}, "away": {}}
    foul_benched = {"home": set(), "away": set()}
    foul_dq = {"home": set(), "away": set()}
    team_fouls_q = {"home": 0, "away": 0}

    # Fatigue and hot streak tracking
    cont_secs = {"home": {}, "away": {}}  # continuous seconds on floor
    hot_makes = {"home": {}, "away": {}}  # consecutive makes

    def foul_threshold(cur_q: int) -> int:
        qn = min(max(int(cur_q), 1), 4)
        return {1: 2, 2: 3, 3: 4, 4: 5}.get(qn, 5)

    def name_from_row(row) -> str:
        return (str(row.get("player.firstname", "")).strip() + " " + str(row.get("player.lastname", "")).strip()).strip()

    def _role_from_row(row) -> str:
        pos = str(row.get("pos", "")).upper()
        if "C" in pos:
            return "big"
        if "F" in pos and "G" not in pos:
            return "wing"
        if "G" in pos:
            return "guard"
        return "wing"

    def lineup_with_foul_sub(side: str, cur_q: int):
        base = (home_q_lineups if side == "home" else away_q_lineups).get(min(cur_q, 4))
        base = base.copy().reset_index(drop=True)
        benched = foul_benched[side] | foul_dq[side]
        # If nobody is benched, return base
        if not benched:
            return base
        # Build candidate pool (full team df sorted by predicted minutes desc)
        pool = (home_df if side == "home" else away_df).copy()
        pool["pred_minutes"] = pd.to_numeric(pool.get("pred_minutes"), errors="coerce").fillna(0.0)
        pool = pool.sort_values("pred_minutes", ascending=False)
        on_floor = set(name_from_row(base.iloc[i]) for i in range(len(base)))
        # Replace any benched player with next best from pool
        for i in range(len(base)):
            nm = name_from_row(base.iloc[i])
            if nm in benched:
                # find first candidate not on floor and not benched
                wanted_role = _role_from_row(base.iloc[i])
                for _, prow in pool.iterrows():
                    cn = name_from_row(prow)
                    if cn not in on_floor and cn not in benched and _role_from_row(prow) == wanted_role:
                        base.iloc[i] = prow
                        on_floor.add(cn)
                        on_floor.discard(nm)
                        break
                else:
                    # fallback any role
                    for _, prow in pool.iterrows():
                        cn = name_from_row(prow)
                        if cn not in on_floor and cn not in benched:
                            base.iloc[i] = prow
                            on_floor.add(cn)
                            on_floor.discard(nm)
                            break
        return base

    def pick_defender_to_foul(def_df_local: pd.DataFrame) -> str:
        # Weight by minutes and defensive presence if available
        mins = pd.to_numeric(def_df_local.get("pred_minutes"), errors="coerce").fillna(0.0).to_numpy()
        per_d = pd.to_numeric(def_df_local.get("Perimeter Defense_z"), errors="coerce").fillna(0.0).to_numpy()
        int_d = pd.to_numeric(def_df_local.get("Interior Defense_z"), errors="coerce").fillna(0.0).to_numpy()
        w = np.maximum(mins * (1.0 + 0.1 * (np.maximum(per_d, int_d))), 1e-6)
        idx = int(rng.choice(len(def_df_local), p=(w / w.sum()))) if len(def_df_local) else 0
        return name_from_row(def_df_local.iloc[idx])

    # Build per-team usage targets (expected FGAs) to curb unrealistic single-player outputs
    def _usage_targets(team_df, prof, poss):
        df = team_df.copy()
        df["pred_minutes"] = pd.to_numeric(df.get("pred_minutes"), errors="coerce").fillna(0.0)
        # Scoring tendency proxy
        scr = (
            pd.to_numeric(df.get("Three-Point Shot_z"), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("Mid-Range Shot_z"), errors="coerce").fillna(0)
            + pd.to_numeric(df.get("Close Shot_z"), errors="coerce").fillna(0)
            + 0.5 * pd.to_numeric(df.get("Shot IQ_z"), errors="coerce").fillna(0)
        )
        w = (df["pred_minutes"].clip(lower=6.0) * (1.0 + 0.3 * scr.clip(-2, 2))).to_numpy()
        w = np.maximum(w, 1e-6)
        # Expected team FGAs ~ possessions minus turnovers plus a bit for OREB
        exp_fga = poss * (1.0 - prof["tov"])
        exp_fga *= 1.02  # smaller bump for second chances
        w = exp_fga * (w / w.sum())
        targets = {}
        caps = {}
        # Role-based usage caps from minutes rank
        order = np.argsort(-df["pred_minutes"].to_numpy())
        learned_caps = load_usage_caps()
        for rank, i in enumerate(order):
            name = (str(df.iloc[i].get("player.firstname", "")).strip() + " " + str(df.iloc[i].get("player.lastname", "")).strip()).strip()
            pm = float(df.iloc[i]["pred_minutes"]) if not pd.isna(df.iloc[i]["pred_minutes"]) else 0.0
            # Prefer learned cap by rank (1-based), fallback to minute-band defaults
            cap = learned_caps.get(rank + 1)
            if cap is None:
                if rank == 0:
                    cap = 0.18
                elif rank == 1:
                    cap = 0.16
                elif pm >= 24:
                    cap = 0.12
                elif pm >= 16:
                    cap = 0.08
                else:
                    cap = 0.05
            caps[name] = cap
        for i in range(len(df)):
            name = (str(df.iloc[i].get("player.firstname", "")).strip() + " " + str(df.iloc[i].get("player.lastname", "")).strip()).strip()
            targets[name] = float(w[i])
        return targets, caps

    home_targets, home_caps = _usage_targets(home_df, home_prof, poss_total / 2.0)
    away_targets, away_caps = _usage_targets(away_df, away_prof, poss_total / 2.0)
    home_shots = {}
    away_shots = {}

    # Box score tallies by player
    box_rows = {}

    def log(team: str, opponent: str, etype: str, player: str, pts: int, desc: str):
        nonlocal score_h, score_a, clock, q
        if team == "home":
            score_h += pts
        else:
            score_a += pts
        if player:
            key = (team, player)
            if key not in box_rows:
                box_rows[key] = {"team": home_team if team == "home" else away_team,
                                 "player": player, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
            if pts:
                box_rows[key]["points"] += pts
        events.append({
            "quarter": q,
            "game_clock": _fmt_clock(clock - quarter_end[q - 1] if q <= 4 else clock),
            "team": home_team if team == "home" else away_team,
            "opponent": away_team if team == "home" else home_team,
            "event_type": etype,
            "player": player,
            "points": pts,
            "score_home": score_h,
            "score_away": score_a,
            "description": desc,
        })

    while clock > 0 and len(events) < poss_total * 3:  # hard cap to avoid infinite loops
        # Update quarter
        if q <= 4 and clock <= quarter_end[q - 1]:
            q += 1
            # reset team fouls at new quarter
            if q <= 4:
                team_fouls_q = {"home": 0, "away": 0}
            # clear foul benched for new quarter (DQ remains)
            foul_benched = {"home": set(), "away": set()}
        # Possession length
        poss_len = float(np.clip(rng.normal(14.0, 4.0), 4.0, 24.0))
        clock -= poss_len

        # Choose current lineups based on quarter
        atk_df = lineup_with_foul_sub(offense, q)
        def_df = lineup_with_foul_sub("home" if offense == "away" else "away", q)
        prof = home_prof if offense == "home" else away_prof

        # Update fatigue meters for players currently on floor
        def _inc_cont(side, df):
            for i in range(len(df)):
                nm = name_from_row(df.iloc[i])
                cont_secs[side][nm] = cont_secs[side].get(nm, 0.0) + poss_len
        _inc_cont(offense, atk_df)
        _inc_cont("home" if offense == "away" else "away", def_df)

        # Apply small fatigue adjustment to make prob and turnover
        def _fatigue_penalty(df):
            vals = []
            for i in range(len(df)):
                nm = name_from_row(df.iloc[i])
                mins = cont_secs["home" if offense == "away" else "away"].get(nm, 0.0) / 60.0
                vals.append(max(0.0, mins - 8.0))
            avg_over = np.mean(vals) if vals else 0.0
            return float(np.clip(avg_over / 16.0 * 0.08, 0.0, 0.08))
        fatigue_pen = _fatigue_penalty(atk_df)

        # Turnover check
        tov_p = float(np.clip(prof["tov"] * (1.0 + fatigue_pen), 0.01, 0.5))
        if rng.random() < tov_p:
            pl = _pick_player_for_event(atk_df, "tov", rng,
                                        shots_taken=home_shots if offense == "home" else away_shots,
                                        shot_targets=home_targets if offense == "home" else away_targets,
                                        shot_caps=home_caps if offense == "home" else away_caps)
            log(offense, "home" if offense == "away" else "away", "Turnover", pl, 0, f"{pl} turns it over")
            offense = "home" if offense == "away" else "away"
            continue

        # Non-shooting foul: chance to occur before shot
        non_shoot_foul_p = float(np.clip(0.04 + 0.5 * prof["draw_foul"], 0.01, 0.20))
        if rng.random() < non_shoot_foul_p:
            defender = pick_defender_to_foul(def_df)
            def_side = "home" if offense == "away" else "away"
            foul_counts[def_side][defender] = foul_counts[def_side].get(defender, 0) + 1
            team_fouls_q[def_side] += 1
            log(offense, "home" if offense == "away" else "away", "Team Foul", defender, 0, f"Non-shooting foul on {defender}")
            # Credit personal foul to defender's box
            dteam = "home" if offense == "away" else "away"
            dkey = (dteam, defender)
            if dkey not in box_rows:
                box_rows[dkey] = {"team": home_team if dteam == "home" else away_team,
                                  "player": defender, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
            box_rows[dkey]["fouls"] += 1
            # DQ if 6 fouls
            if foul_counts[def_side][defender] >= 6:
                foul_dq[def_side].add(defender)
            # Bonus free throws on 5+ team fouls in quarter
            if team_fouls_q[def_side] >= 5:
                pl = _pick_player_for_event(atk_df, "rim", rng,
                                            shots_taken=home_shots if offense == "home" else away_shots,
                                            shot_targets=home_targets if offense == "home" else away_targets,
                                            shot_caps=home_caps if offense == "home" else away_caps)
                made = 0
                for i in range(2):
                    if rng.random() < prof["ft_pct"]:
                        made += 1
                if made > 0:
                    log(offense, "home" if offense == "away" else "away", "Bonus Free Throws", pl, made, f"{pl} makes {made}/2 FT (bonus)")
                else:
                    log(offense, "home" if offense == "away" else "away", "Bonus Free Throws", pl, 0, f"{pl} misses both FT (bonus)")
            # Offense retains possession
            continue

        # Shooting foul leading to free throws
        if rng.random() < prof["draw_foul"]:
            pl = _pick_player_for_event(atk_df, "rim", rng,
                                        shots_taken=home_shots if offense == "home" else away_shots,
                                        shot_targets=home_targets if offense == "home" else away_targets,
                                        shot_caps=home_caps if offense == "home" else away_caps)
            # Assign a defender to take the personal foul
            defender = pick_defender_to_foul(def_df)
            def_side = "home" if offense == "away" else "away"
            foul_counts[def_side][defender] = foul_counts[def_side].get(defender, 0) + 1
            log(offense, "home" if offense == "away" else "away", "Shooting Foul", defender, 0, f"Foul on {defender}")
            # Bench if in foul trouble for the current quarter
            if foul_counts[def_side][defender] >= foul_threshold(q):
                foul_benched[def_side].add(defender)
            if foul_counts[def_side][defender] >= 6:
                foul_dq[def_side].add(defender)
            # Credit personal foul to defender's box
            dteam = "home" if offense == "away" else "away"
            dkey = (dteam, defender)
            if dkey not in box_rows:
                box_rows[dkey] = {"team": home_team if dteam == "home" else away_team,
                                  "player": defender, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
            box_rows[dkey]["fouls"] += 1
            made = 0
            for i in range(2):
                if rng.random() < prof["ft_pct"]:
                    made += 1
            if made > 0:
                log(offense, "home" if offense == "away" else "away", "Free Throws", pl, made, f"{pl} makes {made}/2 FT")
            else:
                log(offense, "home" if offense == "away" else "away", "Free Throws", pl, 0, f"{pl} misses both FT")
            offense = "home" if offense == "away" else "away"
            continue

        # Field goal attempt: decide shot type
        mix = prof["mix"]
        typ_idx = rng.choice(3, p=mix)
        if typ_idx == 2:
            shot_kind = "three"; make_p = prof["p3"]; pts_val = 3
        elif typ_idx == 1:
            shot_kind = "mid"; make_p = prof["p2_mid"]; pts_val = 2
        else:
            shot_kind = "rim"; make_p = prof["p2_rim"]; pts_val = 2
        pl = _pick_player_for_event(atk_df, shot_kind, rng,
                                    shots_taken=home_shots if offense == "home" else away_shots,
                                    shot_targets=home_targets if offense == "home" else away_targets,
                                    shot_caps=home_caps if offense == "home" else away_caps)

        # Reduce make probabilities to avoid inflated scoring further and apply fatigue/hot streak
        make_p_adj = max(0.0, make_p - float(make_nerf))
        # Apply calibration to make probability: offense boosts, opponent defense suppresses
        if team_calib:
            if offense == "home":
                hc = team_calib.get(home_team, {})
                ac = team_calib.get(away_team, {})
            else:
                hc = team_calib.get(away_team, {})
                ac = team_calib.get(home_team, {})
            make_scale = float(hc.get("shot_make_scale", 1.0)) * (1.0 / max(1e-6, float(ac.get("opp_shot_make_scale", 1.0))))
            # gentle effect, slightly more suppressive baseline
            make_p_adj = float(np.clip(make_p_adj * (0.45 + 0.55 * make_scale), 0.05, 0.75))
        # Hot streak: small boost if shooter is hot
        hm = hot_makes[offense].get(pl, 0)
        if hm >= 3:
            make_p_adj = float(np.clip(make_p_adj * 1.05, 0.01, 0.9))
        # Fatigue penalty
        shooter_mins = cont_secs[offense].get(pl, 0.0) / 60.0
        if shooter_mins > 10.0:
            make_p_adj = float(np.clip(make_p_adj * 0.97, 0.01, 0.9))
        if rng.random() < make_p_adj:
            # Pick potential assister (no assist on FTs)
            # Assisted rate higher for rim and three, lower for mid
            assist_rate = 0.62 if shot_kind in ("rim", "three") else 0.45
            assist_rate = float(np.clip(assist_rate + 0.05 * (home_agg["play"] if offense == "home" else away_agg["play"]), 0.3, 0.8))
            assister = None
            if rng.random() < assist_rate:
                # Pick assister weighted by playmaking
                assister = _pick_player_for_event(atk_df.assign(**{"Mid-Range Shot_z": pd.to_numeric(atk_df.get("Playmaking_z"), errors="coerce").fillna(0.0)}),
                                                  "mid", rng, exclude_player=pl)
                akey_team = offense
                if assister:
                    key = (akey_team, assister)
                    if key not in box_rows:
                        box_rows[key] = {"team": home_team if akey_team == "home" else away_team,
                                         "player": assister, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
                    if "assists" not in box_rows[key]:
                        box_rows[key]["assists"] = 0
                    box_rows[key]["assists"] += 1

            desc = f"{pl} hits a {pts_val}-pt {shot_kind} shot" + (f" (assist {assister})" if assister else "")
            log(offense, "home" if offense == "away" else "away", "Made Shot", pl, pts_val, desc)
            hot_makes[offense][pl] = hot_makes[offense].get(pl, 0) + 1
            # Count a shot made toward usage
            if offense == "home":
                home_shots[pl] = home_shots.get(pl, 0) + 1
            else:
                away_shots[pl] = away_shots.get(pl, 0) + 1
            offense = "home" if offense == "away" else "away"
            continue
        else:
            # Missed shot, log it then rebound check
            log(offense, "home" if offense == "away" else "away", "Missed Shot", pl, 0, f"{pl} misses a {pts_val}-pt {shot_kind} shot")
            hot_makes[offense][pl] = 0
            if offense == "home":
                home_shots[pl] = home_shots.get(pl, 0) + 1
            else:
                away_shots[pl] = away_shots.get(pl, 0) + 1
            if rng.random() < prof["oreb"]:
                rb = _pick_player_for_event(atk_df, "oreb", rng,
                                            shots_taken=home_shots if offense == "home" else away_shots,
                                            shot_targets=home_targets if offense == "home" else away_targets,
                                            shot_caps=home_caps if offense == "home" else away_caps)
                log(offense, "home" if offense == "away" else "away", "Offensive Rebound", rb, 0, f"{rb} grabs offensive board")
                # Box tally
                key = (offense, rb)
                if key not in box_rows:
                    box_rows[key] = {"team": home_team if offense == "home" else away_team,
                                     "player": rb, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
                box_rows[key]["rebounds"] += 1
                # same offense keeps the ball; no switch
                continue
            else:
                rb = _pick_player_for_event(def_df, "dreb", rng,
                                            shots_taken=away_shots if offense == "home" else home_shots,
                                            shot_targets=away_targets if offense == "home" else home_targets,
                                            shot_caps=away_caps if offense == "home" else home_caps)
                log("home" if offense == "away" else "away", offense, "Defensive Rebound", rb, 0, f"{rb} secures the rebound")
                # Box tally for defense side
                dteam = "home" if offense == "away" else "away"
                key = (dteam, rb)
                if key not in box_rows:
                    box_rows[key] = {"team": home_team if dteam == "home" else away_team,
                                     "player": rb, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
                box_rows[key]["rebounds"] += 1
                offense = "home" if offense == "away" else "away"
                continue

    # At end of game, attribute simulated minutes from on-floor tracking
    for side, cmap in cont_secs.items():
        for nm, secs in cmap.items():
            key = (side, nm)
            if key not in box_rows:
                box_rows[key] = {"team": home_team if side == "home" else away_team,
                                 "player": nm, "points": 0, "rebounds": 0, "assists": 0, "fouls": 0, "minutes": 0.0}
            box_rows[key]["minutes"] += float(secs) / 60.0

    pbp_df = pd.DataFrame(events)
    box_df = pd.DataFrame(list(box_rows.values())) if box_rows else pd.DataFrame(columns=["team","player","points","rebounds","assists","fouls","minutes"])
    return pbp_df, box_df


def load_today_context():
    ratings, games_filtered, games_all, player_stats = read_inputs()
    ratings_clean = clean_ratings(ratings)
    ps = enrich_player_stats_with_game_dates(player_stats, games_all)

    today = datetime.now(ZoneInfo("America/Los_Angeles"))
    todays_games = get_todays_games_pst(games_all, today)
    if todays_games.empty:
        raise RuntimeError("No NBA games found for today (PST).")

    rosters = build_today_rosters(ps, todays_games, start_from_pst=today)
    minutes_today = minutes_from_recent_and_ratings(ps, rosters, ratings_clean)

    # Attach opponent codes
    t = todays_games[["home_code", "visitor_code"]].copy()
    opp_map = pd.concat([
        t.assign(team=t["home_code"], opp=t["visitor_code"]).loc[:, ["team", "opp"]],
        t.assign(team=t["visitor_code"], opp=t["home_code"]).loc[:, ["team", "opp"]],
    ])
    minutes_today = minutes_today.merge(opp_map, left_on="team.code", right_on="team", how="left")
    minutes_today.rename(columns={"opp": "opponent_code"}, inplace=True)
    minutes_today.drop(columns=["team"], inplace=True)
    return ratings_clean, todays_games, minutes_today


def load_context_for_date(target_pst_dt: datetime):
    ratings, games_filtered, games_all, player_stats = read_inputs()
    ratings_clean = clean_ratings(ratings)
    ps = enrich_player_stats_with_game_dates(player_stats, games_all)

    todays_games = get_todays_games_pst(games_all, target_pst_dt)
    if todays_games.empty:
        return ratings_clean, todays_games, pd.DataFrame()

    rosters = build_today_rosters(ps, todays_games, start_from_pst=target_pst_dt)
    minutes_today = minutes_from_recent_and_ratings(ps, rosters, ratings_clean)

    # Attach opponent codes
    t = todays_games[["home_code", "visitor_code"]].copy()
    opp_map = pd.concat([
        t.assign(team=t["home_code"], opp=t["visitor_code"]).loc[:, ["team", "opp"]],
        t.assign(team=t["visitor_code"], opp=t["home_code"]).loc[:, ["team", "opp"]],
    ])
    minutes_today = minutes_today.merge(opp_map, left_on="team.code", right_on="team", how="left")
    minutes_today.rename(columns={"opp": "opponent_code"}, inplace=True)
    minutes_today.drop(columns=["team"], inplace=True)
    return ratings_clean, todays_games, minutes_today


def load_team_calibration(calib_path=None):
    if calib_path is None:
        calib_path = _INPUT_OVERRIDES.get("CALIBRATION_FILE", os.path.join(DATA_DIR, "sim_calibration.csv"))
    if not os.path.exists(calib_path):
        return {}
    try:
        df = pd.read_csv(calib_path)
        out = {}
        for _, r in df.iterrows():
            team = str(r.get("team", "")).strip().upper()
            if not team:
                continue
            out[team] = {
                "off_ppp_scale": float(r.get("off_ppp_scale", 1.0)),
                "def_ppp_scale": float(r.get("def_ppp_scale", 1.0)),
                "shot_make_scale": float(r.get("shot_make_scale", 1.0)),
                "opp_shot_make_scale": float(r.get("opp_shot_make_scale", 1.0)),
            }
        return out
    except Exception:
        return {}


def build_calibration_from_backtests(window_days=30, write=True):
    # Aggregate recent sim_backtest_games_* files from cache dir and estimate per-team scalers
    src_dir = BACKTEST_CACHE_DIR if os.path.isdir(BACKTEST_CACHE_DIR) else DATA_DIR
    files = [f for f in os.listdir(src_dir) if f.startswith("sim_backtest_games_") and f.endswith(".csv")]
    if not files:
        return pd.DataFrame()
    frames = []
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=window_days)
    for f in files:
        try:
            df = pd.read_csv(os.path.join(src_dir, f))
            if "date_pst" in df.columns:
                df["date_pst"] = pd.to_datetime(df["date_pst"], errors="coerce")
                df = df[df["date_pst"] >= cutoff.tz_localize(None)]
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    bt = pd.concat(frames, ignore_index=True)
    # Compute team-level offensive bias (pred - actual) and defensive bias (opponent pred - actual)
    off_rows = bt[["home", "home_pred", "home_actual"]].rename(columns={"home":"team", "home_pred":"pred", "home_actual":"act"})
    off_rows = pd.concat([off_rows, bt[["away", "away_pred", "away_actual"]].rename(columns={"away":"team", "away_pred":"pred", "away_actual":"act"})])
    off_rows["bias"] = off_rows["pred"].astype(float) - off_rows["act"].astype(float)
    # Defensive bias: how much opponents overperform against them
    def_rows = bt[["home", "away_pred", "away_actual"]].rename(columns={"home":"team", "away_pred":"pred", "away_actual":"act"})
    def_rows = pd.concat([def_rows, bt[["away", "home_pred", "home_actual"]].rename(columns={"away":"team", "home_pred":"pred", "home_actual":"act"})])
    def_rows["bias"] = def_rows["pred"].astype(float) - def_rows["act"].astype(float)

    agg_off = off_rows.groupby("team").agg(off_bias=("bias","mean")).reset_index()
    agg_def = def_rows.groupby("team").agg(def_bias=("bias","mean")).reset_index()
    calib = agg_off.merge(agg_def, on="team", how="outer").fillna(0.0)
    # Convert biases to gentle scalers around 1.0. Assume average ppp ~ 1.1 over ~100 possessions => ~110 points.
    # Scale factor = 1 - bias / 110, clipped 0.9..1.1
    calib["off_ppp_scale"] = (1.0 - calib["off_bias"] / 110.0).clip(0.9, 1.1)
    calib["def_ppp_scale"] = (1.0 - calib["def_bias"] / 110.0).clip(0.9, 1.1)
    # Use smaller effect on make probabilities
    calib["shot_make_scale"] = (1.0 - calib["off_bias"] / 440.0).clip(0.95, 1.05)
    calib["opp_shot_make_scale"] = (1.0 - calib["def_bias"] / 440.0).clip(0.95, 1.05)
    out = calib[["team","off_ppp_scale","def_ppp_scale","shot_make_scale","opp_shot_make_scale"]]
    if write and not out.empty:
        out.to_csv(os.path.join(DATA_DIR, "sim_calibration.csv"), index=False)
    return out


def run_today_simulations(output_dir=None, write_json=False, write_pbp=True, simulations=1, rng_seed=None, pace_scale: float = 1.0, make_nerf: float = 0.10):
    # Resolve output locations
    if output_dir is None:
        output_dir = DATA_DIR
    os.makedirs(output_dir, exist_ok=True)
    if write_pbp:
        os.makedirs(PBP_DIR, exist_ok=True)

    ratings_clean, todays_games, minutes_today = load_today_context()

    # Build team frames with ratings merged
    if "player_name_key" not in minutes_today.columns:
        minutes_today["player_name_key"] = (
            minutes_today["player.firstname"].astype(str).str.strip() + " " +
            minutes_today["player.lastname"].astype(str).str.strip()
        ).str.lower()
    team_frames = {}
    for team, grp in minutes_today.groupby("team.code"):
        tf = grp.merge(ratings_clean, on="player_name_key", how="left")
        team_frames[str(team)] = tf

    # Dynamic per-game pace based on recent indicators
    games_filtered_path = _INPUT_OVERRIDES.get("FILTERED_GAME_SCORES_FILE", os.path.join(DATA_DIR, "filtered_game_scores.csv"))
    pace_lookup = {}
    if os.path.exists(games_filtered_path):
        try:
            gf = pd.read_csv(games_filtered_path)
            base_total = float(pd.to_numeric(gf.get("league_total", pd.Series([220])), errors="coerce").fillna(220).mean()) if "league_total" in gf.columns else 220.0
            for _, r in todays_games.iterrows():
                home = str(r["home_code"]); away = str(r["visitor_code"]) 
                h10 = pd.to_numeric(gf.get("home_last10_total", pd.Series([np.nan])), errors="coerce").dropna()
                a10 = pd.to_numeric(gf.get("away_last10_total", pd.Series([np.nan])), errors="coerce").dropna()
                pace_lookup[(home, away)] = float(np.nanmean([h10.mean() if len(h10) else np.nan, a10.mean() if len(a10) else np.nan, base_total]))
        except Exception:
            pass

    today_str = datetime.now().strftime("%Y%m%d")
    # Load team calibration if available
    team_calib = load_team_calibration()
    per_game_sims = {}  # (home,away) -> list of DataFrames
    per_game_boxes = {} # (home,away) -> list of box DataFrames

    # Seeds
    base_seed = rng_seed if rng_seed is not None else 1234
    seeds = [base_seed + i * 7919 for i in range(max(1, int(simulations)))]

    for sim_idx, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        for _, g in todays_games.iterrows():
            home = str(g["home_code"]) ; away = str(g["visitor_code"]) 
            pace_total = pace_lookup.get((home, away), 220.0) * float(pace_scale)
            pbp, box = simulate_game(home, away, team_frames, pace_total, rng, team_calib=team_calib, make_nerf=make_nerf)
            pbp["game_id"] = g.get("game.id", "")
            pbp["simulation"] = sim_idx
            box["simulation"] = sim_idx
            key = (home, away)
            per_game_sims.setdefault(key, []).append(pbp)
            per_game_boxes.setdefault(key, []).append(box)
            if write_pbp:
                out_name = f"play_by_play_{away}_at_{home}_{today_str}_sim{sim_idx}.csv"
                pbp.to_csv(os.path.join(PBP_DIR, out_name), index=False)

    # Combined outputs and summaries
    combined_pbp = []
    games_summary_rows = []
    players_summary_rows = []

    for (home, away), df_list in per_game_sims.items():
        dfc = pd.concat(df_list, ignore_index=True)
        combined_pbp.append(dfc)
        # Game-level stats by simulation
        sim_totals = dfc.groupby(["simulation"]).agg(home_pts=("score_home", "max"), away_pts=("score_away", "max")).reset_index()
        # Robust summary (median) and spread
        games_summary_rows.append({
            "home": home,
            "away": away,
            "simulations": len(df_list),
            "home_mean": float(sim_totals["home_pts"].mean()),
            "home_median": float(sim_totals["home_pts"].median()),
            "home_p05": float(np.percentile(sim_totals["home_pts"], 5)),
            "home_p95": float(np.percentile(sim_totals["home_pts"], 95)),
            "away_mean": float(sim_totals["away_pts"].mean()),
            "away_median": float(sim_totals["away_pts"].median()),
            "away_p05": float(np.percentile(sim_totals["away_pts"], 5)),
            "away_p95": float(np.percentile(sim_totals["away_pts"], 95)),
        })

        # Player-level points across simulations
        if per_game_boxes.get((home, away)):
            bcat = pd.concat(per_game_boxes[(home, away)], ignore_index=True)
            player_stats = bcat.groupby(["team", "player"]).agg(
                sims=("simulation", "nunique"),
                mean_pts=("points", "mean"),
                median_pts=("points", "median"),
                p05_pts=("points", lambda x: float(np.percentile(x, 5))),
                p95_pts=("points", lambda x: float(np.percentile(x, 95))),
                mean_reb=("rebounds", "mean"),
                median_reb=("rebounds", "median"),
                p05_reb=("rebounds", lambda x: float(np.percentile(x, 5))),
                p95_reb=("rebounds", lambda x: float(np.percentile(x, 95))),
                mean_ast=("assists", "mean"),
                median_ast=("assists", "median"),
                p05_ast=("assists", lambda x: float(np.percentile(x, 5))),
                p95_ast=("assists", lambda x: float(np.percentile(x, 95))),
            ).reset_index()
            player_stats["home"] = home
            player_stats["away"] = away
            players_summary_rows.extend(player_stats.to_dict("records"))

    # Write combined and summaries
    out_files = []
    if combined_pbp:
        combined = pd.concat(combined_pbp, ignore_index=True)
        combined_path_csv = os.path.join(output_dir, f"play_by_play_today_{today_str}.csv")
        combined.to_csv(combined_path_csv, index=False)
        out_files.append(combined_path_csv)
        if write_json:
            try:
                combined_path_json = os.path.join(output_dir, f"play_by_play_today_{today_str}.json")
                combined.to_json(combined_path_json, orient="records")
                out_files.append(combined_path_json)
            except Exception:
                pass

    if games_summary_rows:
        games_summary = pd.DataFrame(games_summary_rows)
        games_summary_path = os.path.join(output_dir, f"games_summary_today_{today_str}.csv")
        games_summary.to_csv(games_summary_path, index=False)
        out_files.append(games_summary_path)

    if players_summary_rows:
        players_summary = pd.DataFrame(players_summary_rows)
        players_summary_path = os.path.join(output_dir, f"players_summary_today_{today_str}.csv")
        players_summary.to_csv(players_summary_path, index=False)
        out_files.append(players_summary_path)

    # Return list of relevant output files for this run
    return out_files


def simulate_for_date(dt_pst: datetime, simulations=100, rng_seed=None, write_pbp=False, pace_scale: float = 1.0, use_calibration: bool = True, make_nerf: float = 0.10):
    ratings_clean, todays_games, minutes_today = load_context_for_date(dt_pst)
    if todays_games is None or todays_games.empty or minutes_today is None or minutes_today.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "player_name_key" not in minutes_today.columns:
        minutes_today["player_name_key"] = (
            minutes_today["player.firstname"].astype(str).str.strip() + " " +
            minutes_today["player.lastname"].astype(str).str.strip()
        ).str.lower()
    team_frames = {}
    for team, grp in minutes_today.groupby("team.code"):
        tf = grp.merge(ratings_clean, on="player_name_key", how="left")
        team_frames[str(team)] = tf

    games_filtered_path = os.path.join(DATA_DIR, "filtered_game_scores.csv")
    pace_lookup = {}
    if os.path.exists(games_filtered_path):
        try:
            gf = pd.read_csv(games_filtered_path)
            base_total = float(pd.to_numeric(gf.get("league_total", pd.Series([220])), errors="coerce").fillna(220).mean()) if "league_total" in gf.columns else 220.0
            for _, r in todays_games.iterrows():
                home = str(r["home_code"]); away = str(r["visitor_code"]) 
                h10 = pd.to_numeric(gf.get("home_last10_total", pd.Series([np.nan])), errors="coerce").dropna()
                a10 = pd.to_numeric(gf.get("away_last10_total", pd.Series([np.nan])), errors="coerce").dropna()
                pace_lookup[(home, away)] = float(np.nanmean([h10.mean() if len(h10) else np.nan, a10.mean() if len(a10) else np.nan, base_total]))
        except Exception:
            pass

    per_game_sims = {}
    per_game_boxes = {}
    base_seed = rng_seed if rng_seed is not None else 1234
    seeds = [base_seed + i * 7919 for i in range(max(1, int(simulations)))]
    team_calib = load_team_calibration() if use_calibration else None
    for sim_idx, seed in enumerate(seeds, start=1):
        rng = np.random.default_rng(seed)
        for _, g in todays_games.iterrows():
            home = str(g["home_code"]) ; away = str(g["visitor_code"]) 
            pace_total = pace_lookup.get((home, away), 220.0) * float(pace_scale)
            pbp, box = simulate_game(home, away, team_frames, pace_total, rng, team_calib=team_calib, make_nerf=make_nerf)
            pbp["game_id"] = g.get("game.id", "")
            pbp["simulation"] = sim_idx
            box["simulation"] = sim_idx
            box["home"] = home
            box["away"] = away
            key = (home, away)
            per_game_sims.setdefault(key, []).append(pbp)
            per_game_boxes.setdefault(key, []).append(box)
            if write_pbp:
                day_dir = os.path.join(DATA_DIR, "Play_by_Play", dt_pst.strftime("%Y%m%d"))
                os.makedirs(day_dir, exist_ok=True)
                pbp.to_csv(os.path.join(day_dir, f"{away}_at_{home}_sim{sim_idx}.csv"), index=False)

    # Build summaries
    games_rows = []
    players_rows = []
    for (home, away), df_list in per_game_sims.items():
        dfc = pd.concat(df_list, ignore_index=True)
        sim_totals = dfc.groupby(["simulation"]).agg(home_pts=("score_home", "max"), away_pts=("score_away", "max")).reset_index()
        games_rows.append({
            "date_pst": dt_pst.date(),
            "home": home,
            "away": away,
            "sims": len(df_list),
            "home_mean": float(sim_totals["home_pts"].mean()),
            "home_median": float(sim_totals["home_pts"].median()),
            "away_mean": float(sim_totals["away_pts"].mean()),
            "away_median": float(sim_totals["away_pts"].median()),
        })
        if per_game_boxes.get((home, away)):
            b = pd.concat(per_game_boxes[(home, away)], ignore_index=True)
            p = b.groupby(["team", "player"]).agg(
                sims=("simulation", "nunique"),
                mean_pts=("points", "mean"), median_pts=("points", "median"),
                mean_reb=("rebounds", "mean"), median_reb=("rebounds", "median"),
                mean_ast=("assists", "mean"), median_ast=("assists", "median"),
                mean_fouls=("fouls", "mean"), median_fouls=("fouls", "median"),
                mean_mins=("minutes", "mean"), median_mins=("minutes", "median"),
            ).reset_index()
            p["date_pst"] = dt_pst.date()
            p["home"] = home
            p["away"] = away
            players_rows.extend(p.to_dict("records"))

    # Also write per-simulation player stats list for the date
    try:
        if players_rows:
            # Reconstruct long-form from per_game_boxes aggregated earlier
            long_players = []
            for (home, away), df_list in per_game_boxes.items():
                for dfb in df_list:
                    dfb2 = dfb.copy()
                    dfb2["date_pst"] = dt_pst.date()
                    # include fouls and minutes if present
                    cols = ["team","player","simulation","points","rebounds","assists","fouls","minutes","home","away","date_pst"]
                    cols = [c for c in cols if c in dfb2.columns]
                    long_players.append(dfb2[cols])
            if long_players:
                long_df = pd.concat(long_players, ignore_index=True)
                out_path = os.path.join(DATA_DIR, f"players_sim_stats_{dt_pst.strftime('%Y%m%d')}.csv")
                long_df.to_csv(out_path, index=False)
    except Exception:
        pass

    return pd.DataFrame(games_rows), pd.DataFrame(players_rows)


def backtest_simulation(start_date=None, end_date=None, simulations=100, rng_seed=7, write_outputs=True, use_calibration: bool = True, pace_scale: float = 1.0, make_nerf: float = 0.10):
    _, _, games_all, player_stats = read_inputs()
    ps = enrich_player_stats_with_game_dates(player_stats, games_all)
    now_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
    season_start = _season_start_for_date(now_pst)
    start_dt = pd.to_datetime(start_date) if start_date else pd.Timestamp(season_start)
    end_dt = pd.to_datetime(end_date) if end_date else pd.Timestamp(now_pst)
    # Ensure TZ-aware PST timestamps
    if start_dt.tz is None:
        start_ts = start_dt.tz_localize("America/Los_Angeles")
    else:
        start_ts = start_dt.tz_convert("America/Los_Angeles")
    if end_dt.tz is None:
        end_ts = end_dt.tz_localize("America/Los_Angeles")
    else:
        end_ts = end_dt.tz_convert("America/Los_Angeles")
    dates = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D", tz="America/Los_Angeles")

    game_metrics = []
    player_metrics = []

    total_days = len(dates)
    for idx, dt in enumerate(dates, start=1):
        # Resolve per-date cache paths
        date_str = str(dt.date())
        games_cache_path = os.path.join(BACKTEST_CACHE_DIR, f"sim_backtest_games_{date_str}.csv")
        players_cache_path = os.path.join(BACKTEST_CACHE_DIR, f"sim_backtest_players_{date_str}.csv")

        # Fetch games for this date to report progress and avoid duplicate work
        games_today = get_todays_games_pst(games_all, dt)
        try:
            print(f"[Backtest] {idx}/{total_days} {date_str} - {len(games_today)} games, sims={int(simulations)}")
        except Exception:
            pass

        # If cache exists for this date, load and continue
        if os.path.exists(games_cache_path):
            try:
                cached_games = pd.read_csv(games_cache_path)
                if not cached_games.empty:
                    game_metrics.extend(cached_games.to_dict("records"))
                if os.path.exists(players_cache_path):
                    cached_players = pd.read_csv(players_cache_path)
                    if not cached_players.empty:
                        player_metrics.extend(cached_players.to_dict("records"))
                print(f"  [cache] Loaded backtest for {date_str}")
                continue
            except Exception:
                # If cache read fails, fall back to recomputing
                pass

        # No cache: run simulations for this date
        gsum, psum = simulate_for_date(dt, simulations=simulations, rng_seed=rng_seed, write_pbp=False, pace_scale=pace_scale, use_calibration=use_calibration, make_nerf=make_nerf)
        if gsum.empty:
            # Nothing to record; still consider writing empty cache to avoid re-attempts
            if write_outputs:
                try:
                    pd.DataFrame().to_csv(games_cache_path, index=False)
                    pd.DataFrame().to_csv(players_cache_path, index=False)
                except Exception:
                    pass
            continue
        # Actual team totals for that date
        day_ps = ps.copy()
        if "game_date_pst" in day_ps.columns:
            day_ps = day_ps[day_ps["game_date_pst"].dt.date == dt.date()]
        else:
            day_ps = day_ps[pd.to_datetime(day_ps["game_date"], errors="coerce").dt.tz_localize("UTC").dt.tz_convert("America/Los_Angeles").dt.date == dt.date()]
        act_team = day_ps.groupby(["game.id", "team.code"]).agg(actual_pts=("points", "sum")).reset_index()
        if not games_today.empty:
            for _, r in games_today.iterrows():
                gid = r.get("game.id")
                home = str(r["home_code"]) ; away = str(r["visitor_code"]) 
                row = gsum[(gsum["home"] == home) & (gsum["away"] == away)]
                if row.empty:
                    continue
                act_home = act_team[(act_team["game.id"] == gid) & (act_team["team.code"] == home)]["actual_pts"].sum()
                act_away = act_team[(act_team["game.id"] == gid) & (act_team["team.code"] == away)]["actual_pts"].sum()
                game_metrics.append({
                    "date_pst": dt.date(), "game.id": gid, "home": home, "away": away,
                    # Use per-simulation average (mean) rather than median
                    "home_pred": float(row.iloc[0]["home_mean"]), "away_pred": float(row.iloc[0]["away_mean"]),
                    "home_actual": float(act_home), "away_actual": float(act_away),
                    "home_error": float(row.iloc[0]["home_mean"] - act_home),
                    "away_error": float(row.iloc[0]["away_mean"] - act_away),
                })
        # Player-level comparison (points; and rebounds/assists if actuals exist)
        if not psum.empty:
            # Prepare a working copy to standardize actual stat columns
            day_ps_work = day_ps.copy()
            # Rebounds: accept common names or sum offensive+defensive variants
            reb_cands = ["rebounds", "reb", "total_rebounds", "tot_reb", "totReb", "totalReb"]
            oreb_cands = ["oreb", "offensive_rebounds", "offensive rebounds", "offensive_reb", "off_reb"]
            dreb_cands = ["dreb", "defensive_rebounds", "defensive rebounds", "defensive_reb", "def_reb"]
            reb_col = next((c for c in reb_cands if c in day_ps_work.columns), None)
            if reb_col is not None:
                day_ps_work["rebounds_actual"] = pd.to_numeric(day_ps_work[reb_col], errors="coerce").fillna(0.0)
            else:
                oreb_col = next((c for c in oreb_cands if c in day_ps_work.columns), None)
                dreb_col = next((c for c in dreb_cands if c in day_ps_work.columns), None)
                if oreb_col and dreb_col:
                    day_ps_work["rebounds_actual"] = (
                        pd.to_numeric(day_ps_work[oreb_col], errors="coerce").fillna(0.0)
                        + pd.to_numeric(day_ps_work[dreb_col], errors="coerce").fillna(0.0)
                    )
            # Assists: normalize common aliases
            ast_cands = ["assists", "ast"]
            ast_col = next((c for c in ast_cands if c in day_ps_work.columns), None)
            if ast_col is not None and ast_col != "assists":
                day_ps_work.rename(columns={ast_col: "assists"}, inplace=True)

            # Build aggregation map dynamically
            agg_kwargs = {"points": ("points", "sum")}
            if "rebounds_actual" in day_ps_work.columns:
                agg_kwargs["rebounds_actual"] = ("rebounds_actual", "sum")
            if "assists" in day_ps_work.columns:
                agg_kwargs["assists"] = ("assists", "sum")

            day_act = day_ps_work.groupby(["team.code", "player.firstname", "player.lastname"]).agg(**agg_kwargs).reset_index()
            # Normalize output names
            if "rebounds_actual" in day_act.columns:
                day_act.rename(columns={"rebounds_actual": "rebounds"}, inplace=True)

            psum["player.firstname"] = psum["player"].str.split().str[0]
            psum["player.lastname"] = psum["player"].str.split().str[-1]
            merged = psum.merge(
                day_act,
                left_on=["team", "player.firstname", "player.lastname"],
                right_on=["team.code", "player.firstname", "player.lastname"],
                how="left",
            )
            # Use per-simulation averages for errors
            merged["player_points_error"] = merged["mean_pts"] - pd.to_numeric(merged.get("points"), errors="coerce").fillna(0.0)
            if "rebounds" in merged.columns and "mean_reb" in merged.columns:
                merged["player_rebounds_error"] = merged["mean_reb"] - pd.to_numeric(merged.get("rebounds"), errors="coerce").fillna(0.0)
            if "assists" in merged.columns and "mean_ast" in merged.columns:
                merged["player_assists_error"] = merged["mean_ast"] - pd.to_numeric(merged.get("assists"), errors="coerce").fillna(0.0)

            player_metrics.extend(merged.to_dict("records"))

        # Persist per-date caches immediately (always write cache)
        try:
            gm_day = pd.DataFrame([m for m in game_metrics if str(m.get("date_pst")) == date_str])
            pl_day = pd.DataFrame([m for m in player_metrics if str(m.get("date_pst")) == date_str])
            gm_day.to_csv(games_cache_path, index=False)
            pl_day.to_csv(players_cache_path, index=False)
            print(f"  [cache] Wrote backtest for {date_str} -> {os.path.basename(games_cache_path)}")
        except Exception:
            pass

    game_bt = pd.DataFrame(game_metrics)
    player_bt = pd.DataFrame(player_metrics)
    return game_bt, player_bt


if __name__ == "__main__":
    # Defaults: write PBP to Play_by_Play, 1 simulation
    paths = run_today_simulations(write_pbp=False, simulations=1)
    print("Saved:")
    for p in paths:
        print(" -", p)
