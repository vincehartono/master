import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import boto3  # <-- add this

from NBA_Simulation import s3_io

# Ensure parent (NBA_Simulation) is importable when run from subfolder
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Local imports from sibling module
import Predict_Results as PR
import check_and_pull_players
try:
    # When run as a package-module
    from Predict_Results.Learn_Adjustments import learn_team_adjustments
except Exception:
    # When run as a plain script from this folder
    from Learn_Adjustments import learn_team_adjustments

ses = boto3.client("ses", region_name=os.environ.get("AWS_REGION", "us-west-1"))

def main():
    total_start = time.time()
    here = os.path.dirname(__file__)
    data_dir = PARENT_DIR

    # 0) Refresh game and player scores (updates nba_game_scores.csv / player_scores.csv)
    print("[pipeline] Refreshing NBA game and player scores via check_and_pull_players...")
    try:
        check_and_pull_players.main()
    except Exception as e:
        print(f"[pipeline] Warning: check_and_pull_players.main() failed: {e}")

    # 1) Load and normalize history
    hist_raw = PR._read_history(data_dir)
    if hist_raw is None or hist_raw.empty:
        print("[pipeline] No historical game results found. Expected nba_game_scores.csv or Backtest_cache.")
        sys.exit(1)
    hist = PR._normalize_history(hist_raw)

    # 2) Hyperparameter tuning (simple grid) and Backtest from season start
    today_pst = datetime.now(ZoneInfo("America/Los_Angeles"))
    season_start = PR._season_start_for_date(today_pst)
    # Grid search
    grid_last_n = [int(x) for x in os.environ.get("PR_GRID_LASTN", "5,8,10,12,15").split(',')]
    grid_wfor = [float(x) for x in os.environ.get("PR_GRID_WFOR", "0.4,0.5,0.6").split(',')]
    grid_home = [float(x) for x in os.environ.get("PR_GRID_HOME", "0.5,1.0,1.5,2.0").split(',')]
    use_decay = bool(int(os.environ.get("PR_USE_DECAY", "0")))
    grid_hl = [float(x) for x in os.environ.get("PR_GRID_HALFLIFE", "3,5,7").split(',')] if use_decay else [0.0]

    best = None
    for ln in grid_last_n:
        for wf in grid_wfor:
            for ha in grid_home:
                for hl in grid_hl:
                    bt_try = PR.backtest_from_history(hist, season_start, last_n=ln, home_adv=ha, w_for=wf, use_decay=use_decay, half_life=hl)
                    if bt_try.empty:
                        continue
                    mae = float((bt_try["home_error"].abs().mean() + bt_try["away_error"].abs().mean()) / 2.0)
                    cand = (mae, ln, wf, ha, hl)
                    if best is None or mae < best[0]:
                        best = cand

    if best is None:
        # Fallback defaults
        last_n = int(os.environ.get("PR_LAST_N", 10))
        w_for = float(os.environ.get("PR_W_FOR", 0.5))
        home_adv = float(os.environ.get("PR_HOME_ADV", 1.5))
        half_life = 0.0
    else:
        _, last_n, w_for, home_adv, half_life = best

    print(f"[pipeline] Tuned params: last_n={last_n}, w_for={w_for}, home_adv={home_adv}, use_decay={int(use_decay)}, half_life={half_life}")
    bt = PR.backtest_from_history(hist, season_start, last_n=last_n, home_adv=home_adv, w_for=w_for, use_decay=use_decay, half_life=half_life)
    bt_path = os.path.join(here, "backtest_results.csv")
    bt.to_csv(bt_path, index=False)
    # Default simulation spread around model predictions; may be overridden by backtest-derived value
    sim_sigma = None
    if not bt.empty:
        home_mae = float(bt["home_error"].abs().mean())
        away_mae = float(bt["away_error"].abs().mean())
        # Use empirical std of prediction errors as a proxy for simulation noise
        try:
            err_std_home = float(bt["home_error"].std())
            err_std_away = float(bt["away_error"].std())
            sim_sigma = max(1.0, (err_std_home + err_std_away) / 2.0)
            print(f"[pipeline] Derived PR_SIM_STD from backtest errors: home_std={err_std_home:.2f}, away_std={err_std_away:.2f} -> base_sigma={sim_sigma:.2f}")
        except Exception:
            sim_sigma = None
        summary = pd.DataFrame([
            {"metric": "home_mae", "value": home_mae},
            {"metric": "away_mae", "value": away_mae},
            {"metric": "games", "value": int(len(bt))},
        ])
        summary_path = os.path.join(here, "backtest_summary.csv")
        summary.to_csv(summary_path, index=False)
        print(f"[pipeline] Backtest summary: home_mae={home_mae:.2f}, away_mae={away_mae:.2f}, n={len(bt)}")
        print(f"[pipeline] Wrote {bt_path} and {summary_path}")
    else:
        print("[pipeline] Backtest produced no rows (insufficient prior data).")

    # 3) Learn adjustments from backtest
    print("[pipeline] Learning adjustments from backtest...")
    adj_path = learn_team_adjustments(bt_path, here)
    print(f"[pipeline] Wrote adjustments to {adj_path}")

    # 4) Save tuned globals for downstream use
    try:
        params_path = os.path.join(here, "model_adjustments.csv")
        if os.path.exists(params_path):
            params = pd.read_csv(params_path)
        else:
            params = pd.DataFrame(columns=["param","value"])
        upserts = {
            "w_for": w_for,
            "home_adv": home_adv,
            "last_n": last_n,
            "use_decay": int(use_decay),
            "half_life": half_life,
        }
        for k, v in upserts.items():
            if (params["param"] == k).any():
                params.loc[params["param"] == k, "value"] = v
            else:
                params = pd.concat([params, pd.DataFrame([{"param": k, "value": v}])], ignore_index=True)
        params.to_csv(params_path, index=False)
        print(f"[pipeline] Updated tuned params at {params_path}")
    except Exception:
        pass

    # 5) Predict today using learned adjustments and tuned globals
    tg = PR.todays_games(data_dir)
    if tg.empty:
        print("[pipeline] No games found for today. Skipping today predictions.")
        return
    sim_runs = 0
    sim_duration = None
    preds_today = PR.predict_from_history(hist, tg, last_n=last_n, home_adv=home_adv, w_for=w_for, use_decay=use_decay, half_life=half_life)
    # Add model totals/spreads
    try:
        preds_today["model_total"] = pd.to_numeric(preds_today["home_pred"], errors="coerce") + pd.to_numeric(preds_today["away_pred"], errors="coerce")
        preds_today["model_spread"] = pd.to_numeric(preds_today["home_pred"], errors="coerce") - pd.to_numeric(preds_today["away_pred"], errors="coerce")
    except Exception:
        pass

    # Optionally merge betting lines if provided in model_inputs.txt
    try:
        import simulation_engine as SE
        lines_path = SE._INPUT_OVERRIDES.get("GAME_LINES_FILE") if hasattr(SE, "_INPUT_OVERRIDES") else None
        if lines_path and os.path.exists(lines_path):
            lines = pd.read_csv(lines_path)
            # Normalize columns
            # Expected: home, away, spread (home minus away, home favorites negative), total
            if "home" in lines.columns and "away" in lines.columns:
                lines["home"] = lines["home"].astype(str)
                lines["away"] = lines["away"].astype(str)
                keep = [c for c in ["home","away","spread","total"] if c in lines.columns]
                lines = lines[keep]
                merged = preds_today.merge(lines, on=["home","away"], how="left")
                if "spread" in merged.columns:
                    merged["spread_edge"] = pd.to_numeric(merged["model_spread"], errors="coerce") - pd.to_numeric(merged["spread"], errors="coerce")
                if "total" in merged.columns:
                    merged["total_edge"] = pd.to_numeric(merged["model_total"], errors="coerce") - pd.to_numeric(merged["total"], errors="coerce")
                preds_today = merged
            else:
                print(f"[pipeline] GAME_LINES_FILE missing home/away columns: {lines_path}")
    except Exception as e:
        print(f"[pipeline] Failed to merge lines: {e}")

    # Lightweight in-process Monte Carlo simulations based on model predictions
    try:
        sim_runs = int(os.environ.get("PR_SIMULATIONS", "1000"))
    except Exception:
        sim_runs = 0
    if sim_runs > 0:
        # Prefer backtest-derived sigma if available; otherwise fall back to env/default
        if sim_sigma is None:
            try:
                sim_sigma = float(os.environ.get("PR_SIM_STD", "12.0"))
            except Exception:
                sim_sigma = 12.0
        base_sigma = float(sim_sigma)
        print(f"[pipeline] Running {sim_runs} in-process simulations (std={base_sigma} pts) over model predictions...")
        sim_start = time.time()
        rng = np.random.default_rng()
        sim_rows = []
        for _, row in preds_today.iterrows():
            home = str(row["home"])
            away = str(row["away"])
            mu_home = float(pd.to_numeric(row["home_pred"], errors="coerce"))
            mu_away = float(pd.to_numeric(row["away_pred"], errors="coerce"))
            h_samples = rng.normal(loc=mu_home, scale=base_sigma, size=sim_runs)
            a_samples = rng.normal(loc=mu_away, scale=base_sigma, size=sim_runs)
            h_samples = np.clip(h_samples, 60.0, None)
            a_samples = np.clip(a_samples, 60.0, None)
            total_samples = h_samples + a_samples
            spread_samples = h_samples - a_samples
            sim_rows.append({
                "home": home,
                "away": away,
                "simulations": sim_runs,
                "sim_home_mean": float(h_samples.mean()),
                "sim_home_median": float(np.median(h_samples)),
                "sim_home_p05": float(np.percentile(h_samples, 5)),
                "sim_home_p95": float(np.percentile(h_samples, 95)),
                "sim_away_mean": float(a_samples.mean()),
                "sim_away_median": float(np.median(a_samples)),
                "sim_away_p05": float(np.percentile(a_samples, 5)),
                "sim_away_p95": float(np.percentile(a_samples, 95)),
                "sim_total_mean": float(total_samples.mean()),
                "sim_total_median": float(np.median(total_samples)),
                "sim_total_p05": float(np.percentile(total_samples, 5)),
                "sim_total_p95": float(np.percentile(total_samples, 95)),
                "sim_spread_mean": float(spread_samples.mean()),
                "sim_spread_median": float(np.median(spread_samples)),
                "sim_spread_p05": float(np.percentile(spread_samples, 5)),
                "sim_spread_p95": float(np.percentile(spread_samples, 95)),
            })
        sim_df = pd.DataFrame(sim_rows)
        preds_today = preds_today.merge(sim_df, on=["home", "away"], how="left")
        sim_duration = time.time() - sim_start
        per_sim = sim_duration / float(sim_runs)
        print(f"[pipeline] Simulations: {sim_runs} runs in {sim_duration:.1f}s (~{per_sim:.2f}s per simulation)")
    out_today = os.path.join(here, "predicted_game_scores_today.csv")
    preds_today.to_csv(out_today, index=False)
    print(f"[pipeline] Wrote {out_today}")

    # Final timing summary
    total_duration = time.time() - total_start
    print(f"[pipeline] Total pipeline runtime: {total_duration:.1f}s")

    # Note: removed simulation blending step for faster runtime

    s3_io.write_csv(preds_today, "predicted_game_scores_today.csv")
    print(f"[pipeline] Wrote predicted_game_scores_today.csv to S3")

    # 6) Email notification via SES
    try:
        sender = os.environ.get("NBA_EMAIL_FROM")
        recipient = os.environ.get("NBA_EMAIL_TO")
        bucket = os.environ.get("NBA_DATA_BUCKET", "")
        if sender and recipient:
            subject = "NBA pipeline results: predicted_game_scores_today.csv"
            body_text = (
                "The NBA pipeline Lambda just finished.\n"
                f"S3 bucket: {bucket}\n"
                "Key: predicted_game_scores_today.csv\n"
            )
            ses.send_email(
                Source=sender,
                Destination={"ToAddresses": [recipient]},
                Message={
                    "Subject": {"Data": subject},
                    "Body": {"Text": {"Data": body_text}},
                },
            )
            print(f"[pipeline] Sent SES email to {recipient}")
        else:
            print("[pipeline] SES email skipped (NBA_EMAIL_FROM or NBA_EMAIL_TO not set).")
    except Exception as e:
        print(f"[pipeline] Warning: failed to send SES email: {e}")


if __name__ == "__main__":
    main()
