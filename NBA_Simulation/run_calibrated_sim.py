import argparse
from datetime import datetime
from zoneinfo import ZoneInfo

# Local imports
from simulation_engine import (
    backtest_simulation,
    build_calibration_from_backtests,
    run_today_simulations,
)


def main():
    p = argparse.ArgumentParser(description="Backtest -> Build Calibration -> Simulate Today")
    p.add_argument("--start-date", type=str, default=None, help="Backtest start date (YYYY-MM-DD, PST)")
    p.add_argument("--end-date", type=str, default=None, help="Backtest end date (YYYY-MM-DD, PST). Default = today")
    p.add_argument("--bt-sims", type=int, default=100, help="Simulations per date for backtest")
    p.add_argument("--bt-seed", type=int, default=7, help="RNG seed for backtest")
    p.add_argument("--bt-days", type=int, default=None, help="Limit backtest to the most recent N days (overrides --start-date)")
    p.add_argument("--bt-write", action="store_true", help="Write backtest CSV outputs (games/players)")
    p.add_argument("--window-days", type=int, default=30, help="Days of backtests to aggregate for calibration")
    p.add_argument("--today-sims", type=int, default=200, help="Simulations for today's run")
    p.add_argument("--today-seed", type=int, default=42, help="RNG seed for today's run")
    p.add_argument("--write-pbp", action="store_true", help="Write per-game play-by-play files")
    p.add_argument("--skip-backtest", action="store_true", help="Skip backtest step and use existing cached backtest files for calibration")
    p.add_argument("--reuse-backtest", action="store_true", help="Alias for --skip-backtest; reuse previously saved sim_backtest_* files")
    p.add_argument("--skip-calibration", action="store_true", help="Skip calibration rebuild; reuse existing sim_calibration.csv")
    p.add_argument("--learn-caps", action="store_true", help="Learn usage caps from historical boxes before running")
    p.add_argument("--report-top-performers", action="store_true", default=True, help="Report top 10 players with lowest prediction error (pts/reb/ast)")
    p.add_argument("--report-last-n", type=int, default=5, help="Number of most recent games per player for error report")
    p.add_argument("--report-only", action="store_true", help="Only generate the top performers report from cache and exit")
    p.add_argument("--no-report", action="store_true", help="Do not generate the top performers report")
    p.add_argument("--no-calib-backtest", action="store_true", help="Do not use team calibration during backtest (avoid circularity)")
    p.add_argument("--pace-scale", type=float, default=1.0, help="Scale factor applied to game pace (e.g., 1.07 to increase points)")
    p.add_argument("--make-nerf", type=float, default=0.10, help="Subtract this from base make prob before calibration (smaller = more scoring)")
    # New toggles for faster iteration
    p.add_argument(
        "--sims",
        type=int,
        default=None,
        help="Override: use this number of simulations for both backtest and today",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Shortcut for quick runs: sets --sims to a small number (e.g., 10)",
    )
    args = p.parse_args()

    # Default: run report unless explicitly disabled
    if args.no_report:
        args.report_top_performers = False
    else:
        args.report_top_performers = True if args.report_top_performers is None else args.report_top_performers

    # Apply fast/sims overrides
    if args.fast and args.sims is None:
        args.sims = 10
    if args.sims is not None:
        args.bt_sims = args.sims
        args.today_sims = args.sims

    # Optional: learn usage caps first
    if args.learn_caps:
        try:
            from learn_usage_caps import learn_usage_caps as _learn_caps
            import os
            data_dir = os.path.dirname(__file__)
            box = os.path.join(data_dir, "player_scores.csv")
            out = os.path.join(data_dir, "usage_caps.csv")
            _learn_caps(box, out)
            print(f"[0/3] Learned usage caps -> {out}")
        except Exception as e:
            print(f"[0/3] Learn caps failed: {e}")

    # If report-only, generate report from cache and exit
    if args.report_only:
        args.report_top_performers = True
        # fall through to the report section at the end without running steps 1-3
    else:
        pass

    # Step 1: Backtest
    # If --bt-days provided, compute start-date from end-date or today
    if args.bt_days is not None and args.bt_days > 0:
        end_ref = args.end_date or datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")
        try:
            end_dt = datetime.strptime(end_ref, "%Y-%m-%d")
        except ValueError:
            end_dt = datetime.now(ZoneInfo("America/Los_Angeles")).replace(hour=0, minute=0, second=0, microsecond=0)
        from datetime import timedelta
        start_dt = end_dt - timedelta(days=args.bt_days - 1)
        args.start_date = start_dt.strftime("%Y-%m-%d")
    if args.reuse_backtest:
        args.skip_backtest = True
    if not args.skip_backtest:
        print("[1/3] Running backtest...")
        games_bt, players_bt = backtest_simulation(
            start_date=args.start_date,
            end_date=args.end_date,
            simulations=args.bt_sims,
            rng_seed=args.bt_seed,
            write_outputs=args.bt_write,
            use_calibration=(not args.no_calib_backtest),
            pace_scale=args.pace_scale,
            make_nerf=args.make_nerf,
        )
        print(f"  Backtest dates: {args.start_date or 'season_start'} -> {args.end_date or 'today'}")
        print(f"  Sims used: backtest={args.bt_sims}, today={args.today_sims}")
        if not games_bt.empty:
            mae_home = (games_bt["home_error"].abs()).mean()
            mae_away = (games_bt["away_error"].abs()).mean()
            print(f"  Game MAE: home={mae_home:.2f}, away={mae_away:.2f}")
        else:
            print("  No games found during backtest range.")
    else:
        print("[1/3] Skipping backtest (per flag)")

    # Step 2: Build calibration
    if not args.skip_calibration:
        print("[2/3] Building team calibration from backtests...")
        calib_df = build_calibration_from_backtests(window_days=args.window_days, write=True)
        if calib_df is not None and not calib_df.empty:
            print(f"  Wrote sim_calibration.csv with {len(calib_df)} teams.")
        else:
            print("  No calibration created (insufficient backtests).")
    else:
        print("[2/3] Skipping calibration (per flag)")

    # Step 3: Simulate today with calibration (skip if report-only)
    if not args.report_only:
        print("[3/3] Running today's calibrated simulations...")
        outs = run_today_simulations(write_pbp=args.write_pbp, simulations=args.today_sims, rng_seed=args.today_seed, pace_scale=args.pace_scale, make_nerf=args.make_nerf)
        print("  Outputs:")
        for pth in outs:
            print(f"   - {pth}")

        # Build players_top_summary: top 20 by predicted points (filter pred > 5)
        try:
            import pandas as pd
            import os as _os
            from zoneinfo import ZoneInfo as _ZI
            today_str = datetime.now(_ZI("America/Los_Angeles")).strftime("%Y%m%d")
            players_path = _os.path.join(_os.path.dirname(__file__), f"players_summary_today_{today_str}.csv")
            if _os.path.exists(players_path):
                dft = pd.read_csv(players_path)
                # Normalize columns
                if "player" in dft.columns:
                    dft["player_name"] = dft["player"].astype(str)
                else:
                    fn = dft.get("player.firstname").astype(str).fillna("")
                    ln = dft.get("player.lastname").astype(str).fillna("")
                    dft["player_name"] = (fn.str.strip() + " " + ln.str.strip()).str.strip()
                team_col = "team" if "team" in dft.columns else ("team.code" if "team.code" in dft.columns else None)
                if team_col is None:
                    dft["team_display"] = ""
                else:
                    dft["team_display"] = dft[team_col].astype(str)
                # Use mean_pts if available, else fall back to median_pts
                pred_col = "mean_pts" if "mean_pts" in dft.columns else ("median_pts" if "median_pts" in dft.columns else None)
                if pred_col is not None:
                    dfp = dft[["player_name", "team_display", pred_col]].rename(columns={pred_col: "pred_pts"}).copy()
                    dfp = dfp[pd.to_numeric(dfp["pred_pts"], errors="coerce") > 5]
                    dfp = dfp.sort_values("pred_pts", ascending=False).head(20)
                    out_top = _os.path.join(_os.path.dirname(__file__), f"players_top_summary_today_{today_str}.csv")
                    dfp.to_csv(out_top, index=False)
                    print(f"  Wrote {out_top} (top 20 by predicted points > 5)")
                # Also write potential lineups (starters = top 5 by pred_minutes per team)
                if "pred_minutes" in dft.columns:
                    pl = dft.copy()
                    team_col = "team" if "team" in pl.columns else ("team.code" if "team.code" in pl.columns else None)
                    if team_col is not None:
                        pl["pred_minutes"] = pd.to_numeric(pl["pred_minutes"], errors="coerce").fillna(0.0)
                        starters = (pl.sort_values([team_col, "pred_minutes"], ascending=[True, False])
                                      .groupby(team_col)
                                      .head(5)
                                   )
                        starters = starters.assign(role="starter")
                        bench = pl.merge(starters[["player_name"]], on="player_name", how="left", indicator=True)
                        bench = bench[bench["_merge"] == "left_only"].drop(columns=["_merge"]).assign(role="bench")
                        pot = pd.concat([starters, bench], ignore_index=True)
                        out_lineups = _os.path.join(_os.path.dirname(__file__), f"potential_lineups_today_{today_str}.csv")
                        keep_cols = [c for c in ["player_name", team_col, "pred_minutes", "role"] if c in pot.columns]
                        pot[keep_cols].to_csv(out_lineups, index=False)
                        print(f"  Wrote {out_lineups} (potential lineups)")
        except Exception as _e:
            print(f"  [warn] Failed to build players_top_summary: {_e}")

    # Optional: report top performers with lowest error over recent games
    if args.report_top_performers:
        try:
            import os
            import pandas as pd
            cache_dir = os.path.join(os.path.dirname(__file__), "Backtest_cache")
            files = [os.path.join(cache_dir, f) for f in os.listdir(cache_dir) if f.startswith("sim_backtest_players_") and f.endswith(".csv")]
            if not files:
                print("[report] No backtest player caches found. Run backtest with --bt-write or ensure cache exists.")
            else:
                frames = []
                for f in files:
                    try:
                        df = pd.read_csv(f)
                        frames.append(df)
                    except Exception:
                        pass
                if not frames:
                    print("[report] Unable to read player cache files.")
                else:
                    df = pd.concat(frames, ignore_index=True)
                    # Ensure date sortable
                    if "date_pst" in df.columns:
                        df["date_pst"] = pd.to_datetime(df["date_pst"], errors="coerce")
                    else:
                        # fallback if missing
                        df["date_pst"] = pd.to_datetime(df.get("date"), errors="coerce")
                    # Build a player key
                    if "player" in df.columns:
                        df["player_name"] = df["player"].astype(str)
                    else:
                        fn = df.get("player.firstname").astype(str).fillna("")
                        ln = df.get("player.lastname").astype(str).fillna("")
                        df["player_name"] = (fn.str.strip() + " " + ln.str.strip()).str.strip()
                    team_col = "team" if "team" in df.columns else ("team.code" if "team.code" in df.columns else None)
                    if team_col is None:
                        df["team_display"] = ""
                    else:
                        df["team_display"] = df[team_col].astype(str)

                    # Build today's player set to filter to those playing today
                    today_players = set()
                    try:
                        out_dir = os.path.dirname(__file__)
                        from zoneinfo import ZoneInfo as _ZI
                        today_str = datetime.now(_ZI("America/Los_Angeles")).strftime("%Y%m%d")
                        today_players_path = os.path.join(out_dir, f"players_summary_today_{today_str}.csv")
                        if os.path.exists(today_players_path):
                            df_today_names = pd.read_csv(today_players_path)
                            if "player" in df_today_names.columns:
                                df_today_names["player_name"] = df_today_names["player"].astype(str)
                            else:
                                fn = df_today_names.get("player.firstname").astype(str).fillna("")
                                ln = df_today_names.get("player.lastname").astype(str).fillna("")
                                df_today_names["player_name"] = (fn.str.strip() + " " + ln.str.strip()).str.strip()
                            today_players = set(df_today_names["player_name"].dropna().astype(str))
                        else:
                            print("[report] Today's players file not found; cannot filter to those playing today.")
                    except Exception:
                        pass

                    # Helper to compute top N by mean absolute error over the last N games per player, including avg predicted/actual
                    def top_by_metric(pred_col, act_col, label):
                        d = df.dropna(subset=[pred_col])
                        d[act_col] = pd.to_numeric(d.get(act_col), errors="coerce")
                        d = d.dropna(subset=[act_col])
                        d = d.sort_values(["player_name", "date_pst"])  # ascending by date
                        # take last N per player
                        d["rn"] = d.groupby("player_name").cumcount(ascending=True)
                        # Using tail via groupby.nth of negative indices is messy; filter by last N using rank from end
                        d["rn_rev"] = d.groupby("player_name").cumcount(ascending=True)
                        # Map max index per player
                        max_idx = d.groupby("player_name")["rn_rev"].transform("max")
                        d_recent = d[max_idx - d["rn_rev"] < args.report_last_n]
                        d_recent["abs_err"] = (pd.to_numeric(d_recent[pred_col], errors="coerce") - d_recent[act_col]).abs()
                        agg = d_recent.groupby(["player_name", "team_display"], as_index=False).agg(
                            mae=("abs_err", "mean"),
                            games=("abs_err", "count"),
                            pred_mean=(pred_col, "mean"),
                            act_mean=(act_col, "mean"),
                        )
                        agg = agg[agg["games"] >= max(1, args.report_last_n // 2)]
                        # Filter to players active today, if we have that list
                        if today_players:
                            agg = agg[agg["player_name"].isin(today_players)]
                        out = agg.sort_values("mae").head(20)
                        print(f"[report] Top 20 lowest MAE for {label} over last {args.report_last_n} games:")
                        for _, r in out.iterrows():
                            print(
                                f"  {r['player_name']} ({r['team_display']}): MAE={r['mae']:.2f}, "
                                f"pred_mean={r['pred_mean']:.2f}, act_mean={r['act_mean']:.2f} over {int(r['games'])} games"
                            )
                        return out.assign(stat=label)

                    # Compute per metric if available
                    outputs = []
                    # Only compute for points
                    if "mean_pts" in df.columns and ("points" in df.columns):
                        res = top_by_metric("mean_pts", "points", "points")
                        if res is not None:
                            outputs.append(res)

                    # Write only a single combined CSV, attach today's predicted pts if available, and merge betting lines if present
                    try:
                        out_dir = os.path.dirname(__file__)
                        ts_suffix = f"last{args.report_last_n}"
                        if outputs:
                            combined = pd.concat(outputs, ignore_index=True)
                            # Attach today's predictions if today's players_summary file exists (only points kept)
                            try:
                                from zoneinfo import ZoneInfo as _ZI
                                today_str = datetime.now(_ZI("America/Los_Angeles")).strftime("%Y%m%d")
                                today_players_path = os.path.join(out_dir, f"players_summary_today_{today_str}.csv")
                                if os.path.exists(today_players_path):
                                    df_today = pd.read_csv(today_players_path)
                                    if "player" in df_today.columns:
                                        df_today["player_name"] = df_today["player"].astype(str)
                                    else:
                                        fn = df_today.get("player.firstname").astype(str).fillna("")
                                        ln = df_today.get("player.lastname").astype(str).fillna("")
                                        df_today["player_name"] = (fn.str.strip() + " " + ln.str.strip()).str.strip()
                                    pred_cols_map = {"mean_pts": "pred_pts"}
                                    avail = [c for c in pred_cols_map if c in df_today.columns]
                                    df_today = df_today[["player_name"] + avail].rename(columns=pred_cols_map)
                                    # Collapse multiple rows per player_name (if any) to a single prediction
                                    if "pred_pts" in df_today.columns:
                                        df_today = (
                                            df_today.groupby("player_name", as_index=False)["pred_pts"].mean()
                                        )
                                    combined = combined.merge(df_today, on="player_name", how="left")
                                    # If merge created duplicates due to upstream dupes, prefer rows with non-null pred_pts
                                    if "pred_pts" in combined.columns:
                                        combined["_pred_notna"] = combined["pred_pts"].notna().astype(int)
                                        combined = (
                                            combined.sort_values(["player_name","team_display","stat","_pred_notna"], ascending=[True,True,True,False])
                                                    .drop_duplicates(subset=["player_name","team_display","stat"], keep="first")
                                                    .drop(columns=["_pred_notna"])
                                        )
                                    # Also update console print lines to show predictions when available
                                    pass
                            except Exception:
                                pass
                            # Keep useful columns only
                            keep_cols = ["player_name","team_display","stat","mae","games","pred_pts"]
                            for col in keep_cols:
                                if col not in combined.columns:
                                    combined[col] = None
                            combined = combined[keep_cols]
                            # Merge player points lines if configured
                            try:
                                # read model_inputs for optional line paths
                                from simulation_engine import _INPUT_OVERRIDES as _IOV
                                player_lines = _IOV.get("PLAYER_LINES_FILE")
                                if player_lines and os.path.exists(player_lines):
                                    lp = pd.read_csv(player_lines)
                                    if "player_name" not in lp.columns and "player" in lp.columns:
                                        lp["player_name"] = lp["player"].astype(str)
                                    if "pts_line" in lp.columns:
                                        lp2 = lp[["player_name","pts_line"]].copy()
                                        combined = combined.merge(lp2, on="player_name", how="left")
                                        if "pts_line" in combined.columns:
                                            combined["line_diff"] = pd.to_numeric(combined["pred_pts"], errors="coerce") - pd.to_numeric(combined["pts_line"], errors="coerce")
                            except Exception:
                                pass

                            # Merge game lines (spread/total) for reference if available
                            try:
                                from simulation_engine import _INPUT_OVERRIDES as _IOV2
                                game_lines = _IOV2.get("GAME_LINES_FILE")
                                if game_lines and os.path.exists(game_lines):
                                    gl = pd.read_csv(game_lines)
                                    # Attach to combined only as context (not per-row accurate without game mapping)
                                    # We will also write a separate picks CSV leveraging player lines only for now.
                                    pass
                            except Exception:
                                pass

                            # Derive a picks view: prioritize low MAE and farthest from line
                            picks = combined.dropna(subset=["mae","pred_pts","pts_line"]) if "pts_line" in combined.columns else pd.DataFrame()
                            if not picks.empty:
                                picks_sorted = picks.sort_values(["mae", "line_diff"], ascending=[True, False]).head(20)
                                picks_path = os.path.join(out_dir, f"top_performers_picks_{ts_suffix}.csv")
                                picks_sorted.to_csv(picks_path, index=False)
                                print(f"[report] Wrote picks CSV to {picks_path}")
                            combined.to_csv(os.path.join(out_dir, f"top_performers_all_{ts_suffix}.csv"), index=False)
                            print(f"[report] Wrote combined CSV to {out_dir} (top_performers_all_{ts_suffix}.csv)")
                    except Exception:
                        pass
            # End early if report-only
            if args.report_only:
                return
        except Exception as e:
            print(f"[report] Failed to build top-performers report: {e}")
            if args.report_only:
                return


if __name__ == "__main__":
    main()
