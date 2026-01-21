from __future__ import annotations

import argparse

from nba_sim.config import SimConfig
from nba_sim.data_loader import load_shots
from nba_sim.engine import simulate_possessions, simulate_single_game_from_nbastatsv3
from nba_sim.playbyplay import load_nbastatsv3
from nba_sim.viz import plot_shot_scatter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NBA simulations.")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Legacy shot-based possession sim
    p_shots = subparsers.add_parser("shots", help="Simulate possessions from shotdetail data.")
    p_shots.add_argument("shots_path", type=str, help="Path to shotdetail CSV/parquet.")
    p_shots.add_argument("--n", type=int, default=1000, help="Number of possessions to simulate.")
    p_shots.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_shots.add_argument("--plot-only", action="store_true", help="Only plot historical shots (no sim).")

    # New: game-level sim from nbastatsv3
    p_game = subparsers.add_parser("game", help="Simulate a full game from nbastatsv3 data.")
    p_game.add_argument("nbastatsv3_path", type=str, help="Path to nbastatsv3_YYYY.csv.")
    p_game.add_argument("game_id", type=str, help="Game ID to simulate.")
    p_game.add_argument("--seed", type=int, default=42, help="Random seed.")

    # Predict today's games using an external scores file (schedule/results)
    p_pred = subparsers.add_parser("predict_today", help="Predict today's games from a scores/schedule CSV.")
    p_pred.add_argument("nbastatsv3_path", type=str, help="Path to nbastatsv3_YYYY.csv.")
    p_pred.add_argument("scores_csv", type=str, help="Path to nba_game_scores.csv (with game_date, visitor_code, home_code).")
    p_pred.add_argument("--n-sims", type=int, default=500, help="Number of simulations per game.")
    p_pred.add_argument("--today", type=str, default="", help="Override today date (YYYY-MM-DD).")
    p_pred.add_argument("--points-scale", type=float, default=2.0, help="Multiplier applied to simulated points.")

    # Backtest against historical scores for a given date
    p_back = subparsers.add_parser("backtest", help="Backtest predictions vs actual scores for a given date.")
    p_back.add_argument("nbastatsv3_path", type=str, help="Path to nbastatsv3_YYYY.csv.")
    p_back.add_argument("scores_csv", type=str, help="Path to nba_game_scores.csv.")
    p_back.add_argument("date", type=str, help="Date to backtest (YYYY-MM-DD).")
    p_back.add_argument("--n-sims", type=int, default=200, help="Number of simulations per game.")
    p_back.add_argument("--points-scale", type=float, default=2.0, help="Multiplier applied to simulated points.")

    # Simple text UI: live play-by-play for a single simulated game
    p_live = subparsers.add_parser("live_game", help="Show a play-by-play simulation for one matchup.")
    p_live.add_argument("nbastatsv3_path", type=str, help="Path to nbastatsv3_YYYY.csv.")
    p_live.add_argument("home_code", type=str, help="Home team code, e.g. HOU.")
    p_live.add_argument("away_code", type=str, help="Away team code, e.g. LAC.")
    p_live.add_argument("--seed", type=int, default=42, help="Random seed.")
    p_live.add_argument("--points-scale", type=float, default=2.0, help="Multiplier applied to simulated points.")
    p_live.add_argument("--sleep", type=float, default=0.5, help="Seconds to wait between plays.")
    p_live.add_argument("--out-csv", type=str, default="", help="If set, write play-by-play to this CSV instead of streaming.")

    # Replay an actual game from nbastatsv3 play-by-play
    p_replay = subparsers.add_parser("replay_game", help="Replay a real game from nbastatsv3 play-by-play.")
    p_replay.add_argument("nbastatsv3_path", type=str, help="Path to nbastatsv3_YYYY.csv.")
    p_replay.add_argument("game_id", type=str, help="Game ID to replay (e.g. 22500001).")
    p_replay.add_argument("--sleep", type=float, default=0.3, help="Seconds to wait between events.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "shots":
        shots = load_shots(args.shots_path)
        plot_shot_scatter(shots)

        if args.plot_only:
            return

        sim_config = SimConfig(n_possessions=args.n, seed=args.seed)
        sim = simulate_possessions(shots, model={}, config=sim_config)

        makes = int(sim["sim_make"].sum())
        print(f"[shots] Simulated {len(sim)} possessions, makes={makes}, pct={makes / len(sim):.3f}")

    elif args.mode == "game":
        df = load_nbastatsv3(args.nbastatsv3_path)
        game_state = simulate_single_game_from_nbastatsv3(df, args.game_id, seed=args.seed)

        scores = {tid: t.score for tid, t in game_state.teams.items()}
        print(f"[game] Simulated game_id={args.game_id}")
        print("Scores (team_id -> points):")
        for tid, score in scores.items():
            print(f"  {tid}: {score}")
        print(f"Events simulated: {len(game_state.events)}")

    elif args.mode == "predict_today":
        import pandas as pd
        from nba_sim.engine import simulate_matchup_from_nbastatsv3

        df_pbp = load_nbastatsv3(args.nbastatsv3_path)
        df_scores = pd.read_csv(args.scores_csv)

        df_scores["game_date"] = pd.to_datetime(df_scores["game_date"])
        if args.today:
            today = pd.to_datetime(args.today).date()
        else:
            today = pd.Timestamp("now").normalize().date()

        today_games = df_scores[df_scores["game_date"].dt.date == today]
        if today_games.empty:
            print(f"[predict_today] No games found for {today}")
            return

        print(f"[predict_today] Games on {today}:")

        for _, row in today_games.iterrows():
            visitor = row["visitor_code"]
            home = row["home_code"]

            home_wins = 0
            away_wins = 0
            n_sims = args.n_sims
            home_score_sum = 0.0
            visitor_score_sum = 0.0

            for i in range(n_sims):
                gs = simulate_matchup_from_nbastatsv3(
                    df_pbp, home, visitor, seed=42 + i, points_scale=args.points_scale
                )
                scores = {tid: t.score for tid, t in gs.teams.items()}
                if len(scores) != 2:
                    continue

                mask = df_pbp["teamTricode"].isin([home, visitor])
                mapping_rows = df_pbp[mask]
                id_to_code = (
                    mapping_rows.dropna(subset=["teamTricode", "teamId"])
                    .groupby("teamId")["teamTricode"]
                    .first()
                    .to_dict()
                )

                home_tid = next((tid for tid, code in id_to_code.items() if code == home), None)
                visitor_tid = next((tid for tid, code in id_to_code.items() if code == visitor), None)
                if home_tid is None or visitor_tid is None:
                    continue

                home_score = scores.get(home_tid, 0)
                visitor_score = scores.get(visitor_tid, 0)

                home_score_sum += home_score
                visitor_score_sum += visitor_score

                if home_score > visitor_score:
                    home_wins += 1
                elif visitor_score > home_score:
                    away_wins += 1

            total = home_wins + away_wins
            if total == 0:
                home_prob = away_prob = 0.5
                avg_home = avg_visitor = 0.0
            else:
                home_prob = home_wins / float(total)
                away_prob = away_wins / float(total)
                avg_home = home_score_sum / float(total)
                avg_visitor = visitor_score_sum / float(total)

            print(
                f"  {visitor} @ {home}: "
                f"home_win_prob={home_prob:.3f}, away_win_prob={away_prob:.3f}, "
                f"avg_scores={home}:{avg_home:.1f} vs {visitor}:{avg_visitor:.1f}"
            )

    elif args.mode == "backtest":
        import pandas as pd
        from nba_sim.engine import simulate_matchup_from_nbastatsv3

        df_pbp = load_nbastatsv3(args.nbastatsv3_path)
        df_scores = pd.read_csv(args.scores_csv)
        df_scores["game_date"] = pd.to_datetime(df_scores["game_date"])

        target = pd.to_datetime(args.date).date()
        day_games = df_scores[df_scores["game_date"].dt.date == target]
        if day_games.empty:
            print(f"[backtest] No games found for {target}")
            return

        print(f"[backtest] {target}:")

        for _, row in day_games.iterrows():
            visitor = row["visitor_code"]
            home = row["home_code"]
            actual_v = row["visitor_score"]
            actual_h = row["home_score"]

            home_sum = 0.0
            vis_sum = 0.0
            n_sims = args.n_sims

            for i in range(n_sims):
                gs = simulate_matchup_from_nbastatsv3(
                    df_pbp, home, visitor, seed=1000 + i, points_scale=args.points_scale
                )
                scores = {tid: t.score for tid, t in gs.teams.items()}
                if len(scores) != 2:
                    continue

                mask = df_pbp["teamTricode"].isin([home, visitor])
                mapping_rows = df_pbp[mask]
                id_to_code = (
                    mapping_rows.dropna(subset=["teamTricode", "teamId"])
                    .groupby("teamId")["teamTricode"]
                    .first()
                    .to_dict()
                )
                home_tid = next((tid for tid, code in id_to_code.items() if code == home), None)
                vis_tid = next((tid for tid, code in id_to_code.items() if code == visitor), None)
                if home_tid is None or vis_tid is None:
                    continue

                home_sum += scores.get(home_tid, 0)
                vis_sum += scores.get(vis_tid, 0)

            avg_h = home_sum / float(n_sims) if n_sims else 0.0
            avg_v = vis_sum / float(n_sims) if n_sims else 0.0

            print(
                f"  {visitor} @ {home} | actual {home}:{actual_h} vs {visitor}:{actual_v} "
                f"| pred {home}:{avg_h:.1f} vs {visitor}:{avg_v:.1f}"
            )

    elif args.mode == "live_game":
        import time
        import pandas as pd
        from nba_sim.engine import simulate_matchup_from_nbastatsv3

        df_pbp = load_nbastatsv3(args.nbastatsv3_path)
        home = args.home_code
        away = args.away_code

        print(f"[live_game] Simulating {away} @ {home}")
        gs = simulate_matchup_from_nbastatsv3(
            df_pbp,
            home,
            away,
            seed=args.seed,
            points_scale=args.points_scale,
        )

        # Map teamId -> tricode for display
        mask = df_pbp["teamTricode"].isin([home, away])
        mapping_rows = df_pbp[mask]
        id_to_code = (
            mapping_rows.dropna(subset=["teamTricode", "teamId"])
            .groupby("teamId")["teamTricode"]
            .first()
            .to_dict()
        )

        home_tid = next((tid for tid, code in id_to_code.items() if code == home), None)
        away_tid = next((tid for tid, code in id_to_code.items() if code == away), None)

        home_score = 0
        away_score = 0
        home_assists = 0
        away_assists = 0
        home_rebounds = 0
        away_rebounds = 0

        rows = []
        synthetic_game_id = f"SIM_{away}_AT_{home}"

        # Basic clock model: 4 quarters of 12 minutes, fixed time step per play.
        total_secs = 4 * 12 * 60
        num_events = sum(1 for ev in gs.events if ev.get("event_type") == "shot")
        step = max(total_secs // max(num_events, 1), 1)

        def clock_for_index(idx: int) -> tuple[int, str]:
            elapsed = idx * step
            if elapsed >= total_secs:
                elapsed = total_secs - 1
            period = min(elapsed // (12 * 60) + 1, 4)
            period_start = (period - 1) * 12 * 60
            within = elapsed - period_start
            remaining = 12 * 60 - within
            m, s = divmod(int(remaining), 60)
            return period, f"{m:02d}:{s:02d}"

        event_index = 0

        for i, ev in enumerate(gs.events, start=1):
            if ev.get("event_type") != "shot":
                continue
            event_index += 1
            period, clock_str = clock_for_index(event_index)

            team_id = ev.get("team_id")
            shot_value = ev.get("shot_value", 0)
            made = ev.get("made", False)
            team_code = id_to_code.get(team_id, "UNK")

            # Simple assist model: some made shots are assisted.
            assisted = False
            if made and shot_value > 1:
                assisted = bool((i * 37) % 100 < 60)  # ~60% assisted heuristic

            if team_id == home_tid and made:
                home_score += shot_value
                if assisted:
                    home_assists += 1
            elif team_id == away_tid and made:
                away_score += shot_value
                if assisted:
                    away_assists += 1

            shot_result = "Made" if made else "Missed"
            assist_text = " (assist)" if assisted else ""

            # Shot event
            row_out = {
                "actionNumber": len(rows) + 1,
                "clock": clock_str,
                "period": period,
                "teamId": int(team_id) if team_id is not None else 0,
                "teamTricode": team_code,
                "personId": 0,
                "playerName": "",
                "playerNameI": "",
                "xLegacy": 0.0,
                "yLegacy": 0.0,
                "shotDistance": 0.0,
                "shotResult": shot_result,
                "isFieldGoal": 1,
                "scoreHome": home_score,
                "scoreAway": away_score,
                "pointsTotal": home_score + away_score,
                "location": "",
                "description": f"{team_code} {shot_value}-pt {shot_result}{assist_text}",
                "actionType": "Shot",
                "subType": "",
                "videoAvailable": 0,
                "shotValue": shot_value,
                "actionId": len(rows) + 1,
                "gameId": synthetic_game_id,
            }
            rows.append(row_out)

            # Rebound event after missed shot
            if not made:
                # Simple rebound model: 80% defensive, 20% offensive.
                off_reb = bool((i * 17) % 100 < 20)
                if team_id == home_tid:
                    # Home shot; away defensive rebound usually.
                    reb_team_id = home_tid if off_reb else away_tid
                else:
                    reb_team_id = away_tid if off_reb else home_tid

                if reb_team_id == home_tid:
                    home_rebounds += 1
                elif reb_team_id == away_tid:
                    away_rebounds += 1

                reb_team_code = id_to_code.get(reb_team_id, "UNK")
                reb_desc = "Offensive Rebound" if off_reb else "Defensive Rebound"

                reb_row = {
                    "actionNumber": len(rows) + 1,
                    "clock": clock_str,
                    "period": period,
                    "teamId": int(reb_team_id) if reb_team_id is not None else 0,
                    "teamTricode": reb_team_code,
                    "personId": 0,
                    "playerName": "",
                    "playerNameI": "",
                    "xLegacy": 0.0,
                    "yLegacy": 0.0,
                    "shotDistance": 0.0,
                    "shotResult": "",
                    "isFieldGoal": 0,
                    "scoreHome": home_score,
                    "scoreAway": away_score,
                    "pointsTotal": home_score + away_score,
                    "location": "",
                    "description": reb_desc,
                    "actionType": "Rebound",
                    "subType": "Offensive" if off_reb else "Defensive",
                    "videoAvailable": 0,
                    "shotValue": 0,
                    "actionId": len(rows) + 1,
                    "gameId": synthetic_game_id,
                }
                rows.append(reb_row)

            if not args.out_csv:
                print(
                    f"Play {i:3d}: {team_code} {shot_value}-pt {shot_result.lower()}{assist_text} | "
                    f"Score {away}: {away_score}  {home}: {home_score}"
                )
                time.sleep(args.sleep)

        if args.out_csv:
            df_out = pd.DataFrame(rows)
            df_out.to_csv(args.out_csv, index=False)
            print(f"[live_game] Saved play-by-play to {args.out_csv}")
        else:
            print(f"[live_game] Final score {away}: {away_score}  {home}: {home_score}")

    elif args.mode == "replay_game":
        import time
        import pandas as pd

        df = load_nbastatsv3(args.nbastatsv3_path)
        gid = str(args.game_id)
        game = df[df["gameId"].astype(str) == gid].copy()
        if game.empty:
            print(f"[replay_game] No rows found for gameId={gid}")
            return

        # Sort by period, then actionNumber
        if "actionNumber" in game.columns:
            game = game.sort_values(["period", "actionNumber"])
        else:
            game = game.sort_values(["period"])

        teams = game["teamTricode"].dropna().unique().tolist()
        if len(teams) >= 2:
            home, away = teams[0], teams[1]
        else:
            home, away = "HOME", "AWAY"

        print(f"[replay_game] Replaying gameId={gid} ({away} @ {home})")

        home_score = 0
        away_score = 0

        for _, row in game.iterrows():
            period = int(row.get("period", 0))
            clock = row.get("clock", "")
            team = row.get("teamTricode", "")
            desc = row.get("description", "")
            action_type = row.get("actionType", "")
            shot_result = row.get("shotResult", "")
            is_fg = int(row.get("isFieldGoal", 0)) == 1
            shot_value = int(row.get("shotValue", 0))

            # Update running score using scoreHome/scoreAway if available
            if pd.notna(row.get("scoreHome")) and pd.notna(row.get("scoreAway")):
                home_score = int(row["scoreHome"])
                away_score = int(row["scoreAway"])
            elif is_fg and shot_result == "Made":
                # Fallback: infer scoring if running totals not provided
                if team == home:
                    home_score += shot_value
                elif team == away:
                    away_score += shot_value

            line = f"Q{period} {clock:>8} | {team:>3} {action_type} - {desc}"
            score_str = f"Score {away}: {away_score}  {home}: {home_score}"
            print(f"{line}\n    {score_str}")
            time.sleep(args.sleep)

        print(f"[replay_game] Final score {away}: {away_score}  {home}: {home_score}")



if __name__ == "__main__":
    main()
