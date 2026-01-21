import os
import pandas as pd
from simulate_game import Game, build_team
from mlb_constants import team_abbr, fuzzy_team_abbr

# Setup
team_name_to_abbr = {v: k for k, v in team_abbr.items()}
DATA_DIR = r"C:\\Users\\Vince\\master\\Betting\\Simulation_MLB"
BATTING_FILE = os.path.join(DATA_DIR, "batting_2023_2025.parquet")
PITCHING_FILE = os.path.join(DATA_DIR, "pitching_2023_2025.parquet")
SCHEDULE_FILE = os.path.join(DATA_DIR, "todays_schedule.parquet")
RESULTS_FILE = os.path.join(DATA_DIR, "simulated_results.csv")
SKIPPED_FILE = os.path.join(DATA_DIR, "skipped_games.csv")
PLAYER_STATS_FILE = os.path.join(DATA_DIR, "player_stats.csv")
AVERAGED_STATS_FILE = os.path.join(DATA_DIR, "averaged_player_stats.csv")

# Load data
batting_df = pd.read_parquet(BATTING_FILE)
pitching_df = pd.read_parquet(PITCHING_FILE)
schedule_df = pd.read_parquet(SCHEDULE_FILE)

# 🧠 Ensure platoon columns exist
if 'Side' not in batting_df.columns:
    batting_df['Side'] = 'R'  # Assume right-handed batters as default

if 'Throws' not in pitching_df.columns:
    pitching_df['Throws'] = 'R'  # Assume right-handed pitchers as default

simulated_games = set()
results = []
skipped_games = []
all_player_stats = []

NUM_SIMULATIONS = 10

for _, row in schedule_df.iterrows():
    home, away = row['Home'], row['Away']
    home_sp = row.get('Home SP')
    away_sp = row.get('Away SP')
    game_id = tuple(sorted([home, away]))

    if game_id in simulated_games:
        continue

    try:
        home_abbr = team_name_to_abbr.get(home) or fuzzy_team_abbr(home)
        away_abbr = team_name_to_abbr.get(away) or fuzzy_team_abbr(away)
        if not home_abbr or not away_abbr:
            raise ValueError("Missing team abbreviation")

        if batting_df[batting_df['Team'] == home_abbr].empty or pitching_df[pitching_df['Team'] == home_abbr].empty:
            raise ValueError(f"No data for home team: {home_abbr}")
        if batting_df[batting_df['Team'] == away_abbr].empty or pitching_df[pitching_df['Team'] == away_abbr].empty:
            raise ValueError(f"No data for away team: {away_abbr}")

        sim_scores = []
        sim_stats = []

        for _ in range(NUM_SIMULATIONS):
            team1 = build_team(batting_df, pitching_df, away_abbr, away, starter_name=away_sp)
            team2 = build_team(batting_df, pitching_df, home_abbr, home, starter_name=home_sp)

            if not team1.lineup or not team2.lineup:
                raise ValueError("One or both lineups are empty.")

            game = Game(team1, team2)
            game.simulate_game(allow_mid_inning_pitch_change=True)

            sim_scores.append((team1.score, team2.score))
            sim_stats.extend(game.get_player_stats())

        avg_score1 = round(sum(s[0] for s in sim_scores) / NUM_SIMULATIONS, 2)
        avg_score2 = round(sum(s[1] for s in sim_scores) / NUM_SIMULATIONS, 2)

        results.append({
            'Team1': away,
            'Team2': home,
            'AvgScore1': avg_score1,
            'AvgScore2': avg_score2
        })

        all_player_stats.extend(sim_stats)
        simulated_games.add(game_id)

    except Exception as e:
        skipped_games.append({'Team1': away, 'Team2': home, 'Error': str(e)})

# Save game-level results
if results:
    pd.DataFrame(results).to_csv(RESULTS_FILE, index=False)

if skipped_games:
    pd.DataFrame(skipped_games).to_csv(SKIPPED_FILE, index=False)

# Save player-level stats
if all_player_stats:
    df_all_stats = pd.DataFrame(all_player_stats)
    df_all_stats.to_csv(PLAYER_STATS_FILE, index=False)

    numeric_cols = ['Hits', 'RBI', 'IP', 'ER']
    averaged_df = (
        df_all_stats
        .groupby(['Team', 'Player', 'Pitcher'], as_index=False)[numeric_cols]
        .mean()
        .round(2)
    )

    averaged_df.to_csv(AVERAGED_STATS_FILE, index=False)
