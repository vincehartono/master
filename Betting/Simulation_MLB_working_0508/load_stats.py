from pybaseball import batting_stats, pitching_stats
import pandas as pd
import os
from mlb_constants import fuzzy_team_abbr, team_abbr

# Extend fuzzy matcher to handle MLB-style codes
mlb_abbr_fallback = {
    'WSN': 'WSH',
    'SDP': 'SD',
    'TBR': 'TB',
    'SFG': 'SF',
    'KCR': 'KC'
}

def ensure_pitcher_column(df):
    if 'outs' not in df.columns:
        df['outs'] = 0  # dummy fallback
    return df

save_dir = r"C:\\Users\\Vince\\master\\Betting\\Simulation_MLB"
batting_file = os.path.join(save_dir, "batting_2023_2025.parquet")
pitching_file = os.path.join(save_dir, "pitching_2023_2025.parquet")

def load_and_save_batting():
    all_years = []
    for year in [2023, 2024, 2025]:
        print(f"Loading batting stats for {year}...")
        df = batting_stats(year, qual=0)
        raw_teams = set(df['Team'].unique())
        unmatched = raw_teams - set(team_abbr.values())
        df['Season'] = year
        df['Team'] = df['Team'].apply(lambda x: fuzzy_team_abbr(mlb_abbr_fallback.get(x, x)))
        df = df[df['Team'].notnull()]  # drop unrecognized teams
        all_years.append(df)
    batting_combined = pd.concat(all_years, ignore_index=True)
    batting_combined.to_parquet(batting_file)
    print(f"Batting stats saved to {batting_file}")

def load_and_save_pitching():
    all_years = []
    for year in [2023, 2024, 2025]:
        print(f"Loading pitching stats for {year}...")
        df = pitching_stats(year, qual=0)
        raw_teams = set(df['Team'].unique())
        unmatched = raw_teams - set(team_abbr.values())
        df['Season'] = year
        df['Team'] = df['Team'].apply(lambda x: fuzzy_team_abbr(mlb_abbr_fallback.get(x, x)))
        df = df[df['Team'].notnull()]  # drop unrecognized teams
        df = ensure_pitcher_column(df)
        all_years.append(df)
    pitching_combined = pd.concat(all_years, ignore_index=True)
    pitching_combined.to_parquet(pitching_file)
    print(f"Pitching stats saved to {pitching_file}")

if __name__ == "__main__":
    load_and_save_batting()
    load_and_save_pitching()
