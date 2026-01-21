import pandas as pd
from mlb_constants import team_abbr

# Load your data
batting_df = pd.read_parquet("C:/Users/Vince/master/Betting/Simulation_MLB/batting_2023_2025.parquet")
pitching_df = pd.read_parquet("C:/Users/Vince/master/Betting/Simulation_MLB/pitching_2023_2025.parquet")

# Extract teams from player data
batting_teams = set(batting_df['Team'].unique())
pitching_teams = set(pitching_df['Team'].unique())

# Expected teams
all_teams = set(team_abbr.keys())

# Compare
missing_batting = sorted(all_teams - batting_teams)
missing_pitching = sorted(all_teams - pitching_teams)

print("❌ Missing batting data for:", missing_batting)
print("❌ Missing pitching data for:", missing_pitching)
print("✅ Covered in both:", sorted(all_teams & batting_teams & pitching_teams))
