import pandas as pd
from datetime import datetime
import pytz

# Load the game data
file_path = "nba_game_scores.csv"
df = pd.read_csv(file_path)

# Convert 'game_date' to datetime and ensure it's timezone-aware
df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
if df['game_date'].dt.tz is None:
    df['game_date'] = df['game_date'].dt.tz_localize(pytz.UTC)

# Convert to Pacific Time
pacific = pytz.timezone('US/Pacific')
df['game_date_pacific'] = df['game_date'].dt.tz_convert(pacific)
df.drop('game_date', axis=1, inplace=True)

# Ensure required columns exist
required_columns = {'visitor_code', 'home_code', 'game_date_pacific'}
if not required_columns.issubset(df.columns):
    raise ValueError(f"The following required columns are missing in the dataset: {required_columns}")

# Reshape the dataframe to include 'location' column
visitor_df = df[['visitor_code', 'game_date_pacific']].rename(columns={'visitor_code': 'team_code'})
visitor_df['location'] = 'visitor'

home_df = df[['home_code', 'game_date_pacific']].rename(columns={'home_code': 'team_code'})
home_df['location'] = 'home'

df_with_location = pd.concat([visitor_df, home_df], ignore_index=True)

# Filter games for today in Pacific Time
today_pacific = datetime.now(pacific).date()
df_today = df_with_location[df_with_location['game_date_pacific'].dt.date == today_pacific]

# Load player minutes data
simulated_minutes_path = "simulated_player_metrics.csv"
player_minutes_df = pd.read_csv(simulated_minutes_path)

# Filter players on today's teams
relevant_teams = df_today['team_code'].unique()
filtered_players = player_minutes_df[player_minutes_df['team.code'].isin(relevant_teams)]

# Merge location info with player data based on the team code
merged_players = pd.merge(filtered_players, df_today[['team_code', 'location']], left_on='team.code', right_on='team_code', how='inner')

# Get unique players and their simulated minutes, including location
unique_players_today = merged_players[['player_name', 'team.code', 'simulated_min', 'simulated_uPER', 'normalized_PER', 'location']].drop_duplicates()

# Save to CSV
output_path = "unique_players_today.csv"
unique_players_today.to_csv(output_path, index=False)

print(f"Unique players data with location saved to: {output_path}")
