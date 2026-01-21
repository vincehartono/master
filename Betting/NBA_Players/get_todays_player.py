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

# Load the CSV file into a DataFrame (without assuming the first row is the header)
odds_df = pd.read_csv('odds_today.csv', header=None)

# Step 1: Select rows 1st, 3rd, 9th, 11th, etc.
selected_rows = odds_df.iloc[[i for i in range(len(odds_df)) if i % 6 == 0 or i % 6 == 2]].reset_index(drop=True)

# Step 2: Split into "odd" and "even" rows
odd_rows = selected_rows.iloc[::2].reset_index(drop=True)  # rows like 1st, 9th, 17th...
even_rows = selected_rows.iloc[1::2].reset_index(drop=True)  # rows like 3rd, 11th, 19th...

# Step 3: Clean "Over(" and ")" from even rows (removes "Over(" and ")")
for col in even_rows.columns:
    even_rows[col] = even_rows[col].astype(str)  # Ensure string type
    try:
        even_rows[col] = (
            even_rows[col]
            .str.replace(r"Over\(", "", regex=True)  # Correctly escape '('
            .str.replace(r"\)", "", regex=True)  # Correctly escape ')'
        )
    except Exception as e:
        print(f"Error cleaning column '{col}': {e}")
        print("Problematic data:", even_rows[col].head())

# Step 4: Move cleaned even rows to new columns
combined_odds = odd_rows.copy()
for col in even_rows.columns:
    combined_odds[f"{col}_even"] = even_rows[col]

# Step 5: Assign column names at the end
combined_odds.columns = ['player_name', 'over/under']

# Merge unique_players_today with combined_odds on 'player_name', keeping only rows from combined_odds
merged_odds_and_players = pd.merge(combined_odds, unique_players_today, on='player_name', how='left')

merged_odds_and_players = merged_odds_and_players.dropna(subset=['team.code'])

# Save the final merged DataFrame to CSV
merged_odds_and_players.to_csv('merged_odds_and_players.csv', index=False)

print("Merged data with players saved to 'merged_odds_and_players.csv'.")