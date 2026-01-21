import pandas as pd
import numpy as np
import os

# Step 1: Define paths to the data files
player_data_path = 'player_scores.csv'
game_data_path = 'nba_game_scores.csv'


# Step 2: Function to check if files exist
def check_files():
    if not os.path.exists(player_data_path):
        raise FileNotFoundError(f"Missing file: {player_data_path}")
    if not os.path.exists(game_data_path):
        raise FileNotFoundError(f"Missing file: {game_data_path}")
    print("Files exist, proceeding to data validation.")


# Step 3: Load the CSV files and validate content
def load_and_validate_data():
    try:
        player_scores = pd.read_csv(player_data_path)
        game_scores = pd.read_csv(game_data_path)
    except pd.errors.ParserError:
        raise ValueError("Error parsing one of the data files.")

    print(f"Loaded {len(player_scores)} rows of player data.")
    print(f"Loaded {len(game_scores)} rows of game data.")

    # Validation: Check for missing data
    check_missing_data(player_scores, 'player_scores')
    check_missing_data(game_scores, 'game_scores')

    return player_scores, game_scores


# Step 4: Check if columns are missing (basic validation)
def check_missing_data(df, df_name):
    required_columns = {
        'player_scores': ['player.firstname', 'player.lastname', 'team.code', 'game_date', 'fga', 'fgm'],
        'game_scores': ['game_date', 'home_code', 'visitor_code']
    }

    missing_cols = [col for col in required_columns[df_name] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {df_name}: {', '.join(missing_cols)}")
    print(f"All required columns are present in {df_name}.")


# Step 5: Check if players have enough data
def check_players(player_scores):
    insufficient_data = player_scores[player_scores['min'] < 5]  # example condition: check for players with <5 minutes
    if len(insufficient_data) > 0:
        print(f"Warning: Found {len(insufficient_data)} players with insufficient playtime (less than 5 minutes).")
    else:
        print("All players have sufficient playtime.")


# Step 6: Trigger the validation process
def validate():
    check_files()
    player_scores, game_scores = load_and_validate_data()
    check_players(player_scores)

    print("Data validation complete.")


# Run validation
if __name__ == '__main__':
    validate()
