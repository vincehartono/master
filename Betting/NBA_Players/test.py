from datetime import datetime
import pandas as pd
from pytz import timezone

def compare_game_date(csv_file, game_id):
    try:
        # Load the CSV file
        print("Loading CSV file...")
        df = pd.read_csv(csv_file)
        print(f"CSV loaded. Rows: {len(df)}")

        # Convert game.id to string to ensure consistent filtering
        df['game.id'] = df['game.id'].astype(str)
        print(f"Unique game IDs: {df['game.id'].nunique()}")

        # Filter for the specific game ID
        print(f"Filtering for game ID {game_id}...")
        game_row = df[df['game.id'] == str(game_id)]
        if game_row.empty:
            print(f"No entry found for Game ID {game_id}")
            return

        # Get game_date for the specific game ID
        game_date = pd.to_datetime(game_row['game_date'].values[0])
        print(f"Original game_date: {game_date}")

        # Convert game_date to Pacific Time
        pacific_time = timezone('US/Pacific')
        game_date_pacific = game_date.tz_convert(pacific_time)
        print(f"Pacific game_date: {game_date_pacific}")

        # Get today's date in Pacific Time
        today_pacific = datetime.now(pacific_time).date()
        print(f"Today's date in Pacific Time: {today_pacific}")

        # Compare dates
        if game_date_pacific.date() < today_pacific:
            print(f"Game ID {game_id} is before today ({today_pacific}). game_date: {game_date_pacific}")
        elif game_date_pacific.date() == today_pacific:
            print(f"Game ID {game_id} is scheduled for today ({today_pacific}). game_date: {game_date_pacific}")
        else:
            print(f"Game ID {game_id} is after today ({today_pacific}). game_date: {game_date_pacific}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
compare_game_date("nba_game_scores.csv", 14563)
