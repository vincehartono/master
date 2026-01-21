import http.client
import json
import pandas as pd
import time
import logging
from datetime import datetime
from pytz import timezone

REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 30
LOG_FILE = "processed_games.log"
player_scores_file = "player_scores.csv"

# Set up logging
logging.basicConfig(
    filename='pull_players_data.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def fetch_and_save_player_statistics(game_id, api_key):
    conn = http.client.HTTPSConnection("api-nba-v1.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "api-nba-v1.p.rapidapi.com"
    }

    try:
        conn.request("GET", f"/players/statistics?game={game_id}", headers=headers)
        res = conn.getresponse()
        if res.status == 200:
            data = res.read()
            decoded_data = json.loads(data.decode("utf-8"))
            df = pd.json_normalize(decoded_data['response'])
            df = df.drop(columns=['team.logo'])
            df = df[df['comment'].isna()]
            sorted_df = df.sort_values(by='points', ascending=False)
            sorted_df['game.id'] = game_id
            logging.info(f"Fetched player statistics for game ID {game_id}")
            return sorted_df
        elif res.status == 429:
            logging.warning(f"Rate limit exceeded for game ID {game_id}. Waiting before retrying.")
            time.sleep(120)  # Wait for 2 minutes before retrying
            return fetch_and_save_player_statistics(game_id, api_key)  # Retry the request
        else:
            logging.error(f"Error: {res.status} {res.reason} for game ID {game_id}")
            return None
    except Exception as e:
        logging.error(f"An error occurred for game ID {game_id}: {e}")
        return None
    finally:
        conn.close()

def get_nba_game_scores(season: int, api_key: str, existing_df=None):
    conn = http.client.HTTPSConnection("api-nba-v1.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "api-nba-v1.p.rapidapi.com"
    }

    conn.request("GET", f"/games?season={season}", headers=headers)
    res = conn.getresponse()

    if res.status != 200:
        logging.error(f"Error: {res.status} {res.reason} while fetching game scores")
        return None
    else:
        data = res.read()
        games = json.loads(data)

        game_data = []
        for game in games.get('response', []):
            game_id = game.get('id')
            game_date = game['date']['start']
            visitor_code = game['teams']['visitors']['code']
            home_code = game['teams']['home']['code']
            visitor_score = game['scores']['visitors']['points']
            home_score = game['scores']['home']['points']
            game_data.append(
                {'game.id': game_id, 'game_date': game_date, 'visitor_score': visitor_score, 'home_score': home_score, 'visitor_code': visitor_code, 'home_code': home_code})

        df = pd.DataFrame(game_data)
        if existing_df is not None:  # Merge with existing data if provided
            df = pd.concat([existing_df, df]).drop_duplicates(subset='game.id', keep='last')
        logging.info(f"Game scores for season {season} processed.")
        return df

def read_processed_games(log_file):
    try:
        with open(log_file, "r") as file:
            processed_games = set(line.strip() for line in file.readlines())
        logging.info(f"Read processed game IDs from {log_file}")
    except FileNotFoundError:
        processed_games = set()
        logging.info(f"{log_file} not found. Starting with an empty set of processed games.")
    return processed_games

def append_to_log_file(log_file, game_id):
    with open(log_file, "a") as file:
        file.write(f"{game_id}\n")
    logging.info(f"Appended game ID {game_id} to {log_file}")

def filter_games_before_today(csv_file):
    try:
        # Load game scores data
        df = pd.read_csv(csv_file)

        # Convert game_date to datetime objects
        df['game_date'] = pd.to_datetime(df['game_date'])

        # Get current date in Pacific Time Zone
        pacific_time = timezone('US/Pacific')
        today = datetime.now(pacific_time).date()

        # Filter out games scheduled after today
        df = df[df['game_date'].dt.date <= today]
        logging.info("Filtered out games scheduled for future dates.")
        return df
    except Exception as e:
        logging.error(f"An error occurred during date filtering: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def main():
    season = 2024
    api_key = "a22eff4319msh7f19aefa5ca6675p1153b3jsn08075099bd47"  # Replace with your actual API key
    output_file = "nba_game_scores.csv"

    # Filter out games after today from existing game scores
    filtered_df = filter_games_before_today(output_file)

    # Proceed with processing the filtered games
    df = get_nba_game_scores(season, api_key, existing_df=filtered_df)

    if df is not None:
        df.to_csv(output_file, index=False)  # Save the combined filtered DataFrame back to the file
        logging.info(f"Filtered game data saved to {output_file}")

        combined_results = pd.read_csv(player_scores_file)
        request_count = 0
        start_time = time.time()
        processed_games = read_processed_games(LOG_FILE)

        for i, game_id in enumerate(df['game.id']):
            game_id_str = str(game_id)  # Convert game_id to string for comparison
            if game_id_str in processed_games:
                logging.info(f"Game ID {game_id} already processed. Skipping.")
                continue

            if request_count >= REQUESTS_PER_DAY:
                logging.info("Daily request limit reached. Please run the script again tomorrow.")
                break

            if i % REQUESTS_PER_MINUTE == 0 and i != 0:
                elapsed_time = time.time() - start_time
                if elapsed_time < 60:
                    time.sleep(60 - elapsed_time)
                start_time = time.time()

            result_df = fetch_and_save_player_statistics(game_id=game_id, api_key=api_key)

            if result_df is None:
                logging.warning(f"No data returned for game ID {game_id}. Skipping.")
                continue  # Skip if no data was fetched

            # Ensure necessary columns exist in df
            required_columns = ['game.id', 'game_date', 'visitor_score', 'home_score', 'visitor_code', 'home_code']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logging.error(f"Missing required columns in `df`: {missing_columns}")
                continue

            # Merge `result_df` with `df` to update missing columns in `result_df`
            result_df = result_df.merge(
                df[['game.id', 'game_date', 'visitor_score', 'home_score', 'visitor_code', 'home_code']],
                on='game.id',
                how='left'
            )

            # Check if the `combined_results` DataFrame exists before using it
            if not combined_results.empty:
                new_rows = result_df[~result_df['game.id'].isin(combined_results['game.id'])]
                combined_results = pd.concat([combined_results, new_rows], ignore_index=True)
            else:
                combined_results = result_df

            # Save combined results to the file after processing each game
            combined_results.to_csv(player_scores_file, index=False)
            logging.info(f"Appended statistics for game ID {game_id} to {player_scores_file}")

            append_to_log_file(LOG_FILE, game_id)
            request_count += 1

        if not combined_results.empty:
            combined_results.to_csv("player_scores.csv", index=False)
            logging.info("Player statistics saved to player_scores.csv")
        else:
            logging.info("No new game data processed today. Please run the script again tomorrow.")

if __name__ == "__main__":
    main()
