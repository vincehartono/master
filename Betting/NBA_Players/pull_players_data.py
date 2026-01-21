import http.client
import json
import pandas as pd
import time
import logging
from datetime import datetime
from pytz import timezone

REQUESTS_PER_MINUTE = 10
REQUESTS_PER_DAY = 20
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

            # Check if 'team.logo' column exists before attempting to drop
            if 'team.logo' in df.columns:
                df = df.drop(columns=['team.logo'])

            df = df[df['comment'].isna()]  # Filter rows without comments
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
        if str(e).find("['team.logo'] not found in axis") != -1:
            logging.error("Critical column missing. Terminating script to avoid further errors.")
            quit()  # Terminate script immediately on this specific error
        return None

    finally:
        conn.close()

def get_nba_game_scores(season: int, api_key: str):
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

        # Save the game scores to CSV
        df.to_csv("nba_game_scores.csv", index=False)
        logging.info(f"Game scores for season {season} saved to 'nba_game_scores.csv'.")

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
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')  # Handle invalid dates gracefully

        # Drop rows with invalid or missing dates
        df = df.dropna(subset=['game_date'])

        # Ensure `game_date` is timezone-aware and convert to Pacific Time Zone
        pacific_time = timezone('US/Pacific')
        df['game_date_pacific'] = df['game_date'].apply(
            lambda x: x.tz_convert(pacific_time) if x.tzinfo else x.tz_localize('UTC').tz_convert(pacific_time)
        )

        # Remove rows where visitor_score is empty or missing
        df = df.dropna(subset=['visitor_score'])

        # Get today's date in Pacific Time
        today_pacific = datetime.now(pacific_time).date()

        # Filter out games scheduled on or after today in Pacific Time
        df_filtered = df[df['game_date_pacific'].dt.date < today_pacific]

        # Output the filtered DataFrame to CSV for inspection
        df_filtered.to_csv("filtered_game_scores.csv", index=False)
        logging.info(f"Filtered game data saved to 'filtered_game_scores.csv'")

        return df_filtered

    except Exception as e:
        logging.error(f"An error occurred during date filtering: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure


def main():
    season = 2024
    api_key = "a22eff4319msh7f19aefa5ca6675p1153b3jsn08075099bd47"  # Replace with your actual API key
    output_file = "nba_game_scores.csv"

    # Step 1: Fetch and save game scores to CSV
    df = get_nba_game_scores(season, api_key)

    if df is not None:
        # Step 2: Filter out games after today
        filtered_df = filter_games_before_today(output_file)

        # Step 3: Process player statistics based on the filtered data
        combined_results = pd.read_csv(player_scores_file)
        request_count = 0
        start_time = time.time()
        processed_games = read_processed_games(LOG_FILE)

        for i, game_id in enumerate(filtered_df['game.id']):
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

            # Save combined results to the file after processing each game
            if not combined_results.empty:
                new_rows = result_df[~result_df['game.id'].isin(combined_results['game.id'])]
                combined_results = pd.concat([combined_results, new_rows], ignore_index=True)
            else:
                combined_results = result_df

            # Save the updated combined results to CSV
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
