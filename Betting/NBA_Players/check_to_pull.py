import pandas as pd
import logging

LOG_FILE = "processed_games.log"
PLAYER_SCORES_FILE = "player_scores.csv"

# Set up logging
logging.basicConfig(
    filename='check_to_pull.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def update_processed_games_log(player_scores_file, log_file):
    try:
        df = pd.read_csv(player_scores_file)
        game_ids = df['game.id'].unique()

        with open(log_file, "w") as file:
            for game_id in game_ids:
                file.write(f"{game_id}\n")

        logging.info(f"Processed games log updated with {len(game_ids)} game IDs from {player_scores_file}.")

        print(f"Processed games log updated with {len(game_ids)} game IDs.")
    except FileNotFoundError:
        logging.error(f"File {player_scores_file} not found.")
        print(f"File {player_scores_file} not found.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")


# Update the log file
update_processed_games_log(PLAYER_SCORES_FILE, LOG_FILE)
