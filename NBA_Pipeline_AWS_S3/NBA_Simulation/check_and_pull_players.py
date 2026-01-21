import os
import pandas as pd
import logging
import http.client
import json
import time
from datetime import datetime, date
from typing import Optional
from pytz import timezone

from NBA_Simulation import s3_io

# Constants
LOG_KEY = "processed_games.log"
PLAYER_SCORES_KEY = "player_scores.csv"
NBA_GAME_SCORES_KEY = "nba_game_scores.csv"
FILTERED_GAME_SCORES_KEY = "filtered_game_scores.csv"

# Set up logging
logging.basicConfig(
    filename='NBA_Simulation/check_and_pull_players.log',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def update_processed_games_log():
    try:
        df = s3_io.read_csv(PLAYER_SCORES_KEY)
        game_ids = df['game.id'].unique()

        lines = [f"{game_id}\n" for game_id in game_ids]
        s3_io.write_text(LOG_KEY, lines)

        print(f"Processed games log updated with {len(game_ids)} game IDs.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        print(f"An error occurred: {e}")

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

            if 'team.logo' in df.columns:
                df = df.drop(columns=['team.logo'])

            # Filter rows without comments when the column exists
            if 'comment' in df.columns:
                df = df[df['comment'].isna()]
            sorted_df = df.sort_values(by='points', ascending=False)
            sorted_df['game.id'] = game_id
            logging.info(f"Fetched player statistics for game ID {game_id}")
            return sorted_df

        elif res.status == 429:
            logging.warning(f"Rate limit exceeded for game ID {game_id}. Skipping without retry.")
            return None

        else:
            logging.error(f"Error: {res.status} {res.reason} for game ID {game_id}")
            return None

    except Exception as e:
        logging.error(f"An error occurred for game ID {game_id}: {e}")
        return None

    finally:
        conn.close()

def get_nba_game_scores(season: int, api_key: str, allow_fallback: bool = True):
    """Fetch all games for a season (single request) and merge with existing file.

    Uses `league=standard`. Removes unsupported paging param that returned
    "The Page field do not exist." errors in logs.
    Falls back to previous season if zero games are returned.
    """
    headers = {
        'x-rapidapi-key': api_key,
        'x-rapidapi-host': "api-nba-v1.p.rapidapi.com"
    }

    def _fetch_all_once(season_param: int):
        conn = http.client.HTTPSConnection("api-nba-v1.p.rapidapi.com")
        try:
            path = f"/games?league=standard&season={season_param}"
            logging.debug(f"Requesting path: {path}")
            conn.request("GET", path, headers=headers)
            res = conn.getresponse()
            if res.status != 200:
                raw = res.read()
                snippet = raw[:500].decode('utf-8', errors='replace') if raw else ''
                logging.error(f"Error: {res.status} {res.reason} while fetching game scores (season {season_param}). Body snippet: {snippet}")
                return []
            data = res.read()
            games = json.loads(data)
            response_games = games.get('response', [])
            if not response_games:
                snippet = data[:500].decode('utf-8', errors='replace') if isinstance(data, (bytes, bytearray)) else str(data)[:500]
                logging.warning(f"Empty response list for season {season_param}. Raw body snippet: {snippet}")
            all_games_local = []
            for game in response_games:
                game_id = game.get('id')
                game_date = game.get('date', {}).get('start') if game.get('date') else None
                visitor_code = game.get('teams', {}).get('visitors', {}).get('code') if game.get('teams') else None
                home_code = game.get('teams', {}).get('home', {}).get('code') if game.get('teams') else None
                visitor_score = game.get('scores', {}).get('visitors', {}).get('points') if game.get('scores') else None
                home_score = game.get('scores', {}).get('home', {}).get('points') if game.get('scores') else None
                all_games_local.append({
                    'game.id': game_id,
                    'game_date': game_date,
                    'visitor_score': visitor_score,
                    'home_score': home_score,
                    'visitor_code': visitor_code,
                    'home_code': home_code
                })
            logging.info(f"Fetched {len(all_games_local)} games for season {season_param}")
            return all_games_local
        finally:
            conn.close()

    # Primary fetch
    logging.info(f"Starting game fetch for season {season} (league=standard)")
    all_games = _fetch_all_once(season)

    # Fallback: if nothing returned, try previous season (unless disabled)
    if not all_games and allow_fallback:
        logging.warning(f"No games returned for season {season}. Retrying with season {season-1} in case of API season offset.")
        all_games = _fetch_all_once(season-1)
        if all_games:
            logging.info(f"Fetched {len(all_games)} games using fallback season {season-1}.")
        else:
            logging.error(f"Still zero games after fallback to season {season-1}. Check API key/plan or endpoint contract.")

    df = pd.DataFrame(all_games)

    # If we already have a file, merge (append new/updated) instead of full refresh
    try:
        existing = s3_io.read_csv(NBA_GAME_SCORES_KEY)
        merged = pd.concat([existing, df], ignore_index=True)
        if 'game.id' in merged.columns:
            merged = merged.drop_duplicates(subset=['game.id'], keep='last')
        df = merged
    except Exception:
        pass

    s3_io.write_csv(df, NBA_GAME_SCORES_KEY)
    logging.info(f"Game scores saved to S3 key '{NBA_GAME_SCORES_KEY}'. Rows={len(df)} (queried season {season})")

    return df

def read_processed_games():
    lines = s3_io.read_text_lines(LOG_KEY)
    processed_games = set(line.strip() for line in lines)
    return processed_games

def append_to_log_file(log_file, game_id):
    s3_io.append_text_line(LOG_KEY, f"{game_id}\n")
    logging.info(f"Appended game ID {game_id} to {log_file}")

def filter_games_before_today(csv_file):
    try:
        df = s3_io.read_csv(NBA_GAME_SCORES_KEY)
        df['game_date'] = pd.to_datetime(df['game_date'], errors='coerce')
        df = df.dropna(subset=['game_date'])

        pacific_time = timezone('US/Pacific')
        df['game_date_pacific'] = df['game_date'].apply(
            lambda x: x.tz_convert(pacific_time) if x.tzinfo else x.tz_localize('UTC').tz_convert(pacific_time)
        )

        # Ensure scores are numeric; empty strings -> NaN
        df['visitor_score'] = pd.to_numeric(df['visitor_score'], errors='coerce')
        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df = df.dropna(subset=['visitor_score', 'home_score'])

        today_pacific = datetime.now(pacific_time).date()

        # Include all games up to and including today in Pacific time
        df_filtered = df[df['game_date_pacific'].dt.date <= today_pacific]
        s3_io.write_csv(df_filtered, FILTERED_GAME_SCORES_KEY)
        logging.info(f"Filtered game data saved to S3 key '{FILTERED_GAME_SCORES_KEY}'")

        return df_filtered

    except Exception as e:
        logging.error(f"An error occurred during date filtering: {e}")
        return pd.DataFrame()

def _current_nba_season(today: Optional[date] = None) -> int:
    """Return the NBA season year per API-NBA convention (e.g., 2024 for the 2024-25 season)."""
    d = today or date.today()
    # NBA season typically starts in October; API-NBA uses the starting year
    return d.year if d.month >= 10 else d.year - 1


def main():
    # Allow forcing a specific season via env var, e.g. NBA_SEASON_FORCE=2025
    season_override = os.environ.get("NBA_SEASON_FORCE")
    if season_override:
        try:
            season = int(season_override)
            logging.info(f"Using forced NBA season from NBA_SEASON_FORCE={season}")
        except ValueError:
            logging.warning(f"Invalid NBA_SEASON_FORCE value '{season_override}', falling back to computed season.")
            season = _current_nba_season()
            season_override = None
    else:
        season = _current_nba_season()

    try:
        api_key = os.environ["RAPIDAPI_KEY"]
    except KeyError:
        # Fallback: try loading from local .env-style file if present
        env_path = os.path.join(os.path.dirname(__file__), "RAPIDAPI_KEY.env")
        api_key = None
        if os.path.exists(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    first = f.read().strip()
                # Support either a bare key or KEY=VALUE format
                if "=" in first:
                    _, val = first.split("=", 1)
                    api_key = val.strip()
                else:
                    api_key = first
                if not api_key:
                    raise ValueError("Empty RAPIDAPI_KEY in RAPIDAPI_KEY.env")
                logging.info(f"Loaded RAPIDAPI_KEY from {env_path}")
            except Exception as e:
                logging.error(f"Failed to load RAPIDAPI_KEY from {env_path}: {e}")
                api_key = None
        if not api_key:
            logging.error("RAPIDAPI_KEY environment variable is not set and RAPIDAPI_KEY.env could not be used.")
            print("RAPIDAPI_KEY is not configured. Set the RAPIDAPI_KEY environment variable or add it to RAPIDAPI_KEY.env.")
            return

    # Always refresh the game list (function merges with existing file)
    # If a season was explicitly forced, disable automatic fallback to season-1
    df = get_nba_game_scores(season, api_key, allow_fallback=not bool(season_override))

    if df is not None:
        # Sync processed log with any games already present in the player_scores file
        try:
            update_processed_games_log()
        except Exception:
            pass

        filtered_df = filter_games_before_today(NBA_GAME_SCORES_KEY)

        combined_results = pd.DataFrame()
        processed_games = read_processed_games()

        # Process every played game not in processed log (no rate limiting)
        for i, game_id in enumerate(filtered_df['game.id']):
            game_id_str = str(game_id)
            if game_id_str in processed_games:
                logging.info(f"Game ID {game_id} already processed. Skipping.")
                continue

            result_df = fetch_and_save_player_statistics(game_id=game_id, api_key=api_key)

            if result_df is None:
                logging.warning(f"No data returned for game ID {game_id}. Skipping.")
                continue

            combined_results = pd.concat([combined_results, result_df], ignore_index=True)

            # Write updated player_scores to S3 (no local file appends)
            try:
                existing_players = s3_io.read_csv(PLAYER_SCORES_KEY)
                updated_players = pd.concat([existing_players, result_df], ignore_index=True)
            except Exception:
                updated_players = result_df

            s3_io.write_csv(updated_players, PLAYER_SCORES_KEY)
            logging.info(f"Appended statistics for game ID {game_id} to S3 key {PLAYER_SCORES_KEY}")

            append_to_log_file(LOG_KEY, game_id)

if __name__ == "__main__":
    main()
