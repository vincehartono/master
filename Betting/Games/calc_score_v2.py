from nba_api.stats.endpoints import playergamelog, scoreboard
import pandas as pd
from datetime import datetime
import time
import requests

# Dictionary of player IDs
player_ids = {
    "Nikola Jokic": 203999,
    "Giannis Antetokounmpo": 203507,
    "Shai Gilgeous-Alexander": 1628983,
    "Luka Doncic": 1629029,
    "Anthony Davis": 203076,
    "Jayson Tatum": 1628369,
    "Karl-Anthony Towns": 1626157,
    "Paolo Banchero": 1631094,
    "LaMelo Ball": 1630163,
    "Domantas Sabonis": 1627734,
    "Victor Wembanyama": 5104157,
    "Cade Cunningham": 1630595,
    "LeBron James": 2544,
    "Damian Lillard": 203081,
    "De'Aaron Fox": 1628368,
    "Nikola Vucevic": 202696,
    "Kevin Durant": 201142,
    "Jalen Brunson": 1628973,
    "Jalen Johnson": 1630552,
    "Franz Wagner": 1630532,
    "Scottie Barnes": 1630567,
    "James Harden": 201935,
    "Stephen Curry": 201939,
    "Kyrie Irving": 202681,
    "Devin Booker": 1626164,
    "Jalen Williams": 1631114,
    "Trae Young": 1629027,
    "Anthony Edwards": 1630162,
    "Alperen Sengun": 1630578,
    "Jaylen Brown": 1627759,
    "RJ Barrett": 1629628,
    "Tyler Herro": 1629639,
    "Donovan Mitchell": 1628378,
    "Brandon Ingram": 1627742,
    "Zion Williamson": 1629627,
    "Ja Morant": 1629630,
    "Tyrese Maxey": 1630178,
    "DeMar DeRozan": 201942,
    "Kristaps Porzingis": 204001,
    "Evan Mobley": 1630596,
    "Jakob Poeltl": 1627751,
    "Darius Garland": 1629636,
    "Pascal Siakam": 1627783,
    "Julius Randle": 203944,
    "Jaren Jackson Jr.": 1628991,
    "Zach LaVine": 203897,
    "Josh Hart": 1628404,
    "Tyrese Haliburton": 1630169,
    "Norman Powell": 1627816,
    "Cam Thomas": 1630556
}

def get_player_game_stats(player_name, season='2024-25', retries=3):
    player_id = player_ids.get(player_name)
    if not player_id:
        return pd.DataFrame()  # Return empty DataFrame if player ID is not found

    for attempt in range(retries):
        try:
            # Fetch game logs
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            game_log_df = game_log.get_data_frames()[0]

            # Format the DataFrame to show only relevant stats
            relevant_stats = game_log_df[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']]
            relevant_stats.insert(0, 'Player', player_name)  # Add player name column

            # Filter out games where PTS, REB, and AST are all zero
            relevant_stats = relevant_stats[(relevant_stats['PTS'] != 0) | (relevant_stats['REB'] != 0) | (relevant_stats['AST'] != 0)]

            return relevant_stats

        except requests.exceptions.ReadTimeout:
            if attempt < retries - 1:
                print(f"ReadTimeoutError: Retrying ({attempt + 1}/{retries})...")
                time.sleep(1)  # Wait a moment before retrying
            else:
                print(f"Failed to retrieve data for {player_name} after {retries} attempts.")
                return pd.DataFrame()

def get_todays_games():
    today = datetime.today().strftime('%Y-%m-%d')
    scoreboard_today = scoreboard.Scoreboard(game_date=today)
    games = scoreboard_today.get_data_frames()[0]
    return games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]

# List of player names
player_names = [
    "Nikola Jokic", "Giannis Antetokounmpo", "Shai Gilgeous-Alexander", "Luka Doncic", "Anthony Davis",
    "Jayson Tatum", "Karl-Anthony Towns", "Paolo Banchero", "LaMelo Ball", "Domantas Sabonis",
    "Victor Wembanyama", "Cade Cunningham", "LeBron James", "Damian Lillard", "De'Aaron Fox",
    "Nikola Vucevic", "Kevin Durant", "Jalen Brunson", "Jalen Johnson", "Franz Wagner",
    "Scottie Barnes", "James Harden", "Stephen Curry", "Kyrie Irving", "Devin Booker",
    "Jalen Williams", "Trae Young", "Anthony Edwards", "Alperen Sengun", "Jaylen Brown",
    "RJ Barrett", "Tyler Herro", "Donovan Mitchell", "Brandon Ingram", "Zion Williamson",
    "Ja Morant", "Tyrese Maxey", "DeMar DeRozan", "Kristaps Porzingis", "Evan Mobley",
    "Jakob Poeltl", "Darius Garland", "Pascal Siakam", "Julius Randle", "Jaren Jackson Jr.",
    "Zach LaVine", "Josh Hart", "Tyrese Haliburton", "Norman Powell", "Cam Thomas"
]

# Create an empty DataFrame to store all players' stats
all_players_stats = pd.DataFrame()

for player_name in player_names:
    player_stats = get_player_game_stats(player_name)
    all_players_stats = pd.concat([all_players_stats, player_stats], ignore_index=True)

# Fetch today's games
todays_games = get_todays_games()
todays_games_teams = pd.concat([todays_games['HOME_TEAM_ID'], todays_games['VISITOR_TEAM_ID']])

# Create a separate DataFrame for players playing today
players_playing_today = all_players_stats[all_players_stats['Player'].isin(todays_games_teams)]

print("All Players' Stats:")
print(all_players_stats)

print("\nPlayers Playing Today:")
print(players_playing_today)
