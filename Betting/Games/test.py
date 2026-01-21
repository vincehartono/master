from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd

# Find the player ID for a given player name
player_dict = players.get_players()
player_name = "LeBron James"
player = [player for player in player_dict if player['full_name'] == player_name][0]
player_id = player['id']

# Fetch game log for the 2024-25 season
gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2024-25')
gamelog_df = gamelog.get_data_frames()[0]

# Print the game log
print(gamelog_df.head())
