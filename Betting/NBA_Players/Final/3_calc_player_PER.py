import pandas as pd

# Load the player scores CSV file
player_scores = pd.read_csv('player_scores.csv')

# Step 1: Combine player.firstname and player.lastname
player_scores['player_name'] = player_scores['player.firstname'] + ' ' + player_scores['player.lastname']

# Step 2: Compute basic stats for PER calculation
player_scores['fg_misses'] = player_scores['fga'] - player_scores['fgm']
player_scores['ft_misses'] = player_scores['fta'] - player_scores['ftm']
player_scores['scoring'] = player_scores['fgm'] * 2 + player_scores['ftm']

player_scores['base_value'] = (
    player_scores['scoring'] +
    player_scores['assists'] * 1.5 +
    player_scores['totReb'] * 1.3 +
    player_scores['steals'] * 2 +
    player_scores['blocks'] * 2 -
    player_scores['turnovers'] * 2 -
    (player_scores['fg_misses'] * 0.7 + player_scores['ft_misses'] * 0.6)
)

# Step 3: Calculate unadjusted PER per minute
player_scores['uPER'] = player_scores['base_value'] / player_scores['min']

# Step 4: Calculate league pace factor
league_totals = player_scores[['scoring', 'totReb', 'assists', 'steals', 'blocks', 'turnovers', 'fg_misses', 'ft_misses', 'min']].sum()
pace_factor = league_totals['min'] / len(player_scores)

# Step 5: Normalize PER by league pace and set average PER to 15
player_scores['normalized_PER'] = player_scores['uPER'] * (15 / pace_factor)

# Step 6: Filter players with minutes played less than 5 to avoid inflated values
filtered_players = player_scores[player_scores['min'] >= 15].copy()

# Step 7: Determine if the player is "home" or "visitor"
filtered_players['location'] = filtered_players.apply(
    lambda x: 'visitor' if x['team.code'] == x['visitor_code'] else 'home', axis=1
)

# Step 8: Save PER data for all players to a CSV, excluding visitor_code and home_code
output_file_path = 'player_PER_scores.csv'
filtered_players[['player_name', 'team.code', 'min', 'points', 'totReb', 'assists', 'uPER', 'normalized_PER', 'location']].to_csv(output_file_path, index=False)

print(f"Player PER scores saved to {output_file_path}")
