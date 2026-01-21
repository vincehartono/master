import pandas as pd

def process_and_filter_data(player_scores_file, game_odds_file):
    """
    Processes game odds and player statistics, filters based on game odds, and returns two DataFrames:
    filtered game odds and processed player scores.

    Args:
        player_scores_file (str): Path to the CSV file containing player scores.
        game_odds_file (str): Path to the CSV file containing game odds.

    Returns:
        tuple: (filtered_game_odds, processed_player_scores) where both are pandas DataFrames.
    """
    # Load player scores
    player_scores = pd.read_csv(player_scores_file)
    player_scores['player.name'] = player_scores['player.firstname'] + ' ' + player_scores['player.lastname']
    player_scores.drop(['player.firstname', 'player.lastname'], axis=1, inplace=True)

    # Step 1: Aggregate team statistics per game
    team_stats = player_scores.groupby(['game.id', 'team.id']).agg({
        'fga': 'sum',
        'tpa': 'sum',
        'offReb': 'sum',
        'defReb': 'sum',
        'totReb': 'sum',
        'assists': 'sum',
        'pFouls': 'sum',
        'steals': 'sum',
        'turnovers': 'sum',
        'blocks': 'sum'
    }).rename(columns={
        'fga': 'team_fga',
        'tpa': 'team_tpa',
        'offReb': 'team_offReb',
        'defReb': 'team_defReb',
        'totReb': 'team_totReb',
        'assists': 'team_assists',
        'pFouls': 'team_pFouls',
        'steals': 'team_steals',
        'turnovers': 'team_turnovers',
        'blocks': 'team_blocks'
    }).reset_index()

    # Step 2: Merge team stats back with player stats
    merged_data = player_scores.merge(team_stats, on=['game.id', 'team.id'])

    # Step 3: Ensure team.name is included for each player
    if 'team.name' not in merged_data.columns:
        team_names = player_scores[['team.id', 'team.name']].drop_duplicates()
        merged_data = merged_data.merge(team_names, on='team.id', how='left')

    # Step 4: Calculate player statistics
    merged_data['fga_pct'] = merged_data['fga'] / merged_data['team_fga']
    merged_data['tpa_pct'] = merged_data['tpa'] / merged_data['team_tpa']
    merged_data['offReb_pct'] = merged_data['offReb'] / merged_data['team_offReb']
    merged_data['defReb_pct'] = merged_data['defReb'] / merged_data['team_defReb']
    merged_data['totReb_pct'] = merged_data['totReb'] / merged_data['team_totReb']
    merged_data['assists_pct'] = merged_data['assists'] / merged_data['team_assists']
    merged_data['pFouls_pct'] = merged_data['pFouls'] / merged_data['team_pFouls']
    merged_data['steals_pct'] = merged_data['steals'] / merged_data['team_steals']
    merged_data['turnovers_pct'] = merged_data['turnovers'] / merged_data['team_turnovers']
    merged_data['blocks_pct'] = merged_data['blocks'] / merged_data['team_blocks']

    merged_data['shot_type_2'] = merged_data['fga'] / (merged_data['fga'] + merged_data['tpa'])
    merged_data['shot_type_3'] = merged_data['tpa'] / (merged_data['fga'] + merged_data['tpa'])
    merged_data['shooting_probabilities'] = (merged_data['fga'] + merged_data['tpa']) / (merged_data['team_fga'] + merged_data['team_tpa'])

    # Step 7: Calculate mean and std for each player
    summary_columns = ['fga_pct', 'fgp', 'points', 'ftp', 'tpa_pct', 'tpp',
                       'offReb_pct', 'defReb_pct', 'totReb_pct', 'assists_pct',
                       'pFouls_pct', 'steals_pct', 'turnovers_pct', 'blocks_pct',
                       'shot_type_2', 'shot_type_3', 'shooting_probabilities']

    summary_stats = (
        merged_data.groupby(['player.name', 'team.name'])[summary_columns]
        .agg(['mean', 'std'])
        .reset_index()
    )

    summary_stats.columns = ['player.name', 'team.name'] + [f'{col[0]}_{col[1]}' for col in summary_stats.columns[2:]]

    # Step 5: Process game odds and filter teams
    df_odds = pd.read_csv(game_odds_file, header=None)
    rows_to_extract = []

    for index, row in df_odds.iterrows():
        if any(cell == 'Spread' for cell in row.values):
            rows_to_extract.extend([
                index - 3 if index - 3 >= 0 else None,
                index - 1 if index - 1 >= 0 else None,
                index + 1 if index + 1 < len(df_odds) else None
            ])
        if any(cell == 'Total' for cell in row.values):
            if index + 1 < len(df_odds):
                rows_to_extract.append(index + 1)

    rows_to_extract = sorted(set(filter(lambda x: x is not None, rows_to_extract)))

    filtered_odds = df_odds.iloc[rows_to_extract]
    filtered_odds.reset_index(drop=True, inplace=True)

    num_rows = len(filtered_odds)
    team1 = filtered_odds.iloc[0:num_rows:4].reset_index(drop=True)
    team2 = filtered_odds.iloc[1:num_rows:4].reset_index(drop=True)
    spread = filtered_odds.iloc[2:num_rows:4].reset_index(drop=True)
    over_under = filtered_odds.iloc[3:num_rows:4].reset_index(drop=True)

    # Clean 'O ' from over_under
    over_under = over_under.applymap(lambda x: x.replace("O ", "") if isinstance(x, str) else x)

    # Combine into a single DataFrame
    filtered_game_odds = pd.concat([team1, team2, spread, over_under], axis=1, ignore_index=True)
    filtered_game_odds.columns = ['team1', 'team2', 'spread', 'over_under']

    # Create team list based on filtered game odds
    team_list = pd.concat([filtered_game_odds['team1'], filtered_game_odds['team2']]).unique()

    # Step 6: Filter players based on the team list
    filtered_players = summary_stats[summary_stats['team.name'].isin(team_list)]

    # Return DataFrames
    return filtered_game_odds, filtered_players

# Paths to input files
player_scores_file = 'player_scores.csv'
game_odds_file = 'game_odds_today.csv'

# Call the function
filtered_game_odds, processed_player_scores = process_and_filter_data(player_scores_file, game_odds_file)

# Display results
print("Filtered Game Odds:")
print(filtered_game_odds.head())

print("\nProcessed Player Scores:")
print(processed_player_scores.head())
processed_player_scores.to_csv("processed_player_scores_new.csv")