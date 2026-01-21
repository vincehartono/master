import random
import numpy as np
import pandas as pd

def make_substitution(active_players, bench, num_subs):
    """
    Perform multiple substitutions: swaps players from the active players with players from the bench.
    
    Args:
        active_players (list): List of currently active players.
        bench (list): List of players on the bench.
        num_subs (int): Number of substitutions to make.
    
    Returns:
        tuple: Updated active players and bench lists.
    """
    if num_subs > len(active_players) or num_subs > len(bench):
        raise ValueError("Number of substitutions cannot exceed the number of players available.")

    # Perform substitutions
    for _ in range(num_subs):
        player_out = random.choice(active_players)
        player_in = random.choice(bench)

        # Perform substitution
        active_players.remove(player_out)
        active_players.append(player_in)
        bench.remove(player_in)
        bench.append(player_out)

    return active_players, bench

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

    # Rename columns and add new columns
    filtered_players.rename(columns={
        'player.name': 'players_name',
        'team.name': 'team',
        'assists_pct_mean': 'assist_probabilities',
        'totReb_pct_mean': 'rebound_probabilities',
        'shot_type_2_mean': 'shot_type_2',
        'shot_type_3_mean': 'shot_type_3',
        'shooting_probabilities_mean': 'shooting_probabilities',
        'ftp_mean': 'shot_success_rates_1',
        'fgp_mean': 'shot_success_rates_2',
        'tpa_pct_mean': 'shot_success_rates_3'
    }, inplace=True)

    filtered_players['foul_threshold'] = 0.85

    # Return DataFrames
    return filtered_game_odds, filtered_players

def assign_assist(players, shooter, assist_probabilities):
    teammates = [player for player in players if player != shooter]
    total_probability = sum(assist_probabilities.get(player, 0) for player in teammates)
    if total_probability == 0:
        return None

    normalized_probabilities = {
        player: assist_probabilities.get(player, 0) / total_probability
        for player in teammates
    }

    random_choice = random.uniform(0, 1)
    cumulative_probability = 0

    for player, probability in normalized_probabilities.items():
        cumulative_probability += probability
        if random_choice <= cumulative_probability:
            return player

    return None

def assign_shot_type(team_weights):
    if not team_weights or sum(team_weights.values()) == 0:
        raise ValueError("Invalid team weights: Ensure the probabilities sum to a positive value.")

    total_weight = sum(team_weights.values())
    normalized_weights = {shot: weight / total_weight for shot, weight in team_weights.items()}

    random_choice = random.uniform(0, 1)
    cumulative_probability = 0

    for shot, probability in normalized_weights.items():
        cumulative_probability += probability
        if random_choice <= cumulative_probability:
            return shot

def pts(team_scoring, points, team1, score_team1, score_team2):
    if points > 0:
        if team_scoring == team1:
            score_team1 += points
        else:
            score_team2 += points

    return score_team1, score_team2

def free_throw(team_scoring, team1, score_team1, score_team2, num_shots, player_free_throw_success_rate):
    successful_free_throws = 0
    for _ in range(num_shots):
        success_rate = random.uniform(0, 1)
        if success_rate <= player_free_throw_success_rate:
            successful_free_throws += 1

    if team_scoring == team1:
        score_team1 += successful_free_throws
    else:
        score_team2 += successful_free_throws

    return score_team1, score_team2, successful_free_throws

def rebound(team_scoring, team1, players_team1, team2, players_team2, rebound_probabilities):
    """
    Determines which team secures the rebound, and selects the specific player.
    :param team_scoring: The team currently possessing the ball (offense).
    :param team1: Name of team 1.
    :param players_team1: List of players in team 1.
    :param team2: Name of team 2.
    :param players_team2: List of players in team 2.
    :param rebound_probabilities: Dictionary containing rebound probabilities for all players.
                                  Each player's value is a combined probability.
    :return: Tuple containing the rebounder's team and player's name.
    """
    # Assign offensive/defensive players and teams
    if team_scoring == team1:
        offensive_team = team1
        offensive_players = players_team1
        defensive_team = team2
        defensive_players = players_team2
    else:
        offensive_team = team2
        offensive_players = players_team2
        defensive_team = team1
        defensive_players = players_team1

    # Get rebound weights for offensive and defensive players
    offensive_rebound_weights = [
        rebound_probabilities.get(player, 0.5) for player in offensive_players
    ]
    defensive_rebound_weights = [
        rebound_probabilities.get(player, 0.5) for player in defensive_players
    ]

    # Ensure valid weights (sum cannot be zero)
    total_weights = sum(offensive_rebound_weights) + sum(defensive_rebound_weights)
    if total_weights == 0:
        raise ValueError("Rebound weights sum to zero. Check rebound_probabilities data.")

    # Compute team weights for determining which team gets the rebound
    team_weights = [
        sum(offensive_rebound_weights) / total_weights,
        sum(defensive_rebound_weights) / total_weights,
    ]
    rebound_team = random.choices([offensive_team, defensive_team], weights=team_weights, k=1)[0]

    # Select rebounder from the appropriate team
    if rebound_team == offensive_team:
        rebounder = random.choices(offensive_players, weights=offensive_rebound_weights, k=1)[0]
    else:
        rebounder = random.choices(defensive_players, weights=defensive_rebound_weights, k=1)[0]

    return rebound_team, rebounder

def simulate_game_with_probabilities(team1, 
                                     team2, 
                                     players_team1, 
                                     players_team2, 
                                     shooting_probabilities_team1, 
                                     shooting_probabilities_team2, 
                                     shot_type_team1,
                                     shot_type_team2,
                                     shot_success_rates_team1, 
                                     shot_success_rates_team2, 
                                     assist_probabilities_team1, 
                                     assist_probabilities_team2,
                                     rebound_probabilities_team1,
                                     rebound_probabilities_team2,
                                     foul_threshold_team1,
                                     foul_threshold_team2, 
                                     total_game_seconds=48*60,
                                     substitution_interval=300):
    score_team1 = 0
    score_team2 = 0
    play_by_play = []

    # Initialize active players (starters) and bench
    active_team1 = players_team1[:5]
    bench_team1 = players_team1[5:]
    active_team2 = players_team2[:5]
    bench_team2 = players_team2[5:]

    # Initialize player statistics
    player_stats = {
        player: {"points": 0, "assists": 0, "rebounds": 0} 
        for player in players_team1 + players_team2
    }

    current_possession = team1
    elapsed_seconds = 0

    while elapsed_seconds < total_game_seconds:
        # Perform substitutions at defined intervals
        if elapsed_seconds % substitution_interval == 0:
            active_team1, bench_team1 = make_substitution(active_team1, bench_team1, int(random.uniform(0, 5)))
            active_team2, bench_team2 = make_substitution(active_team2, bench_team2, int(random.uniform(0, 5)))

        # Determine the team with possession and set the appropriate variables
        if current_possession == team1:
            team_scoring = team1
            players = active_team1
            defending_players = active_team2
            shooting_probabilities = shooting_probabilities_team1
            shot_type_team = shot_type_team1
            shot_success_rates = shot_success_rates_team1
            assist_probabilities = assist_probabilities_team1
            foul_threshold = foul_threshold_team1
            rebound_probabilities = rebound_probabilities_team1
        else:
            team_scoring = team2
            players = active_team2
            defending_players = active_team1
            shooting_probabilities = shooting_probabilities_team2
            shot_type_team = shot_type_team2
            shot_success_rates = shot_success_rates_team2
            assist_probabilities = assist_probabilities_team2
            foul_threshold = foul_threshold_team2
            rebound_probabilities = rebound_probabilities_team2

        # Select a shooter and simulate a shot
        shooter = random.choices(players, weights=[shooting_probabilities[player] for player in players], k=1)[0]
        shot_type = assign_shot_type(shot_type_team)
        success_chance = shot_success_rates[shooter][str(shot_type)] * 100
        is_success = random.uniform(0, 100) <= success_chance
        foul_occurred = random.uniform(0, 1) > foul_threshold

        # Determine possession time
        possession_time = int(np.random.normal(loc=16, scale=5))
        possession_time = max(5, min(24, possession_time)) 
        elapsed_seconds += possession_time
        if elapsed_seconds > total_game_seconds:
            break

        # Calculate points for a successful shot or handle a foul
        points = int(shot_type) if is_success else 0

        if foul_occurred:
            if is_success:
                # And-one scenario
                score_team1, score_team2 = pts(team_scoring, points, team1, score_team1, score_team2)
                player_stats[shooter]["points"] += points
                score_team1, score_team2, made_free_throws = free_throw(
                    team_scoring, team1, score_team1, score_team2, 1, shot_success_rates[shooter]["1"]
                )
                player_stats[shooter]["points"] += made_free_throws
                play_by_play.append(f"Minute {elapsed_seconds // 60}: {shooter} of {team_scoring} made a {shot_type}-point shot with a foul and scored 1 free throw.")
            else:
                # Foul on a missed shot
                free_throws = 3 if shot_type == "3" else 2
                score_team1, score_team2, made_free_throws = free_throw(
                    team_scoring, team1, score_team1, score_team2, free_throws, shot_success_rates[shooter]["1"]
                )
                player_stats[shooter]["points"] += made_free_throws
                play_by_play.append(f"Minute {elapsed_seconds // 60}: {shooter} of {team_scoring} was fouled on a {shot_type}-point attempt and scored {made_free_throws}/{free_throws} free throws.")
        else:
            score_team1, score_team2 = pts(team_scoring, points, team1, score_team1, score_team2)
            player_stats[shooter]["points"] += points
            if is_success:
                assistant = assign_assist(players, shooter, assist_probabilities)
                if assistant:
                    player_stats[assistant]["assists"] += 1
            play_by_play.append(f"Minute {elapsed_seconds // 60}: {shooter} of {team_scoring} {'made' if is_success else 'missed'} a {shot_type}-point shot.")

        # Handle rebounds if the shot is missed
        if not is_success and not foul_occurred:
            rebound_team, rebounder = rebound(
                team_scoring, team1, active_team1, team2, active_team2, rebound_probabilities
            )
            player_stats[rebounder]["rebounds"] += 1
            current_possession = rebound_team
            play_by_play.append(f"Rebound by {rebound_team}! Player: {rebounder}.")
        else:
            current_possession = team1 if team_scoring == team2 else team2

    return {
        "final_scores": {team1: score_team1, team2: score_team2},
        "play_by_play": play_by_play,
        "player_stats": player_stats
    }

def main():
    # Process the data to filter game odds and calculate player statistics
    filtered_game_odds, processed_player_scores = process_and_filter_data(
        'player_scores.csv', 
        'game_odds_today.csv'
    )
    
    # Save the processed data for inspection
    filtered_game_odds.to_csv("filtered_game_odds.csv", index=False)
    processed_player_scores.to_csv("processed_player_scores.csv", index=False)
    
    # Prepare the team configurations for simulation
    data = processed_player_scores
    team_groups = data.groupby("team")
    teams_config = {}

    for team, group in team_groups:
        players = group["players_name"].tolist()
        shot_types = {
            "2": group["shot_type_2"].iloc[0],
            "3": group["shot_type_3"].iloc[0]
        }
        foul_threshold = group["foul_threshold"].iloc[0]
        shooting_probs = dict(zip(group["players_name"], group["shooting_probabilities"]))
        shot_success_rates = {
            player: {
                "1": row["shot_success_rates_1"],
                "2": row["shot_success_rates_2"],
                "3": row["shot_success_rates_3"]
            }
            for _, row in group.iterrows()
            for player in [row["players_name"]]
        }
        assist_probs = dict(zip(group["players_name"], group["assist_probabilities"]))
        rebound_probs = dict(zip(group["players_name"], group["rebound_probabilities"]))

        teams_config[team] = {
            "players": players,
            "shot_types": shot_types,
            "foul_threshold": foul_threshold,
            "shooting_probabilities": shooting_probs,
            "shot_success_rates": shot_success_rates,
            "assist_probabilities": assist_probs,
            "rebound_probabilities": rebound_probs
        }

    # Simulate games for each matchup in filtered_game_odds
    matchups = filtered_game_odds[['team1', 'team2']]
    all_player_stats = []

    for _, matchup in matchups.iterrows():
        team1 = matchup['team1']
        team2 = matchup['team2']

        team1_config = teams_config.get(team1, {})
        team2_config = teams_config.get(team2, {})

        # Skip simulation if team data is missing
        if not team1_config or not team2_config:
            continue

        result = simulate_game_with_probabilities(
            team1=team1,
            team2=team2,
            players_team1=team1_config["players"],
            players_team2=team2_config["players"],
            shooting_probabilities_team1=team1_config["shooting_probabilities"],
            shooting_probabilities_team2=team2_config["shooting_probabilities"],
            shot_type_team1=team1_config["shot_types"],
            shot_type_team2=team2_config["shot_types"],
            shot_success_rates_team1=team1_config["shot_success_rates"],
            shot_success_rates_team2=team2_config["shot_success_rates"],
            assist_probabilities_team1=team1_config["assist_probabilities"],
            assist_probabilities_team2=team2_config["assist_probabilities"],
            rebound_probabilities_team1=team1_config["rebound_probabilities"],
            rebound_probabilities_team2=team2_config["rebound_probabilities"],
            foul_threshold_team1=team1_config["foul_threshold"],
            foul_threshold_team2=team2_config["foul_threshold"],
            substitution_interval=300  # Substitutions every 5 minutes (300 seconds)
        )

        player_stats = result["player_stats"]
        for player, stats in player_stats.items():
            stats["team1"] = team1
            stats["team2"] = team2
            stats["player"] = player
            all_player_stats.append(stats)

    # Save all player stats to a single CSV file
    all_player_stats_df = pd.DataFrame(all_player_stats)
    all_player_stats_df.to_csv("all_player_stats.csv", index=False)

    print("All player stats have been saved to 'all_player_stats.csv'.")

if __name__ == "__main__":
    main()
