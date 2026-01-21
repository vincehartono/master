import random
import numpy as np

def pts(team_scoring, points, player, play_by_play, team1, team2, elapsed_seconds, possession_time, remaining_time, score_team1, score_team2):
    if points == 0:
        shot_type = "missed"
    elif points == 1:
        shot_type = "Free Throw"
        if team_scoring == team1:
            score_team1 += 1
        else:
            score_team2 += 1
    elif points == 2:
        shot_type = "2-pointer"
        if team_scoring == team1:
            score_team1 += 2
        else:
            score_team2 += 2
    elif points == 3:
        shot_type = "3-pointer"
        if team_scoring == team1:
            score_team1 += 3
        else:
            score_team2 += 3

    if points != 0:
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {player} of {team_scoring} scored a {shot_type} of {points} points with {possession_time} seconds lapsed. Remaining time: {remaining_time // 60}m {remaining_time % 60}s. ({team1}: {score_team1}, {team2}: {score_team2})")
    else:
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {player} of {team_scoring} attempted a shot (missed) with {possession_time} seconds lapsed. Remaining time: {remaining_time // 60}m {remaining_time % 60}s. ({team1}: {score_team1}, {team2}: {score_team2})")

    return score_team1, score_team2, play_by_play


def rebound(team_scoring, players, defending_players, elapsed_seconds, possession_time, remaining_time, score_team1, score_team2, play_by_play):
    rebounder = random.choices(players + defending_players, weights=[0.4] * len(players) + [0.4] * len(defending_players), k=1)[0]
    
    if rebounder in players:
        team_rebounding = team_scoring
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {rebounder} of {team_rebounding} grabbed the rebound. Remaining time: {remaining_time // 60}m {remaining_time % 60}s. ({score_team1}: {score_team1}, {score_team2}: {score_team2})")
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {team_rebounding} retains possession. ({score_team1}: {score_team1}, {score_team2}: {score_team2})")
    else:
        team_rebounding = team_scoring
        team_scoring = team2 if team_rebounding == team1 else team1
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {rebounder} of {team_scoring} grabbed the rebound. Remaining time: {remaining_time // 60}m {remaining_time % 60}s. ({score_team1}: {score_team1}, {score_team2}: {score_team2})")
        play_by_play.append(f"Minute {int(elapsed_seconds // 60) + 1}: {team_scoring} now has possession. ({score_team1}: {score_team1}, {score_team2}: {score_team2})")
    
    return team_scoring, play_by_play


def simulate_game_with_probabilities(team1, team2, players_team1, players_team2, shooting_probabilities_team1, shooting_probabilities_team2, shot_success_rates_team1, shot_success_rates_team2, total_game_seconds=48*60):
    """
    Simulates an NBA game and returns the final scores of the two teams and their play-by-play.

    Parameters:
        team1 (str): Name of the first team.
        team2 (str): Name of the second team.
        players_team1 (list): List of players for the first team.
        players_team2 (list): List of players for the second team.
        shooting_probabilities_team1 (dict): Shooting probabilities for team1 players.
        shooting_probabilities_team2 (dict): Shooting probabilities for team2 players.
        shot_success_rates_team1 (dict): Shot success rates for team1 players.
        shot_success_rates_team2 (dict): Shot success rates for team2 players.
        total_game_seconds (int): Total time to simulate in seconds.

    Returns:
        dict: A dictionary containing the scores of both teams and a play-by-play simulation.
    """
    score_team1 = 0
    score_team2 = 0
    play_by_play = []

    # Combine the player probabilities from both teams into one dictionary
    all_shooting_probabilities = {**shooting_probabilities_team1, **shooting_probabilities_team2}
    all_shot_success_rates = {**shot_success_rates_team1, **shot_success_rates_team2}

    # Start with team1 having the ball
    current_possession = team1
    elapsed_seconds = 0

    while elapsed_seconds < total_game_seconds:
        if current_possession == team1:
            team_scoring = team1
            players = players_team1
            defending_players = players_team2
            shooting_probabilities = shooting_probabilities_team1
            shot_success_rates = shot_success_rates_team1
        else:
            team_scoring = team2
            players = players_team2
            defending_players = players_team1
            shooting_probabilities = shooting_probabilities_team2
            shot_success_rates = shot_success_rates_team2

        # Randomly determine which player takes the shot based on their probability
        shooter = random.choices(players, weights=[shooting_probabilities[player] for player in players], k=1)[0]
        
        # Randomly determine the shot type based on success rates for that player
        shot_type = random.choices([1, 2, 3], weights=[shot_success_rates[shooter]["1"], shot_success_rates[shooter]["2"], shot_success_rates[shooter]["3"]], k=1)[0]
        
        if shot_type == 1:
            points = 1  # Free Throw
        elif shot_type == 2:
            points = 2  # 2-pointer
        else:
            points = 3  # 3-pointer

        # Calculate the possession time (randomize between 5 and 24 seconds)
        possession_time = int(np.random.normal(loc=16, scale=5))
        possession_time = max(5, min(24, possession_time)) 

        elapsed_seconds += possession_time
        if elapsed_seconds > total_game_seconds:
            break

        # Calculate remaining time
        remaining_time = total_game_seconds - elapsed_seconds

        # Handle the points scored or missed using the pts function
        score_team1, score_team2, play_by_play = pts(
            team_scoring, points, shooter, play_by_play, team1, team2, elapsed_seconds, possession_time, remaining_time, score_team1, score_team2
        )

        # If the shot missed, we need to handle the rebound and possession
        if points == 0:
            team_scoring, play_by_play = rebound(
                team_scoring, players, defending_players, elapsed_seconds, possession_time, remaining_time, score_team1, score_team2, play_by_play
            )
        else:
            current_possession = team2 if team_scoring == team1 else team1

    return {
        "final_scores": {team1: score_team1, team2: score_team2},
        "play_by_play": play_by_play
    }


# Example usage
team1 = "Lakers"
team2 = "Celtics"
players_team1 = ["LeBron James", "Anthony Davis", "D'Angelo Russell", "Austin Reaves", "Rui Hachimura"]
players_team2 = ["Jayson Tatum", "Marcus Smart", "Jaylen Brown", "Malcolm Brogdon", "Al Horford"]

shooting_probabilities_team1 = {
    "LeBron James": 0.25, 
    "Anthony Davis": 0.3, 
    "D'Angelo Russell": 0.2, 
    "Austin Reaves": 0.15,
    "Rui Hachimura": 0.1
}

shooting_probabilities_team2 = {
    "Jayson Tatum": 0.3, 
    "Marcus Smart": 0.2, 
    "Jaylen Brown": 0.25, 
    "Malcolm Brogdon": 0.15, 
    "Al Horford": 0.1
}

shot_success_rates_team1 = {
    "LeBron James": {"1": 0.75, "2": 0.55, "3": 0.35},  
    "Anthony Davis": {"1": 0.80, "2": 0.65, "3": 0.30},
    "D'Angelo Russell": {"1": 0.85, "2": 0.45, "3": 0.40},
    "Austin Reaves": {"1": 0.90, "2": 0.50, "3": 0.45},
    "Rui Hachimura": {"1": 0.70, "2": 0.60, "3": 0.40}
}

shot_success_rates_team2 = {
    "Jayson Tatum": {"1": 0.75, "2": 0.55, "3": 0.40},
    "Marcus Smart": {"1": 0.80, "2": 0.60, "3": 0.35},
    "Jaylen Brown": {"1": 0.78, "2": 0.52, "3": 0.38},
    "Malcolm Brogdon": {"1": 0.88, "2": 0.60, "3": 0.33},
    "Al Horford": {"1": 0.85, "2": 0.58, "3": 0.30}
}

game_result = simulate_game_with_probabilities(team1, team2, players_team1, players_team2, shooting_probabilities_team1, shooting_probabilities_team2, shot_success_rates_team1, shot_success_rates_team2)

# Print play-by-play
for play in game_result["play_by_play"]:
    print(play)

# Print final score
print(f"\nFinal Score:\n{team1}: {game_result['final_scores'][team1]}\n{team2}: {game_result['final_scores'][team2]}")
