from baseball_analytics.data_loader import load_atbat_data, preprocess_data
from baseball_analytics.stats import calculate_batting_stats, calculate_pitching_stats
from baseball_analytics.simulation.simulate_game import GameSimulator

def main():
    # Load and preprocess data
    data_path = "mlb_atbat_data_sample.csv"
    df = load_atbat_data(data_path)
    df = preprocess_data(df)
    
    # Prepare teams for simulation based on historical data
    home_team_data = prepare_team_for_simulation(df, 'NYY')  # Example team
    away_team_data = prepare_team_for_simulation(df, 'BOS')  # Example team
    
    # Run simulation
    simulator = GameSimulator(home_team_data, away_team_data)
    game_result = simulator.simulate_full_game()
    
    print(f"Final Score: {away_team_data['name']} {game_result['away_score']} - {home_team_data['name']} {game_result['home_score']}")
    
    # Additional analysis and visualization...

def prepare_team_for_simulation(df, team_code):
    """Extract and prepare team data for simulation."""
    team_data = {
        'name': team_code,
        'lineup': [],
        'pitchers': []
    }
    
    # Extract batters for this team
    team_batters = df[df['home_team'] == team_code]['batter'].unique()
    for batter_id in team_batters[:9]:  # Take top 9 batters
        batter_data = calculate_player_stats(df, batter_id, is_pitcher=False)
        team_data['lineup'].append(batter_data)
    
    # Extract pitchers for this team
    team_pitchers = df[df['home_team'] == team_code]['pitcher'].unique()
    for pitcher_id in team_pitchers[:5]:  # Take top 5 pitchers
        pitcher_data = calculate_player_stats(df, pitcher_id, is_pitcher=True)
        team_data['pitchers'].append(pitcher_data)
    
    return team_data

def calculate_player_stats(df, player_id, is_pitcher=False):
    """Calculate relevant stats for a player."""
    # Implementation depends on your stats modules
    # ...