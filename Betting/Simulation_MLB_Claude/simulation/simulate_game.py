from baseball_analytics.simulation.player_models import BatterModel, PitcherModel
from baseball_analytics.simulation.game_events import generate_at_bat_result

class GameSimulator:
    def __init__(self, home_team_data, away_team_data, n_innings=9):
        self.home_team = self._prepare_team(home_team_data, is_home=True)
        self.away_team = self._prepare_team(away_team_data, is_home=False)
        self.n_innings = n_innings
        self.reset_game()
        
    def _prepare_team(self, team_data, is_home):
        # Process team data to create lineup, pitching rotation, etc.
        return {
            'name': team_data['name'],
            'lineup': [BatterModel(player) for player in team_data['lineup']],
            'pitchers': [PitcherModel(player) for player in team_data['pitchers']],
            'is_home': is_home,
            'current_batter_idx': 0,
            'current_pitcher_idx': 0
        }
        
    def reset_game(self):
        self.inning = 1
        self.half_inning = 'top'  # 'top' or 'bottom'
        self.outs = 0
        self.home_score = 0
        self.away_score = 0
        self.bases = [False, False, False]  # First, second, third
        
    def simulate_full_game(self):
        game_log = []
        
        while not self._is_game_over():
            event = self.simulate_at_bat()
            game_log.append(event)
            
        return {
            'home_score': self.home_score,
            'away_score': self.away_score,
            'game_log': game_log
        }
    
    def simulate_at_bat(self):
        # Get current batter and pitcher
        batting_team = self.away_team if self.half_inning == 'top' else self.home_team
        fielding_team = self.home_team if self.half_inning == 'top' else self.away_team
        
        batter = batting_team['lineup'][batting_team['current_batter_idx']]
        pitcher = fielding_team['pitchers'][fielding_team['current_pitcher_idx']]
        
        # Simulate the at-bat result
        result = generate_at_bat_result(batter, pitcher)
        
        # Process the result and update game state
        self._process_at_bat_result(result, batting_team)
        
        # Advance to next batter
        batting_team['current_batter_idx'] = (batting_team['current_batter_idx'] + 1) % len(batting_team['lineup'])
        
        return {
            'inning': self.inning,
            'half': self.half_inning,
            'batter': batter.name,
            'pitcher': pitcher.name,
            'result': result,
            'home_score': self.home_score,
            'away_score': self.away_score
        }
    
    def _process_at_bat_result(self, result, batting_team):
        # Logic to process the result (hit, walk, out, etc.)
        # Update bases, score, outs based on the result
        
        # Example implementation:
        if result == 'out':
            self.outs += 1
            if self.outs >= 3:
                self._advance_inning()
        elif result == 'single':
            # Move runners and possibly score
            self._handle_hit(1)
        # ... other results
    
    def _advance_inning(self):
        self.outs = 0
        self.bases = [False, False, False]
        
        if self.half_inning == 'top':
            self.half_inning = 'bottom'
        else:
            self.half_inning = 'top'
            self.inning += 1
    
    def _is_game_over(self):
        # Check if the game is over based on innings and score
        if self.inning > self.n_innings:
            return True
        if self.inning == self.n_innings and self.half_inning == 'bottom':
            if self.home_score > self.away_score:
                return True
        return False
    
    # Additional helper methods...