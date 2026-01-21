class PlayerModel:
    def __init__(self, player_data):
        self.name = player_data['player_name']
        self.id = player_data.get('id', 0)
        # Common player attributes
        
class BatterModel(PlayerModel):
    def __init__(self, player_data):
        super().__init__(player_data)
        
        # Calculate batting stats from historical data
        self.avg = player_data.get('avg', 0.250)
        self.obp = player_data.get('obp', 0.320)
        self.slg = player_data.get('slg', 0.400)
        self.hr_rate = player_data.get('hr_rate', 0.03)
        # Other relevant stats
        
    def get_matchup_adjustments(self, pitcher):
        """Calculate adjustments based on batter vs pitcher matchup."""
        # Example: adjust for left/right matchups
        if pitcher.throws == 'L' and self.bats == 'R':
            return {'avg_modifier': 0.02}  # Slight advantage
        return {}

class PitcherModel(PlayerModel):
    def __init__(self, player_data):
        super().__init__(player_data)
        
        # Pitcher specific attributes
        self.throws = player_data.get('p_throws', 'R')
        self.era = player_data.get('era', 4.00)
        self.whip = player_data.get('whip', 1.30)
        self.k_rate = player_data.get('k_rate', 0.20)
        self.bb_rate = player_data.get('bb_rate', 0.08)
        # Other relevant stats