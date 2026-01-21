import random

def generate_at_bat_result(batter, pitcher):
    """Simulate the outcome of an at-bat based on batter and pitcher models."""
    # Get base probabilities from player models
    k_probability = pitcher.k_rate
    bb_probability = pitcher.bb_rate
    hit_probability = batter.avg * (1 - k_probability - bb_probability)
    
    # Apply matchup adjustments
    adjustments = batter.get_matchup_adjustments(pitcher)
    hit_probability += adjustments.get('avg_modifier', 0)
    
    # Determine outcome
    roll = random.random()
    
    if roll < k_probability:
        return 'strikeout'
    elif roll < k_probability + bb_probability:
        return 'walk'
    elif roll < k_probability + bb_probability + hit_probability:
        # It's a hit, determine type
        return _determine_hit_type(batter)
    else:
        return 'field_out'

def _determine_hit_type(batter):
    """Determine the type of hit (single, double, triple, home run)."""
    slug_avg_diff = batter.slg - batter.avg
    
    # Calculate probabilities based on slugging
    hr_prob = min(batter.hr_rate, 0.3)
    triple_prob = 0.02  # Triples are rare
    double_prob = slug_avg_diff / 4  # Approximation
    single_prob = 1 - hr_prob - triple_prob - double_prob
    
    roll = random.random()
    
    if roll < single_prob:
        return 'single'
    elif roll < single_prob + double_prob:
        return 'double'
    elif roll < single_prob + double_prob + triple_prob:
        return 'triple'
    else:
        return 'home_run'