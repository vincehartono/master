"""
Script to normalize matchups across all picks to use correct visitor @ home format
"""
import csv

# The canonical matchups for today (based on your NBA schedule)
# Each entry is (visitor_code, home_code): 'VISITOR @ HOME'
CANONICAL_MATCHUPS = {
    ('PHX', 'PHI'): 'PHX @ PHI',
    ('LAC', 'CHI'): 'LAC @ CHI',
    ('SAS', 'HOU'): 'SAS @ HOU',
    ('MIN', 'UTA'): 'MIN @ UTA',
    ('TOR', 'GSW'): 'TOR @ GSW',
    ('MIA', 'SAC'): 'MIA @ SAC',
    ('LAL', 'DEN'): 'LAL @ DEN',
}

# Alternative team codes that should be converted to canonical
CODE_ALIASES = {
    'PHO': 'PHX',  # Phoenix alternative code
}

def normalize_team_code(code):
    """Convert alternative team codes to canonical ones"""
    return CODE_ALIASES.get(code, code)

def normalize_matchup(matchup_str):
    """Normalize a matchup string to canonical format"""
    if not matchup_str or '@' not in matchup_str:
        return matchup_str
    
    # Extract the two team codes
    parts = matchup_str.split('@')
    if len(parts) != 2:
        return matchup_str
    
    team1 = normalize_team_code(parts[0].strip())
    team2 = normalize_team_code(parts[1].strip())
    
    # Check both orderings
    pair1 = (team1, team2)
    pair2 = (team2, team1)
    
    if pair1 in CANONICAL_MATCHUPS:
        return CANONICAL_MATCHUPS[pair1]
    elif pair2 in CANONICAL_MATCHUPS:
        return CANONICAL_MATCHUPS[pair2]
    else:
        # Return normalized but not in canonical list
        return f"{team1} @ {team2}"

# Read the CSV and normalize matchups
normalized_rows = []

with open('./results/super_cleaned_picks.csv', 'r') as infile:
    reader = csv.DictReader(infile)
    for row in reader:
        row['matchup'] = normalize_matchup(row['matchup'])
        normalized_rows.append(row)

# Write back
with open('./results/super_cleaned_picks.csv', 'w', newline='') as outfile:
    writer = csv.DictWriter(outfile, fieldnames=['pick', 'pick_type', 'matchup', 'source'])
    writer.writeheader()
    writer.writerows(normalized_rows)

print(f"Normalized {len(normalized_rows)} picks to canonical matchup format")
