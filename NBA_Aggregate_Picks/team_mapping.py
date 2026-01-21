"""
NBA Team Code Mapping
Maps team names and variations to standard 3-letter team codes
"""

TEAM_MAPPING = {
    # Phoenix Suns
    'PHX': ['Phoenix', 'Suns', 'PHO'],
    'PHI': ['Philadelphia', '76ers', 'Sixers'],
    'LAC': ['Clippers', 'LA Clippers', 'L.A. Clippers'],
    'CHI': ['Chicago', 'Bulls'],
    'SAS': ['San Antonio', 'Spurs'],
    'HOU': ['Houston', 'Rockets'],
    'MIN': ['Minnesota', 'Timberwolves'],
    'UTA': ['Utah', 'Jazz'],
    'LAL': ['Lakers', 'LA Lakers', 'L.A. Lakers', 'LA', 'L.A.'],
    'DEN': ['Denver', 'Nuggets'],
    'TOR': ['Toronto', 'Raptors'],
    'GSW': ['Golden State', 'Warriors', 'GS', 'Golden St.'],
    'MIA': ['Miami', 'Heat'],
    'SAC': ['Sacramento', 'Kings'],
    # Add more teams as needed
    'BOS': ['Boston', 'Celtics'],
    'BRK': ['Brooklyn', 'Nets'],
    'CLE': ['Cleveland', 'Cavaliers'],
    'DAL': ['Dallas', 'Mavericks'],
    'DET': ['Detroit', 'Pistons'],
    'IND': ['Indiana', 'Pacers'],
    'MEM': ['Memphis', 'Grizzlies'],
    'MIL': ['Milwaukee', 'Bucks'],
    'NOP': ['New Orleans', 'Pelicans'],
    'NYK': ['New York', 'Knicks'],
    'OKC': ['Oklahoma City', 'Thunder'],
    'ORL': ['Orlando', 'Magic'],
    'POR': ['Portland', 'Trail Blazers', 'Blazers'],
    'WAS': ['Washington', 'Wizards'],
    'ATL': ['Atlanta', 'Hawks'],
}

def get_team_code(team_name: str) -> str:
    """
    Convert team name to team code
    
    Args:
        team_name: Team name (e.g., 'Lakers', 'LA Lakers', 'LAL')
    
    Returns:
        Team code (e.g., 'LAL') or original input if not found
    """
    if not team_name:
        return ''
    
    team_upper = team_name.upper().strip()
    
    # Check if it's already a code
    if team_upper in TEAM_MAPPING:
        return team_upper
    
    # Search for the team name in variations
    for code, names in TEAM_MAPPING.items():
        for name in names:
            if name.upper() == team_upper:
                return code
    
    return team_name


def create_reverse_mapping() -> dict:
    """Create reverse mapping from team codes to primary names"""
    reverse = {}
    for code, names in TEAM_MAPPING.items():
        reverse[code] = names[0]  # Use first name as primary
    return reverse


if __name__ == '__main__':
    # Test the mapping
    test_names = ['Lakers', 'LA Lakers', 'LAL', 'Golden State', 'Warriors', 'GSW', 'Chicago', 'Bulls', 'CHI']
    for name in test_names:
        print(f"{name:20} -> {get_team_code(name)}")
    
    print("\nReverse mapping:")
    reverse = create_reverse_mapping()
    for code, name in sorted(reverse.items()):
        print(f"{code} -> {name}")
