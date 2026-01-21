import requests
from bs4 import BeautifulSoup
import re

# Team name to code mapping
TEAM_MAPPING = {
    'LAC': ['Clippers', 'LA Clippers', 'L.A. Clippers', 'Los Angeles Clippers'],
    'LAL': ['Lakers', 'LA Lakers', 'L.A. Lakers', 'Los Angeles Lakers'],
    'GSW': ['Warriors', 'Golden State Warriors', 'Golden State'],
    'TOR': ['Raptors', 'Toronto Raptors'],
    'SAS': ['Spurs', 'San Antonio Spurs', 'SA'],
    'HOU': ['Rockets', 'Houston Rockets'],
    'DEN': ['Nuggets', 'Denver Nuggets'],
    'MIA': ['Heat', 'Miami Heat'],
    'SAC': ['Kings', 'Sacramento Kings'],
    'MIN': ['Timberwolves', 'Minnesota Timberwolves'],
    'UTA': ['Jazz', 'Utah Jazz'],
    'PHX': ['Suns', 'Phoenix Suns'],
    'PHI': ['76ers', 'Philadelphia 76ers'],
    'CHI': ['Bulls', 'Chicago Bulls'],
    'BOS': ['Celtics', 'Boston Celtics'],
    'NYK': ['Knicks', 'New York Knicks'],
}

def get_team_code_from_name(name):
    """Extract team code from full team name"""
    name_lower = name.lower().strip()
    for code, variants in TEAM_MAPPING.items():
        for variant in variants:
            if variant.lower() == name_lower:
                return code
    return name

main_url = "https://sportschatplace.com/nba-picks/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(main_url, headers=headers, timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')

game_links = soup.find_all('a', href=re.compile(r'https://sportschatplace\.com/nba-picks/.*-nba-picks-today'))

print(f"Found {len(game_links)} game links\n")

# Check game links to extract matchups
for link in game_links[:5]:
    game_url = link.get('href', '')
    
    try:
        game_response = requests.get(game_url, headers=headers, timeout=10)
        game_response.raise_for_status()
        game_soup = BeautifulSoup(game_response.content, 'html.parser')
        
        # Get page title
        title = game_soup.find('title')
        if title:
            title_text = title.get_text()
            print(f"Title: {title_text}")
            
            # Extract matchup from title like "Nuggets vs Lakers Prediction"
            # Pattern: Word(s) vs Word(s) Prediction
            match = re.search(r'(.+?)\s+vs\s+(.+?)\s+Prediction', title_text, re.IGNORECASE)
            if match:
                team1 = match.group(1).strip()
                team2 = match.group(2).strip()
                code1 = get_team_code_from_name(team1)
                code2 = get_team_code_from_name(team2)
                
                # Sort alphabetically
                matchup = f"{sorted([code1, code2])[0]} vs {sorted([code1, code2])[1]}"
                print(f"  Extracted: {team1} vs {team2}")
                print(f"  Codes: {code1} vs {code2}")
                print(f"  Matchup: {matchup}")
        
        print()
    except Exception as e:
        print(f"  Error: {e}")
        print()
