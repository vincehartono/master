import requests
from bs4 import BeautifulSoup
import re

main_url = "https://sportschatplace.com/nba-picks/"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(main_url, headers=headers, timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.content, 'html.parser')

game_links = soup.find_all('a', href=re.compile(r'https://sportschatplace\.com/nba-picks/.*-nba-picks-today'))

print(f"Found {len(game_links)} game links\n")

# Check the first few game links
for link in game_links[:5]:
    game_url = link.get('href', '')
    game_text = link.get_text(strip=True)
    print(f"URL: {game_url}")
    print(f"Text: {game_text}\n")
    
    # Try to extract matchup from URL
    matchup_match = re.search(r'/([a-z-]+)-nba-picks-today', game_url)
    if matchup_match:
        matchup_str = matchup_match.group(1)
        print(f"  Extracted from URL: {matchup_str}")
    
    # Visit the page
    try:
        game_response = requests.get(game_url, headers=headers, timeout=10)
        game_response.raise_for_status()
        game_soup = BeautifulSoup(game_response.content, 'html.parser')
        
        # Look for title or heading that might contain matchup
        title = game_soup.find('title')
        if title:
            print(f"  Page title: {title.get_text()}")
        
        h1 = game_soup.find('h1')
        if h1:
            print(f"  H1: {h1.get_text(strip=True)}")
        
        # Look for any text containing team codes or matchup info
        page_text = game_soup.get_text()
        # Search for team code patterns like "LAC vs DEN" or similar
        team_pattern = re.search(r'([A-Z]{2,3})\s+(?:vs|@)\s+([A-Z]{2,3})', page_text)
        if team_pattern:
            print(f"  Teams found on page: {team_pattern.group(1)} vs {team_pattern.group(2)}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print()
