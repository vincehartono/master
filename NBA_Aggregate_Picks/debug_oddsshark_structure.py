import requests
from bs4 import BeautifulSoup

url = "https://www.oddsshark.com/nba/computer-picks"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

print("Looking for game/matchup structure...\n")

# Find pick containers and their parent elements
pick_elements = soup.find_all('div', class_='expert-pick-headline')
print(f"Found {len(pick_elements)} picks\n")

# Check the structure around picks
for i, pick_elem in enumerate(pick_elements[:3]):
    print(f"\nPick {i+1}: {pick_elem.get_text(strip=True)[:50]}")
    
    # Look at parent elements
    parent = pick_elem.parent
    if parent:
        print(f"  Parent tag: {parent.name}")
        print(f"  Parent classes: {parent.get('class', [])}")
        
        # Look for game info in siblings or parent's siblings
        siblings = list(parent.find_all(['div', 'span'], class_=lambda x: x and ('game' in x.lower() or 'matchup' in x.lower() or 'team' in x.lower())))
        if siblings:
            print(f"  Found game/team info: {[s.get_text(strip=True)[:30] for s in siblings[:2]]}")
        
        # Look at grandparent
        grandparent = parent.parent
        if grandparent:
            print(f"  Grandparent tag: {grandparent.name}")
            print(f"  Grandparent classes: {grandparent.get('class', [])}")
