import requests
from bs4 import BeautifulSoup
import re

url = "https://www.oddsshark.com/nba/expert-picks"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Find all expert pick elements
elements = soup.find_all('div', class_='expert-pick-headline')
print(f"Found {len(elements)} expert pick elements\n")

# Analyze each one
for i, element in enumerate(elements[:10]):
    element_text = element.get_text(strip=True)
    print(f"Element {i}:")
    print(f"  Text: {element_text}")
    
    # Check parent wrapper
    parent = element.parent
    if parent:
        parent_text = parent.get_text(strip=True)
        print(f"  Parent text: {parent_text[:100]}...")
        
        # Try to extract matchup
        matchup_match = re.search(r'([A-Z]{2,3})\s+@\s+([A-Z]{2,3})', parent_text)
        if matchup_match:
            print(f"  Matchup found: {matchup_match.group(1)} @ {matchup_match.group(2)}")
        else:
            print(f"  Matchup: NOT FOUND in parent")
            # Try looking higher up
            grandparent = parent.parent
            if grandparent:
                gp_text = grandparent.get_text(strip=True)
                print(f"  Grandparent text: {gp_text[:100]}...")
                matchup_match = re.search(r'([A-Z]{2,3})\s+@\s+([A-Z]{2,3})', gp_text)
                if matchup_match:
                    print(f"  Matchup found in grandparent: {matchup_match.group(1)} @ {matchup_match.group(2)}")
    print()
