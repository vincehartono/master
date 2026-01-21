import requests
from bs4 import BeautifulSoup
import re

url = "https://www.oddsshark.com/nba/computer-picks"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
response.raise_for_status()
soup = BeautifulSoup(response.text, 'html.parser')

# Find all expert pick elements
elements = soup.find_all('div', class_='expert-pick-headline')
print(f"Found {len(elements)} expert pick elements\n")

# Group by parent wrapper
from collections import defaultdict
picks_by_wrapper = defaultdict(list)

for i, element in enumerate(elements):
    element_text = element.get_text(strip=True)
    parent_wrapper = element.parent
    if parent_wrapper:
        wrapper_text = parent_wrapper.get_text(strip=True)
        matchup_match = re.search(r'([A-Z]{2,3})\s+@\s+([A-Z]{2,3})', wrapper_text)
        matchup = f"{matchup_match.group(1)} @ {matchup_match.group(2)}" if matchup_match else "NO_MATCHUP"
    else:
        matchup = "NO_WRAPPER"
    
    picks_by_wrapper[matchup].append(element_text)

print("Picks grouped by matchup:")
for matchup in sorted(picks_by_wrapper.keys()):
    print(f"\n{matchup}:")
    for pick in picks_by_wrapper[matchup]:
        print(f"  - {pick}")
