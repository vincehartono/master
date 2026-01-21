import requests
from bs4 import BeautifulSoup

url = "https://www.oddsshark.com/nba/computer-picks"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

response = requests.get(url, headers=headers, timeout=10)
soup = BeautifulSoup(response.content, 'html.parser')

# Find pick containers and trace up the DOM
pick_elements = soup.find_all('div', class_='expert-pick-headline')
print(f"Found {len(pick_elements)} picks\n")

if pick_elements:
    pick_elem = pick_elements[1]  # Check the Spurs +4 pick
    print(f"Pick: {pick_elem.get_text(strip=True)}\n")
    
    # Walk up the DOM and collect all classes
    current = pick_elem
    level = 0
    while current and level < 10:
        if hasattr(current, 'name') and current.name:
            classes = current.get('class', [])
            print(f"Level {level}: <{current.name}> classes={classes}")
            
            # Look for text content that might indicate game
            text = current.get_text(strip=True)[:100] if current.name not in ['script', 'style'] else ''
            if text and len(text) > 0:
                print(f"       Text: {text[:60]}...")
        
        current = current.parent
        level += 1
