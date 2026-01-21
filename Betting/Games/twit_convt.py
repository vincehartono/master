import re
from datetime import datetime
import tweepy

# Team code mapping
team_code_map = {
    'Atlanta Hawks': 'ATL', 'Brooklyn Nets': 'BKN', 'Boston Celtics': 'BOS', 'Charlotte Hornets': 'CHA',
    'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE', 'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN',
    'Detroit Pistons': 'DET', 'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'Indiana Pacers': 'IND',
    'Los Angeles Clippers': 'LAC', 'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
    'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP', 'New York Knicks': 'NYK',
    'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL', 'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX',
    'Portland Trail Blazers': 'POR', 'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
    'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
}

# Ask for user input
input_text = input("Please enter the game details: ")

# Get today's date
today_date = datetime.today().strftime("%B %d, %Y")

# Split the input text into individual game details
game_lines = re.split(r'Basketball - NBA -', input_text)[1:]

# Function to get team codes
def get_team_code(team_name):
    return team_code_map.get(team_name.strip(), team_name.strip())

game_details = []
for game_line in game_lines:
    # Extract team names, spread, and odds
    match = re.search(r'(.*?) vs (.*?) -.*?\| \d+ (.*?) (.*?) For Game', game_line)
    if match:
        team1 = get_team_code(match.group(1).strip())
        team2 = get_team_code(match.group(2).strip())
        spread = match.group(3).strip()
        odds = match.group(4).strip()
        game_details.append(f"{team1} vs {team2} | {spread} {odds}")

# Generate the output text
output_text = f"{today_date}\n"
for detail in game_details:
    output_text += f"{detail}\n"

output_text += "#NBABets #NBAOdds #NBAPropBets #DailyNBA"

# Print the output text
print(output_text)
