"""
NBA Picks Aggregator

This module collects and aggregates NBA picks from multiple sources:
- OddsShark
- CBS Sports
- PicksWise
- SportsBook Review
- Sports Chat Place

Features:
- Normalizes team names and betting formats
- Cleans contradictions (same matchup picks from different sources)
- Generates recommendations based on consensus
- Exports to CSV and generates reports
- Optional S3 upload capability

Author: VT
Date: 2026
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import re
import logging
from typing import List, Dict, Tuple, Optional
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nba_picks.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def extract_team_from_pick(pick_text: str) -> Optional[str]:
    """
    Extract team name from a pick text
    
    Examples:
        "Lakers -3.5" -> "Lakers"
        "PHO -2.5" -> "PHO"
        "Spurs +4" -> "Spurs"
        "D. Fox O8.5" -> "Fox" (player last name from prop)
        "Lebron James O6.5" -> "James" (player last name from prop)
    
    Args:
        pick_text: The pick text to parse
        
    Returns:
        Team name/code or player last name if found, None otherwise
    """
    # Pattern for spread: Team +/- Number
    spread_pattern = r'^([A-Za-z\.\s]+?)\s+[+-]\d+\.?\d*'
    spread_match = re.search(spread_pattern, pick_text.strip())
    if spread_match:
        return spread_match.group(1).strip()
    
    # Pattern for abbreviation like "PHO -2.5"
    abbrev_pattern = r'^([A-Z]{2,3})\s+[+-]\d+\.?\d*'
    abbrev_match = re.search(abbrev_pattern, pick_text.strip())
    if abbrev_match:
        return abbrev_match.group(1).strip()
    
    # Pattern for player prop: "Player Name O/U Number" - extract last name
    player_pattern = r'^(?:(?:[A-Z]\.?\s+)?[A-Z][a-z]+\s+)?([A-Z][a-z]+)\s+[OU]\d+\.?\d*'
    player_match = re.search(player_pattern, pick_text.strip())
    if player_match:
        # Try to match the player last name with a team (some players are well-known team names)
        return player_match.group(1).strip()
    
    return None

def is_player_prop(pick_text: str) -> bool:
    """
    Detect if a pick is a player prop (vs team prop/spread)
    
    Player props typically have format: "Player Name O/U Number" or "Player Name O Number"
    Examples: "J. Green O13.5", "LeBron James U6.5", "Kyrie Irving O25.0"
    
    Args:
        pick_text: The pick text to analyze
        
    Returns:
        True if appears to be a player prop, False otherwise
    """
    # Pattern: Capital letter(s) possibly with period, followed by name, then O/U, then number
    # Examples: "J. Green O13.5", "LeBron James OVER 6.5", "Anthony Davis U20.0"
    pattern = r'([A-Z]\.?\s+)?[A-Z][a-z]+(\s+[A-Z][a-z]+)?\s+[OU]\d+\.?\d*'
    return bool(re.search(pattern, pick_text))

def get_nba_matchups() -> Dict[str, str]:
    """
    Fetch today's NBA matchups from CBS Sports schedule
    
    Returns:
        Dict mapping team names to matchups in standardized format (e.g., 'lakers': 'LAL vs PHI')
    """
    matchups = {}
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Fetch from CBS Sports schedule page
        response = requests.get(
            'https://www.cbssports.com/nba/schedule/',
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all game rows in the schedule table
        game_rows = soup.find_all('tr', class_='TableBase-bodyTr')
        
        for row in game_rows:
            tds = row.find_all('td')
            if len(tds) >= 2:
                away_team = tds[0].get_text(strip=True)
                home_team = tds[1].get_text(strip=True)
                
                if away_team and home_team:
                    # Convert team names to team codes
                    away_code = _get_team_code(away_team)
                    home_code = _get_team_code(home_team)
                    
                    if away_code and home_code:
                        matchup_str = f"{away_code} vs {home_code}"
                        matchups[away_team.lower()] = matchup_str
                        matchups[home_team.lower()] = matchup_str
                        matchups[away_code.lower()] = matchup_str
                        matchups[home_code.lower()] = matchup_str
        
        logger.info(f"Fetched {len(matchups)//4 if matchups else 0} NBA matchups from CBS Sports")
    except Exception as e:
        logger.warning(f"Could not fetch matchups from CBS Sports: {e}")
    
    return matchups


def _get_team_code(team_name: str) -> Optional[str]:
    """
    Convert team name to 3-letter code
    
    Args:
        team_name: Full team name (e.g., "L.A. Clippers")
        
    Returns:
        3-letter team code (e.g., "LAC") or None
    """
    team_name_lower = team_name.lower()
    
    for code, mapping in TEAM_MAPPING.items():
        if isinstance(mapping, list):
            for variant in mapping:
                if variant.lower() in team_name_lower or team_name_lower in variant.lower():
                    return code[:3] if len(code) > 3 else code
        else:
            if mapping.lower() in team_name_lower or team_name_lower in mapping.lower():
                return code[:3] if len(code) > 3 else code
    
    # Try to extract abbreviation from team name
    if ' ' in team_name:
        words = team_name.split()
        # Handle cases like "L.A. Clippers" or "San Antonio"
        if len(words[0]) <= 2:  # L.A., S.F., etc.
            abbrev = words[0].replace('.', '') + words[-1][:2]
        else:
            abbrev = ''.join([w[0] for w in words])
        return abbrev.upper()[:3]
    
    return None


# Import team mapping if available
try:
    from team_mapping import TEAM_MAPPING
except ImportError:
    logger.warning("team_mapping.py not found. Using default team mapping.")
    TEAM_MAPPING = {
        'Atlanta Hawks': 'ATL', 'Boston Celtics': 'BOS', 'Brooklyn Nets': 'BRK',
        'Charlotte Hornets': 'CHA', 'Chicago Bulls': 'CHI', 'Cleveland Cavaliers': 'CLE',
        'Dallas Mavericks': 'DAL', 'Denver Nuggets': 'DEN', 'Detroit Pistons': 'DET',
        'Golden State Warriors': 'GSW', 'Houston Rockets': 'HOU', 'LA Clippers': 'LAC',
        'Los Angeles Lakers': 'LAL', 'Memphis Grizzlies': 'MEM', 'Miami Heat': 'MIA',
        'Milwaukee Bucks': 'MIL', 'Minnesota Timberwolves': 'MIN', 'New Orleans Pelicans': 'NOP',
        'New York Knicks': 'NYK', 'Oklahoma City Thunder': 'OKC', 'Orlando Magic': 'ORL',
        'Philadelphia 76ers': 'PHI', 'Phoenix Suns': 'PHX', 'Portland Trail Blazers': 'POR',
        'Sacramento Kings': 'SAC', 'San Antonio Spurs': 'SAS', 'Toronto Raptors': 'TOR',
        'Utah Jazz': 'UTA', 'Washington Wizards': 'WAS'
    }


# ============================================================================
# EXTERNAL SCRAPER FUNCTIONS
# ============================================================================

def get_sbr_picks() -> List[Dict]:
    """
    Scrape picks from SportsBook Review (SBR)
    
    Returns:
        List of dicts with keys: pick, matchup, pick_type, source_url, confidence, expert
    """
    picks = []
    try:
        url = "https://www.sportsbookreview.com/picks/nba/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find pick containers
        pick_elements = soup.find_all('div', class_=['pick-item', 'pick-container', 'nba-pick'])
        
        for element in pick_elements:
            try:
                # Extract pick information
                pick_text = element.get_text(strip=True)
                
                # Try to extract expert name
                expert = "Unknown"
                expert_elem = element.find(['span', 'div'], class_=['expert', 'author', 'picker-name'])
                if expert_elem:
                    expert = expert_elem.get_text(strip=True)
                
                # Try to extract matchup
                matchup_elem = element.find(['div', 'span'], class_=['matchup', 'game', 'game-info'])
                matchup = matchup_elem.get_text(strip=True) if matchup_elem else ""
                
                # Try to extract pick (team pick)
                pick_elem = element.find(['div', 'span'], class_=['pick', 'selection', 'team-pick'])
                pick_team = pick_elem.get_text(strip=True) if pick_elem else ""
                
                if matchup and pick_team:
                    picks.append({
                        'pick': pick_team,
                        'matchup': matchup,
                        'pick_type': 'Spread',
                        'source_url': url,
                        'confidence': 'Medium',
                        'expert': expert,
                        'source': 'SBR'
                    })
            except Exception as e:
                logger.debug(f"Error parsing SBR pick element: {e}")
                continue
        
        logger.info(f"SBR: Retrieved {len(picks)} picks")
    except Exception as e:
        logger.error(f"Error scraping SBR: {e}")
    
    return picks


def get_sportschatplace_picks() -> List[Dict]:
    """
    Scrape picks from Sports Chat Place
    
    Returns:
        List of dicts with keys: pick, matchup, pick_type, source_url, confidence, expert
    """
    picks = []
    seen_picks = set()  # Track unique picks to avoid duplicates
    
    try:
        main_url = "https://sportschatplace.com/nba-picks/"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        # Fetch main page to get game links
        response = requests.get(main_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all game links
        game_links = soup.find_all('a', href=re.compile(r'https://sportschatplace\.com/nba-picks/.*-nba-picks-today'))
        
        logger.debug(f"Found {len(game_links)} game links on main page")
        
        # Visit each game link and extract picks
        for link in game_links[:10]:  # Limit to first 10 games to avoid excessive requests
            try:
                game_url = link.get('href', '')
                if not game_url:
                    continue
                
                game_response = requests.get(game_url, headers=headers, timeout=10)
                game_response.raise_for_status()
                
                game_soup = BeautifulSoup(game_response.content, 'html.parser')
                
                # Extract matchup from page title
                matchup = ""
                title = game_soup.find('title')
                if title:
                    title_text = title.get_text()
                    # Extract matchup from title like "Nuggets vs Lakers Prediction"
                    title_match = re.search(r'(.+?)\s+vs\s+(.+?)\s+Prediction', title_text, re.IGNORECASE)
                    if title_match:
                        team1 = title_match.group(1).strip()
                        team2 = title_match.group(2).strip()
                        code1 = _get_team_code(team1)
                        code2 = _get_team_code(team2)
                        # Preserve the order from the page (visitor @ home)
                        matchup = f"{code1} @ {code2}"
                
                # Find all H2 tags with picks (format: "Expert Name's Free Pick: Team +/- Spread")
                h2_tags = game_soup.find_all('h2')
                
                for h2 in h2_tags:
                    h2_text = h2.get_text(strip=True)
                    
                    # Look for pattern like "Peter Tran's Free Pick: Denver +2.5"
                    match = re.search(r"(.+?)'s\s+Free\s+Pick:\s+(.+)", h2_text, re.IGNORECASE)
                    if match:
                        expert_name = match.group(1).strip()
                        pick_text = match.group(2).strip()
                        
                        # Create a unique key to track duplicates (expert + pick)
                        pick_key = f"{expert_name}:{pick_text}"
                        
                        if pick_key not in seen_picks:
                            seen_picks.add(pick_key)
                            picks.append({
                                'pick': pick_text,
                                'matchup': matchup,
                                'pick_type': 'spread',
                                'source_url': game_url,
                                'confidence': 0.8,
                                'expert': expert_name,
                                'source': 'SportsChat'
                            })
                        
            except Exception as e:
                logger.debug(f"Error scraping game page {game_url}: {e}")
                continue
        
        logger.info(f"Sports Chat Place: Retrieved {len(picks)} picks")
    except Exception as e:
        logger.error(f"Error scraping Sports Chat Place: {e}")
    
    return picks



# ============================================================================
# MAIN PICKS AGGREGATOR CLASS
# ============================================================================

class PicksAggregator:
    """
    Main class for aggregating NBA picks from multiple sources
    """
    
    def __init__(self, output_dir: str = './output', use_s3: bool = False, s3_bucket: Optional[str] = None):
        """
        Initialize the PicksAggregator
        
        Args:
            output_dir: Directory for output files
            use_s3: Whether to upload results to S3
            s3_bucket: S3 bucket name (if use_s3=True)
        """
        self.output_dir = output_dir
        self.use_s3 = use_s3
        self.s3_bucket = s3_bucket
        self.picks = []
        self.normalized_picks = []
        self.cleaned_picks = []
        self.nba_games = []
        self.timestamp = datetime.now().strftime('%Y%m%d')
        
        # Create output directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        logger.info(f"PicksAggregator initialized. Output dir: {self.output_dir}")
    
    def add_source(self, source_name: str, picks_data: List[Dict]) -> None:
        """
        Add picks from a source, removing duplicates
        
        Args:
            source_name: Name of the source
            picks_data: List of pick dictionaries
        """
        # Remove duplicates within this source
        unique_picks = {}
        for pick in picks_data:
            # Create a unique key based on pick text, matchup, and pick_type
            key = (pick.get('pick', ''), pick.get('matchup', ''), pick.get('pick_type', ''))
            if key not in unique_picks:
                unique_picks[key] = pick
        
        picks_to_add = list(unique_picks.values())
        logger.info(f"Adding {len(picks_to_add)} picks from {source_name} ({len(picks_data) - len(picks_to_add)} duplicates removed)")
        
        for pick in picks_to_add:
            if 'source' not in pick:
                pick['source'] = source_name
        
        self.picks.extend(picks_to_add)
    
    def get_nba_games_today(self) -> List[Dict]:
        """
        Fetch today's NBA games from NBA.com
        
        Returns:
            List of game dictionaries with: home_team, away_team, time, status
        """
        games = []
        try:
            # Using NBA official stats API
            url = "https://stats.nba.com/stats/scoreboard"
            today = datetime.now().strftime('%Y%m%d')
            
            params = {
                'LeagueID': '00',
                'DayOffset': '0'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': 'https://www.nba.com/'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'resultSets' in data and len(data['resultSets']) > 0:
                games_data = data['resultSets'][0]['rowSet']
                headers = data['resultSets'][0]['headers']
                
                for game in games_data:
                    game_dict = dict(zip(headers, game))
                    games.append({
                        'game_id': game_dict.get('GAME_ID', ''),
                        'home_team': game_dict.get('HOME_TEAM_NAME', ''),
                        'away_team': game_dict.get('VISITOR_TEAM_NAME', ''),
                        'time': game_dict.get('GAME_STATUS_TEXT', ''),
                        'status': game_dict.get('GAME_STATUS', '')
                    })
            
            logger.info(f"Retrieved {len(games)} games from NBA.com")
        except Exception as e:
            logger.error(f"Error fetching NBA games: {e}")
        
        self.nba_games = games
        return games
    
    def get_picks_from_oddsshark(self) -> List[Dict]:
        """
        Fetch picks from OddsShark with matchup information
        
        Returns:
            List of pick dictionaries
        """
        picks = []
        try:
            url = "https://www.oddsshark.com/nba/computer-picks"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find pick headlines with class "expert-pick-headline"
            pick_elements = soup.find_all('div', class_='expert-pick-headline')
            
            for element in pick_elements:
                try:
                    element_text = element.get_text(strip=True)
                    
                    # Extract matchup from parent wrapper
                    matchup = ""
                    parent_wrapper = element.parent
                    if parent_wrapper:
                        wrapper_text = parent_wrapper.get_text(strip=True)
                        # Extract matchup like "SAS @ HOU" from "Spurs +4NBA | SAS @ HOU 8:00 PM ET"
                        matchup_match = re.search(r'([A-Z]{2,3})\s+@\s+([A-Z]{2,3})', wrapper_text)
                        if matchup_match:
                            team1 = matchup_match.group(1)
                            team2 = matchup_match.group(2)
                            # Preserve @ relationship from source (team1 @ team2)
                            matchup = f"{team1} @ {team2}"
                    
                    # Skip picks without matchup context (likely from non-game sections)
                    if not matchup:
                        continue
                    
                    # Example: "J. Green OVER 13.5 PTS", "Kevin Durant U10.5 Reb", "UNDER 225.5 Points"
                    added = False
                    
                    # Pattern 1: "Player Name OVER/UNDER/O/U Number" 
                    ou_match = re.search(r'(.+?)\s+(OVER|UNDER|[OU])\s*(\d+\.?\d*)', element_text, re.IGNORECASE)
                    if ou_match and ou_match.group(1).strip():
                        player_name = ou_match.group(1).strip()
                        ou_indicator = 'O' if ou_match.group(2).upper() in ('O', 'OVER') else 'U'
                        ou_value = ou_match.group(3)
                        
                        # Check if player prop before deciding on text format
                        test_pick = f"{player_name} {ou_indicator}{ou_value}"
                        is_prop = is_player_prop(test_pick)
                        
                        # For props, use full original text; for team O/U, use formatted version
                        pick_text = element_text if is_prop else test_pick
                        pick_type = 'prop' if is_prop else 'over_under'
                        
                        picks.append({
                            'pick': pick_text,
                            'matchup': matchup,
                            'pick_type': pick_type,
                            'source_url': url,
                            'confidence': 0.8,
                            'expert': 'OddsShark',
                            'source': 'OddsShark'
                        })
                        added = True
                    
                    if not added:
                        # Pattern 2: "OVER/UNDER Number" (team O/U) or "Team +/- Number" (spread)
                        if element_text.startswith('OVER') or element_text.startswith('UNDER'):
                            ou_match = re.search(r'^(OVER|UNDER)\s*(\d+\.?\d*)', element_text, re.IGNORECASE)
                            if ou_match:
                                ou_indicator = 'O' if ou_match.group(1).upper() == 'OVER' else 'U'
                                ou_value = ou_match.group(2)
                                pick_text = f"{ou_indicator}{ou_value}"
                                picks.append({
                                    'pick': pick_text,
                                    'matchup': matchup,
                                    'pick_type': 'over_under',
                                    'source_url': url,
                                    'confidence': 0.8,
                                    'expert': 'OddsShark',
                                    'source': 'OddsShark'
                                })
                                added = True
                        
                        if not added:
                            # Pattern 3: "Team +/- Number" (spread)
                            spread_match = re.search(r'(.+?)\s+([+-]\d+\.?\d*)', element_text)
                            if spread_match:
                                team_or_name = spread_match.group(1).strip()
                                spread_value = spread_match.group(2)
                                pick_text = f"{team_or_name} {spread_value}"
                                picks.append({
                                    'pick': pick_text,
                                    'matchup': matchup,
                                    'pick_type': 'spread',
                                    'source_url': url,
                                    'confidence': 0.8,
                                    'expert': 'OddsShark',
                                    'source': 'OddsShark'
                                })
                                added = True
                        
                        if not added:
                            # Pattern 4: Just numeric value as fallback
                            match = re.search(r'([+-]?\d+\.?\d*)', element_text)
                            if match and element_text.count(' ') < 3:  # Avoid matching in long descriptions
                                pick_text = match.group(1)
                                picks.append({
                                    'pick': pick_text,
                                    'matchup': matchup,
                                    'pick_type': 'spread',
                                    'source_url': url,
                                    'confidence': 0.8,
                                    'expert': 'OddsShark',
                                    'source': 'OddsShark'
                                })
                except Exception as e:
                    logger.debug(f"Error parsing OddsShark pick: {e}")
                    continue
            
            logger.info(f"OddsShark: Retrieved {len(picks)} picks")
        except Exception as e:
            logger.error(f"Error scraping OddsShark: {e}")
        
        return picks
    
    def get_picks_from_cbssports(self) -> List[Dict]:
        """
        Fetch picks from CBS Sports
        
        Properly extracts picks while associating them with their game matchups.
        Each game row contains team info and expert picks side-by-side.
        
        Returns:
            List of pick dictionaries with proper matchup context
        """
        picks = []
        try:
            url = "https://www.cbssports.com/nba/expert-picks/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all game rows (picks-tr divs) - skip header row
            game_rows = soup.find_all('div', class_='picks-tr')
            
            for row in game_rows[1:]:  # Skip header row
                try:
                    # Extract team info from this row
                    teams = []
                    game_info = row.find('div', class_='picks-td')
                    
                    if game_info:
                        # Get both teams from the game
                        game_info_teams = game_info.find_all('div', class_='game-info-team')
                        for team_div in game_info_teams:
                            team_link = team_div.find('a')
                            if team_link and '/nba/teams/' in team_link.get('href', ''):
                                # Extract team code from URL (e.g., /nba/teams/PHO/)
                                team_code_match = re.search(r'/nba/teams/([A-Z]{2,3})/', team_link.get('href', ''))
                                if team_code_match:
                                    teams.append(team_code_match.group(1))
                    
                    # Get expert picks column
                    picks_columns = row.find_all('div', class_='picks-td')
                    if len(picks_columns) >= 3:  # Game info, Current odds, Expert picks
                        expert_picks_col = picks_columns[2]
                        
                        # Get matchup string
                        matchup = f"{teams[0]} vs {teams[1]}" if len(teams) >= 2 else ""
                        
                        # Extract spread picks
                        spread_elements = expert_picks_col.find_all('div', class_='expert-spread')
                        for spread_elem in spread_elements:
                            try:
                                spread_text = spread_elem.get_text(strip=True)
                                
                                # Extract team code (first token like "PHO")
                                team_code = ""
                                lines = spread_text.split()
                                if lines and len(lines[0]) <= 3 and lines[0].isupper():
                                    team_code = lines[0]
                                
                                # Extract spread value (e.g., "-2.5")
                                spread_match = re.search(r'([+-]?\d+\.?\d*)', spread_text)
                                spread_value = spread_match.group(1) if spread_match else ""
                                
                                if team_code and spread_value:
                                    pick_text = f"{team_code} {spread_value}"
                                    picks.append({
                                        'pick': pick_text,
                                        'matchup': matchup,
                                        'pick_type': 'spread',
                                        'source_url': url,
                                        'confidence': 0.8,
                                        'expert': 'CBS Sports',
                                        'source': 'CBS Sports'
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing CBS Sports spread pick: {e}")
                                continue
                        
                        # Extract O/U picks
                        ou_elements = expert_picks_col.find_all('div', class_='expert-ou')
                        for ou_elem in ou_elements:
                            try:
                                ou_text = ou_elem.get_text(strip=True)
                                
                                # Extract O/U value (e.g., "O222.5")
                                ou_match = re.search(r'([OU]\d+\.?\d*)', ou_text)
                                ou_value = ou_match.group(1) if ou_match else ""
                                
                                if ou_value:
                                    picks.append({
                                        'pick': ou_value,
                                        'matchup': matchup,
                                        'pick_type': 'over_under',
                                        'source_url': url,
                                        'confidence': 0.8,
                                        'expert': 'CBS Sports',
                                        'source': 'CBS Sports'
                                    })
                            except Exception as e:
                                logger.debug(f"Error parsing CBS Sports O/U pick: {e}")
                                continue
                
                except Exception as e:
                    logger.debug(f"Error parsing CBS Sports game row: {e}")
                    continue
            
            logger.info(f"CBS Sports: Retrieved {len(picks)} picks from {len(game_rows) - 1} games")
        except Exception as e:
            logger.error(f"Error scraping CBS Sports: {e}")
        
        return picks
    
    def get_picks_from_pickswise(self) -> List[Dict]:
        """
        Fetch picks from PicksWise
        
        Returns:
            List of pick dictionaries
        """
        picks = []
        try:
            url = "https://www.pickswise.com/nba/picks"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find pick outcome elements with class "SelectionInfo_outcome__VQd3L"
            outcome_elements = soup.find_all('div', class_=re.compile(r'SelectionInfo_outcome'))
            
            for element in outcome_elements:
                try:
                    element_text = element.get_text(strip=True)
                    # Remove the odds notation (e.g., "(-110)")
                    element_text = re.sub(r'\([-+]?\d+\)', '', element_text).strip()
                    
                    # Get team headers from parent container
                    current = element
                    team_headers = []
                    for level in range(10):  # Walk up max 10 levels
                        if current.parent:
                            current = current.parent
                            found_headers = current.find_all('div', class_=re.compile(r'PickCardHeader_team'))
                            if found_headers:
                                team_headers = found_headers
                                break
                        else:
                            break
                    
                    # Extract team codes from headers
                    teams = []
                    for header in team_headers:
                        team_name = header.get_text(strip=True)
                        # Map team name to code
                        for code, mapping in TEAM_MAPPING.items():
                            if any(team_name.lower() == m.lower() for m in mapping):
                                teams.append(code)
                                break
                    
                    # Check if it's an Over/Under pick
                    if element_text.lower().startswith(('over', 'under')):
                        # O/U pick like "Under 224.5" or "Over 238.0"
                        ou_indicator = element_text[0].upper()  # Get O or U from first letter
                        ou_value = re.search(r'(\d+\.?\d*)', element_text)
                        
                        if ou_value and teams:
                            # Use the second team for O/U (the team the line is for)
                            team_code = teams[-1] if len(teams) > 1 else teams[0]
                            ou_num = ou_value.group(1)
                            
                            pick_text = f"{team_code} {ou_indicator}{ou_num}"
                            picks.append({
                                'pick': pick_text,
                                'matchup': "",
                                'pick_type': 'over_under',
                                'source_url': url,
                                'confidence': 0.8,
                                'expert': 'PicksWise',
                                'source': 'PicksWise'
                            })
                    else:
                        # Spread pick like "PHI 76ers -1.0"
                        match = re.search(r'([A-Z]{2,3})\s+[^\-\+\d]*\s*([+-]\d+\.?\d*)', element_text)
                        
                        if match:
                            team_code = match.group(1)
                            spread_value = match.group(2)
                            
                            pick_text = f"{team_code} {spread_value}"
                            picks.append({
                                'pick': pick_text,
                                'matchup': "",
                                'pick_type': 'spread',
                                'source_url': url,
                                'confidence': 0.8,
                                'expert': 'PicksWise',
                                'source': 'PicksWise'
                            })
                except Exception as e:
                    logger.debug(f"Error parsing PicksWise pick: {e}")
                    continue
            
            logger.info(f"PicksWise: Retrieved {len(picks)} picks")
        except Exception as e:
            logger.error(f"Error scraping PicksWise: {e}")
        
        return picks
    
    def normalize_picks(self) -> None:
        """
        Normalize picks by:
        - Converting team names to 3-letter codes using TEAM_MAPPING
        - Normalizing O/U format to "O###" or "U###"
        - Sorting matchups alphabetically
        - Removing duplicate consecutive team codes
        - Removing Unicode characters
        """
        logger.info("Starting normalization of picks...")
        
        normalized = []
        
        for pick in self.picks:
            try:
                normalized_pick = pick.copy()
                
                # Remove Unicode characters and use ASCII equivalents
                for field in ['pick', 'matchup', 'expert']:
                    if field in normalized_pick:
                        # Replace checkmarks and X marks with ASCII
                        normalized_pick[field] = str(normalized_pick[field]).replace('[+]', '[+]').replace('[-]', '[-]')
                
                # Normalize team names to 3-letter codes in pick text
                pick_text = normalized_pick['pick']
                
                # Collect all replacements and sort by length (longest first to avoid overlaps)
                replacements = []
                for code, team_variants in TEAM_MAPPING.items():
                    variants = team_variants if isinstance(team_variants, list) else [team_variants]
                    for variant in variants:
                        # Only add non-code variants (don't replace already-converted codes)
                        if len(variant) > 3:  # Skip variants that are already 3-letter codes
                            replacements.append((variant, code))
                
                # Sort by length descending to replace longer strings first
                replacements.sort(key=lambda x: len(x[0]), reverse=True)
                
                # Apply replacements (only once per pattern to avoid overlaps)
                seen_patterns = set()
                for variant, code in replacements:
                    if variant not in seen_patterns:
                        pattern = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
                        pick_text = pattern.sub(code, pick_text)
                        seen_patterns.add(variant)
                
                # Normalize 2-letter codes to 3-letter codes
                pick_text = re.sub(r'\bSA\b', 'SAS', pick_text, flags=re.IGNORECASE)  # SA -> SAS
                pick_text = re.sub(r'\bGS\b', 'GSW', pick_text, flags=re.IGNORECASE)  # GS -> GSW
                pick_text = re.sub(r'\bLA\b(?!C)', 'LAL', pick_text, flags=re.IGNORECASE)  # LA -> LAL (but not LAC)
                
                normalized_pick['pick'] = pick_text
                
                # Normalize O/U format
                if 'pick_type' in normalized_pick and normalized_pick['pick_type'] == 'Over/Under':
                    pick_text = normalized_pick['pick'].upper()
                    ou_pattern = r'([OU])[\s\-]?(\d+(?:\.\d+)?)'
                    match = re.search(ou_pattern, pick_text)
                    if match:
                        ou_type = match.group(1)
                        ou_line = match.group(2).split('.')[0]  # Integer part only
                        normalized_pick['pick'] = f"{ou_type}{ou_line}"
                
                # Normalize matchup - replace team names with codes and sort
                matchup_text = normalized_pick['matchup']
                
                # Collect all replacements and sort by length (longest first to avoid overlaps)
                replacements = []
                for code, team_variants in TEAM_MAPPING.items():
                    variants = team_variants if isinstance(team_variants, list) else [team_variants]
                    for variant in variants:
                        # Only add non-code variants (don't replace already-converted codes)
                        if len(variant) > 3:  # Skip variants that are already 3-letter codes
                            replacements.append((variant, code))
                
                # Sort by length descending to replace longer strings first
                replacements.sort(key=lambda x: len(x[0]), reverse=True)
                
                # Apply replacements with word boundaries
                seen_patterns = set()
                for variant, code in replacements:
                    if variant not in seen_patterns:
                        pattern = re.compile(r'\b' + re.escape(variant) + r'\b', re.IGNORECASE)
                        matchup_text = pattern.sub(code, matchup_text)
                        seen_patterns.add(variant)
                
                # Extract team codes and sort (only 3-letter codes, and convert shorter codes)
                # First, normalize any 2-letter codes to 3-letter ones
                matchup_text = re.sub(r'\bSA\b', 'SAS', matchup_text, flags=re.IGNORECASE)  # SA -> SAS
                matchup_text = re.sub(r'\bGS\b', 'GSW', matchup_text, flags=re.IGNORECASE)  # GS -> GSW
                
                team_codes = re.findall(r'\b[A-Z]{3}\b', matchup_text)
                team_codes = list(dict.fromkeys(team_codes))  # Remove duplicates, preserve order
                if len(team_codes) >= 2:
                    # Keep the order from the source (visitor @ home)
                    normalized_pick['matchup'] = f"{team_codes[0]} @ {team_codes[1]}"
                else:
                    normalized_pick['matchup'] = matchup_text
                
                normalized.append(normalized_pick)
            except Exception as e:
                logger.debug(f"Error normalizing pick '{pick.get('pick', 'N/A')}': {e}")
                continue
        
        self.normalized_picks = normalized
        logger.info(f"Normalized {len(normalized)} picks")
    
    def clean_contradictions(self) -> None:
        """
        Clean contradictions by removing picks where multiple teams are chosen
        for the same matchup (e.g., both Team A and Team B in same game)
        """
        logger.info("Cleaning contradictions...")
        
        # For now, skip complex contradiction logic and just use normalized picks
        # Picks are already separated by pick_type (game vs player props)
        self.cleaned_picks = self.normalized_picks
        logger.info(f"Cleaned picks: {len(self.cleaned_picks)} picks retained")
    
    def generate_spreads_report(self) -> str:
        """
        Generate spreads recommendations report
        Recommends:
        - Lowest lines for favorites (negative)
        - Highest lines for underdogs (positive)
        
        Returns:
            Report text
        """
        logger.info("Generating spreads report...")
        
        report = []
        report.append("=" * 80)
        report.append("NBA PICKS AGGREGATOR SPREADS RECOMMENDATIONS")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")
        
        # Tally picks by matchup and team
        matchup_tally = {}
        
        for pick in self.cleaned_picks:
            matchup = pick.get('matchup', 'Unknown')
            team = pick.get('pick', 'Unknown')
            source = pick.get('source', 'Unknown')
            
            # Ensure team is a string (not a list)
            if isinstance(team, list):
                team = ', '.join(str(t) for t in team)
            else:
                team = str(team)
            
            if matchup not in matchup_tally:
                matchup_tally[matchup] = {}
            
            if team not in matchup_tally[matchup]:
                matchup_tally[matchup][team] = {
                    'count': 0,
                    'sources': []
                }
            
            matchup_tally[matchup][team]['count'] += 1
            if source not in matchup_tally[matchup][team]['sources']:
                matchup_tally[matchup][team]['sources'].append(source)
        
        # Generate recommendations
        for matchup in sorted(matchup_tally.keys()):
            report.append(f"\n{matchup}")
            report.append("-" * 40)
            
            picks_for_matchup = matchup_tally[matchup]
            for team in sorted(picks_for_matchup.keys()):
                count = picks_for_matchup[team]['count']
                sources = ', '.join(picks_for_matchup[team]['sources'])
                percentage = (count / max(sum([t['count'] for t in picks_for_matchup.values()]), 1)) * 100
                
                report.append(f"  {team}: {count} picks ({percentage:.0f}%)")
                report.append(f"    Sources: {sources}")
        
        report.append("\n" + "=" * 80)
        report_text = '\n'.join(report)
        logger.info("Spreads report generated")
        
        return report_text
        logger.info("Spreads report generated")
        
        return report_text
    
    def fetch_all_sources(self) -> None:
        """
        Orchestrate fetching picks from all 5 sources and populate matchups from CBS Sports
        """
        logger.info("Fetching picks from all sources...")
        
        # Fetch matchups from CBS Sports first
        nba_matchups = get_nba_matchups()
        
        # Fetch from each source
        sources_data = {
            'OddsShark': self.get_picks_from_oddsshark(),
            'CBSSports': self.get_picks_from_cbssports(),
            'PicksWise': self.get_picks_from_pickswise(),
            'SBR': get_sbr_picks(),
            'SportsChat': get_sportschatplace_picks()
        }
        
        # Add all picks and populate matchups
        for source_name, picks_data in sources_data.items():
            # Try to extract team name from pick and find matching matchup
            for pick in picks_data:
                # Check if this is a player prop and update pick_type if needed
                pick_text = pick.get('pick', '')
                if is_player_prop(pick_text) and pick.get('pick_type') != 'prop':
                    pick['pick_type'] = 'prop'
                
                if not pick.get('matchup'):
                    pick_text_lower = pick_text.lower()
                    matched = False
                    
                    # For spreads and O/U, try to extract team name first
                    extracted_team = extract_team_from_pick(pick_text)
                    
                    if extracted_team:
                        # Try to find matching team in matchups
                        extracted_team_lower = extracted_team.lower()
                        for team_key, matchup in nba_matchups.items():
                            if extracted_team_lower in team_key or team_key in extracted_team_lower:
                                pick['matchup'] = matchup
                                matched = True
                                break
                    
                    # First try: match against matchup keys directly
                    if not matched:
                        for team_key, matchup in nba_matchups.items():
                            if team_key in pick_text_lower:
                                pick['matchup'] = matchup
                                matched = True
                                break
                    
                    # Second try: match using team_mapping abbreviations and names
                    if not matched and TEAM_MAPPING:
                        for team_code, team_variants in TEAM_MAPPING.items():
                            # Check each variant (name, abbreviation, etc)
                            for variant in team_variants if isinstance(team_variants, list) else [team_variants]:
                                if variant.lower() in pick_text:
                                    # Find matching matchup with this team
                                    for team_key, matchup in nba_matchups.items():
                                        if variant.lower() in team_key or team_code.lower() in team_key:
                                            pick['matchup'] = matchup
                                            matched = True
                                            break
                                    if matched:
                                        break
                            if matched:
                                break
            
            self.add_source(source_name, picks_data)
        
        logger.info(f"Total picks from all sources: {len(self.picks)}")

    
    def export_results(self) -> Dict[str, str]:
        """
        Export results to CSV and text files
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("Exporting results...")
        
        output_files = {}
        
        # Ensure results directory exists
        results_dir = './results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        try:
            # Export raw picks
            if len(self.picks) > 0:
                raw_df = pd.DataFrame(self.picks)
                # Add pick_type column if not present (separate game picks from player props)
                if 'pick_type' not in raw_df.columns:
                    raw_df['pick_type'] = 'unknown'
                # Ensure matchup column exists
                if 'matchup' not in raw_df.columns:
                    raw_df['matchup'] = ''
                
                # Reorder columns for clarity
                cols = ['pick', 'pick_type', 'matchup', 'source', 'expert', 'confidence', 'source_url']
                cols = [c for c in cols if c in raw_df.columns]
                raw_df = raw_df[cols + [c for c in raw_df.columns if c not in cols]]
                
                raw_file = os.path.join(
                    results_dir, 
                    f"raw_picks_{self.timestamp}.csv"
                )
                raw_df.to_csv(raw_file, index=False)
                output_files['raw_picks'] = raw_file
                logger.info(f"Exported raw picks to {raw_file}")
            
            # Export normalized picks
            if len(self.normalized_picks) > 0:
                norm_df = pd.DataFrame(self.normalized_picks)
                # Add pick_type column if not present
                if 'pick_type' not in norm_df.columns:
                    norm_df['pick_type'] = 'unknown'
                # Ensure matchup column exists
                if 'matchup' not in norm_df.columns:
                    norm_df['matchup'] = ''
                
                # Keep only essential columns: pick, pick_type, matchup, source
                cols = ['pick', 'pick_type', 'matchup', 'source']
                cols = [c for c in cols if c in norm_df.columns]
                norm_df = norm_df[cols]
                
                norm_file = os.path.join(
                    results_dir,
                    f"cleaned_picks_{self.timestamp}.csv"
                )
                norm_df.to_csv(norm_file, index=False)
                output_files['normalized_picks'] = norm_file
                logger.info(f"Exported normalized picks to {norm_file}")
            
            # Export cleaned picks (no contradictions)
            if len(self.cleaned_picks) > 0:
                cleaned_df = pd.DataFrame(self.cleaned_picks)
                # Add pick_type column if not present
                if 'pick_type' not in cleaned_df.columns:
                    cleaned_df['pick_type'] = 'unknown'
                # Ensure matchup column exists
                if 'matchup' not in cleaned_df.columns:
                    cleaned_df['matchup'] = ''
                
                # Keep only essential columns: pick, pick_type, matchup, source
                cols = ['pick', 'pick_type', 'matchup', 'source']
                cols = [c for c in cols if c in cleaned_df.columns]
                cleaned_df = cleaned_df[cols]
                
                cleaned_file = os.path.join(
                    results_dir,
                    "super_cleaned_picks.csv"
                )
                cleaned_df.to_csv(cleaned_file, index=False)
                output_files['cleaned_picks'] = cleaned_file
                logger.info(f"Exported cleaned picks to {cleaned_file}")
            
            # Export spreads report
            report = self.generate_spreads_report()
            report_file = os.path.join(
                results_dir,
                "spreads_recommendations.txt"
            )
            with open(report_file, 'w') as f:
                f.write(report)
            output_files['report'] = report_file
            logger.info(f"Exported spreads report to {report_file}")
        
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
        
        return output_files
    
    def upload_to_s3(self) -> bool:
        """
        Upload results to AWS S3
        
        Returns:
            True if successful, False otherwise
        """
        if not self.use_s3 or not self.s3_bucket:
            logger.info("S3 upload disabled")
            return False
        
        try:
            import boto3
            logger.info(f"Uploading results to S3 bucket: {self.s3_bucket}")
            
            s3_client = boto3.client('s3')
            
            # Upload all files from output directory
            for root, dirs, files in os.walk(self.output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    s3_key = os.path.relpath(file_path, self.output_dir)
                    
                    try:
                        s3_client.upload_file(
                            file_path,
                            self.s3_bucket,
                            f"nba_picks/{s3_key}"
                        )
                        logger.info(f"Uploaded {s3_key} to S3")
                    except Exception as e:
                        logger.error(f"Error uploading {s3_key}: {e}")
            
            logger.info("S3 upload completed")
            return True
        
        except ImportError:
            logger.error("boto3 not installed. Cannot upload to S3")
            return False
        except Exception as e:
            logger.error(f"Error during S3 upload: {e}")
            return False
    
    def run(self) -> Dict[str, str]:
        """
        Main execution method - orchestrates the entire process
        
        Returns:
            Dictionary with output file paths
        """
        logger.info("Starting NBA Picks Aggregator")
        
        try:
            # Step 1: Fetch today's games (optional - not used by current scrapers)
            # self.get_nba_games_today()
            
            # Step 2: Fetch picks from all sources
            self.fetch_all_sources()
            
            # Step 3: Normalize picks
            self.normalize_picks()
            
            # Step 4: Clean contradictions
            self.clean_contradictions()
            
            # Step 5: Export results
            output_files = self.export_results()
            
            # Step 6: Upload to S3 (if enabled)
            if self.use_s3:
                self.upload_to_s3()
            
            logger.info("NBA Picks Aggregator completed successfully")
            
            return output_files
        
        except Exception as e:
            logger.error(f"Error in main execution: {e}", exc_info=True)
            return {}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """
    Main entry point for the script
    """
    try:
        # Initialize aggregator
        aggregator = PicksAggregator(
            output_dir='./nba_picks_output',
            use_s3=False  # Set to True if you want S3 uploads
        )
        
        # Run the aggregator
        output_files = aggregator.run()
        
        # Print summary
        print("\n" + "=" * 80)
        print("NBA PICKS AGGREGATOR EXECUTION SUMMARY")
        print("=" * 80)
        print(f"\nTotal picks collected: {len(aggregator.picks)}")
        print(f"Picks after normalization: {len(aggregator.normalized_picks)}")
        print(f"Picks after cleaning contradictions: {len(aggregator.cleaned_picks)}")
        print(f"\nOutput files generated:")
        for file_type, file_path in output_files.items():
            print(f"  - {file_type}: {file_path}")
        print("\n" + "=" * 80)
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
