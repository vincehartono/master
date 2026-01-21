import pandas as pd
import numpy as np
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# === TEAM NAME MAPS ===
team_name_map = {
    'ARI': 'Arizona Diamondbacks', 'ATL': 'Atlanta Braves', 'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox', 'CHC': 'Chicago Cubs', 'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds', 'CLE': 'Cleveland Guardians', 'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers', 'HOU': 'Houston Astros', 'KC': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels', 'LAD': 'Los Angeles Dodgers', 'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers', 'MIN': 'Minnesota Twins', 'NYM': 'New York Mets',
    'NYY': 'New York Yankees', 'OAK': 'Oakland Athletics', 'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates', 'SD': 'San Diego Padres', 'SEA': 'Seattle Mariners',
    'SF': 'San Francisco Giants', 'STL': 'St. Louis Cardinals', 'TB': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers', 'TOR': 'Toronto Blue Jays', 'WSH': 'Washington Nationals'
}

nickname_map = {
    'Guardians': 'Cleveland Guardians', 'CLE Guardians': 'Cleveland Guardians',
    'NY Mets': 'New York Mets', 'DET Tigers': 'Detroit Tigers', 'CIN Reds': 'Cincinnati Reds',
    'TEX Rangers': 'Texas Rangers', 'NY Yankees': 'New York Yankees',
    'ATL Braves': 'Atlanta Braves', 'LA Dodgers': 'Los Angeles Dodgers',
    'Astros': 'Houston Astros', 'Athletics': 'Oakland Athletics', 'Orioles': 'Baltimore Orioles',
    'Rockies': 'Colorado Rockies', 'Twins': 'Minnesota Twins', 'Dodgers': 'Los Angeles Dodgers',
    'Braves': 'Atlanta Braves', 'Tigers': 'Detroit Tigers', 'Yankees': 'New York Yankees',
    'Mets': 'New York Mets'
}

normalization_map = {
    "Mariners": "Seattle Mariners", "Diamondbacks": "Arizona Diamondbacks",
    "Royals": "Kansas City Royals", "Brewers": "Milwaukee Brewers",
    "Marlins": "Miami Marlins", "Yankees": "New York Yankees",
    "Red Sox": "Boston Red Sox", "Dodgers": "Los Angeles Dodgers",
    "Giants": "San Francisco Giants", "Cubs": "Chicago Cubs",
    "Guardians": "Cleveland Guardians", "Mets": "New York Mets",
    "Angels": "Los Angeles Angels", "Orioles": "Baltimore Orioles",
    "Cardinals": "St. Louis Cardinals", "Tigers": "Detroit Tigers",
    "Astros": "Houston Astros", "Braves": "Atlanta Braves",
    "Padres": "San Diego Padres", "Reds": "Cincinnati Reds",
    "Rangers": "Texas Rangers", "Athletics": "Oakland Athletics",
    "Nationals": "Washington Nationals", "Twins": "Minnesota Twins",
    "Pirates": "Pittsburgh Pirates", "Rockies": "Colorado Rockies",
    "Rays": "Tampa Bay Rays", "Phillies": "Philadelphia Phillies",
    "White Sox": "Chicago White Sox"
}

# === CLEANING FUNCTIONS ===
def remove_over_under(cell):
    if pd.isna(cell):
        return np.nan
    if 'over' in cell.lower() or 'under' in cell.lower():
        return np.nan
    return cell

def standardize_entry(entry):
    if pd.isna(entry):
        return entry
    match = re.search(r'([A-Za-z\s\.]+)[^\d\-+]*([-+]?\d{2,3})', entry)
    if match:
        team_raw = match.group(1).strip()
        odds = match.group(2)
        team_raw = re.sub(r'\s+', ' ', team_raw)
        for abbr, full_name in team_name_map.items():
            if abbr in team_raw.replace(' ', ''):
                team_raw = full_name
                break
        for nickname, full_name in nickname_map.items():
            if nickname in team_raw:
                team_raw = full_name
                break
        return f"{team_raw} {odds}"
    return entry.strip()

def normalize_team_name(entry):
    if pd.isna(entry):
        return entry
    for short, full in normalization_map.items():
        if re.search(rf'\b{re.escape(short)}\b', entry, re.IGNORECASE):
            return full
    return entry.strip()

def post_process_combined_df(df):
    # Clean BetFirm column BEFORE anything else
    if "BetFirm" in df.columns:
        df["BetFirm"] = df["BetFirm"].str.replace(r"(?i)play on\s*:? ?", "", regex=True)

    # Remove over/under
    df = df.map(remove_over_under)

    # Standardize and clean odds
    for col in df.columns:
        df[col] = df[col].dropna().reset_index(drop=True)
        df[col] = df[col].apply(standardize_entry)

    # Strip numeric odds
    df = df.map(lambda x: re.sub(r'[-+]?\d+', '', str(x)).strip() if isinstance(x, str) else x)

    # Normalize team names
    df = df.applymap(normalize_team_name)

    # Drop rows with all NaNs
    df = df.dropna(how='all').reset_index(drop=True)

    return df

def get_today_mlb_matchups_df(chromedriver_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--ignore-certificate-errors")
    chrome_options.add_argument("--proxy-server=http://47.251.122.81:8888")
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    try:
        url = "https://www.usatoday.com/story/sports/mlb/2025/04/28/mlb-games-today-schedule-times-tv-04282025/83316650007/"
        driver.get(url)

        wait = WebDriverWait(driver, 20)

        # Wait until at least one matchup is present
        wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "gnt_ar_b_ul_li")))
        
        # Find all matchup elements
        matchup_elements = driver.find_elements(By.CLASS_NAME, "gnt_ar_b_ul_li")
        
        matchups = []
        for element in matchup_elements:
            text = element.text.strip()
            if text and ' at ' in text:
                team1 = text.split(' at ')[0].strip()
                team2 = text.split(' at ')[1].split(',')[0].strip()
                matchup = f"{team1} vs {team2}"
                matchups.append(matchup)

        print(f"Found {len(matchups)} matchups.")

        # Create a DataFrame with matchups as the index
        df_matchups = pd.DataFrame(matchups)
        return df_matchups

    except Exception as e:
        print(f"Error while fetching matchups: {e}")
        return pd.DataFrame()

def pull_from_cbs(chromedriver_path, driver):

    print("Navigating to CBS Sports...")

    # Navigate to CBS Sports MLB expert picks
    url = "https://www.cbssports.com/mlb/expert-picks/"
    driver.get(url)

    # Wait for expert picks section to be visible
    wait = WebDriverWait(driver, 10)  # Wait up to 10 seconds
    try:
        wait.until(EC.presence_of_element_located((By.XPATH, "//*[@id='content']")))
        print("Page loaded successfully, scraping expert picks...")
    except:
        print("Timed out waiting for expert picks to load.")
        driver.quit()
        return None  # Return None if there is an issue

    # Find all expert spread elements dynamically
    expert_spreads = []

    # General XPath for expert spreads
    try:
        # This XPath looks for any div elements that match expert spread patterns
        experts = driver.find_elements(By.XPATH, "//div[contains(@class, 'expert-spread')]")  # Adjust the class name if needed
        print(f"Found {len(experts)} expert spreads.")

        for expert in experts:
            try:
                spread = expert.text.strip()  # Extract the text from the element
                expert_spreads.append(spread)
            except Exception as e:
                print(f"Error extracting expert spread: {e}")
    except Exception as e:
        print(f"Error scraping expert spreads: {e}")

    # If we found expert spreads, create a DataFrame
    if expert_spreads:
        df = pd.DataFrame(expert_spreads, columns=["CBS"])
        return df
    else:
        print("No expert spreads found.")
        return None

def pull_from_pickswise(chromedriver_path, driver):

    print("Navigating to Pickswise...")
    driver.get("https://www.pickswise.com/mlb/picks/")

    picks = []
    wait = WebDriverWait(driver, 20)

    try:
        wait.until(EC.presence_of_element_located(
            (By.CLASS_NAME, "SelectionInfo_outcome__1i6jL")
        ))
        print("Page loaded successfully, scraping Pickswise picks...")

        pick_elements = driver.find_elements(By.CLASS_NAME, "SelectionInfo_outcome__1i6jL")

        for el in pick_elements:
            text = el.text.strip()
            if text:
                picks.append(text)

    except Exception as e:
        print(f"Error scraping Pickswise: {e}")

    if picks:
        print(f"Found {len(picks)} Pickswise picks.")
        df_pickswise = pd.DataFrame(picks, columns=["Pickswise"])
        return df_pickswise
    else:
        print("No picks found from Pickswise.")
        return None
    
def pull_from_betfirm(chromedriver_path, driver):

    print("Navigating to BetFirm...")
    driver.get("https://www.betfirm.com/free-baseball-picks/")

    picks = []
    wait = WebDriverWait(driver, 20)

    try:
        wait.until(EC.presence_of_element_located(
            (By.CLASS_NAME, "pick-result-success")
        ))
        print("Page loaded successfully, scraping BetFirm picks...")

        pick_elements = driver.find_elements(By.CLASS_NAME, "pick-result-success")

        for el in pick_elements:
            text = el.text.strip()
            if text:
                picks.append(text)

    except Exception as e:
        print(f"Error scraping BetFirm: {e}")

    if picks:
        print(f"Found {len(picks)} BetFirm picks.")
        df_betfirm = pd.DataFrame(picks, columns=["BetFirm"])
        return df_betfirm
    else:
        print("No picks found from BetFirm.")
        return None
    
def pull_from_wagertalk(chromedriver_path, driver):

    print("Navigating to WagerTalk...")
    driver.get("https://www.wagertalk.com/free-sports-picks/mlb")

    picks = []
    wait = WebDriverWait(driver, 20)

    try:
        wait.until(EC.presence_of_all_elements_located(
            (By.CLASS_NAME, "content-play")
        ))
        print("Page loaded successfully, scraping WagerTalk picks...")

        pick_elements = driver.find_elements(By.CLASS_NAME, "content-play")

        for el in pick_elements:
            try:
                # Extracting all the text and joining it into one string
                text = el.text.strip()
                if text:
                    picks.append(text)
            except Exception as e:
                print(f"Error extracting pick: {e}")

    except Exception as e:
        print(f"Error scraping WagerTalk: {e}")

    if picks:
        print(f"Found {len(picks)} WagerTalk picks.")
        df_wagertalk = pd.DataFrame(picks, columns=["WagerTalk"])
        return df_wagertalk
    else:
        print("No picks found from WagerTalk.")
        return None
    
def pull_from_sportsgambler(chromedriver_path, driver):

    print("Navigating to SportsGambler...")
    driver.get("https://www.sportsgambler.com/betting-tips/baseball/mlb-predictions/")

    picks = []
    wait = WebDriverWait(driver, 20)

    try:
        wait.until(EC.presence_of_element_located(
            (By.CLASS_NAME, "ourpred")
        ))
        print("Page loaded successfully, scraping SportsGambler picks...")

        # Find all "Betting Prediction" labels
        prediction_labels = driver.find_elements(By.CLASS_NAME, "ourpred")

        for label in prediction_labels:
            # Try to find the following sibling <span> containing the actual pick
            try:
                prediction_text = label.find_element(By.XPATH, "following-sibling::span[1]").text.strip()
                if prediction_text:
                    picks.append(prediction_text)
            except Exception as e:
                print(f"Error finding sibling span: {e}")

    except Exception as e:
        print(f"Error scraping SportsGambler: {e}")

    if picks:
        print(f"Found {len(picks)} SportsGambler picks.")
        df_sportsgambler = pd.DataFrame(picks, columns=["SportsGambler"])
        return df_sportsgambler
    else:
        print("No picks found from SportsGambler.")
        return None
    
def pull_from_olbg(chromedriver_path, driver):
    print("Navigating to OLBG...")
    driver.get("https://www.olbg.com/betting-tips/Baseball/12")

    picks = []
    wait = WebDriverWait(driver, 20)

    try:
        # Wait for the presence of the new h4 elements
        wait.until(EC.presence_of_all_elements_located(
            (By.CSS_SELECTOR, "h4.truncate.md\\:text-md.md\\:leading-snug.lg\\:text-\\[17px\\]")
        ))
        print("Page loaded successfully, scraping OLBG picks...")

        # Find all h4 elements with the specified class
        pick_elements = driver.find_elements(By.CSS_SELECTOR, "h4.truncate.md\\:text-md.md\\:leading-snug.lg\\:text-\\[17px\\]")

        for el in pick_elements:
            text = el.text.strip()
            if text and '@' not in text and "Improve" not in text:
                picks.append(text)

    except Exception as e:
        print(f"Error scraping OLBG: {e}")

    if picks:
        print(f"Found {len(picks)} OLBG picks (team only).")
        df_olbg = pd.DataFrame(picks, columns=["OLBG"])
        return df_olbg
    else:
        print("No team picks found from OLBG.")
        return None

# === MAIN COMBINE FUNCTION ===
def combine_dfs(chromedriver_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--ignore-certificate-errors")
    service = Service(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=chrome_options)

    sources = {
        'CBS': pull_from_cbs(chromedriver_path, driver),
        'Pickswise': pull_from_pickswise(chromedriver_path, driver),
        'BetFirm': pull_from_betfirm(chromedriver_path, driver),
        'WagerTalk': pull_from_wagertalk(chromedriver_path, driver),
        'SportsGambler': pull_from_sportsgambler(chromedriver_path, driver),
        'OLBG': pull_from_olbg(chromedriver_path, driver)
    }

    driver.quit()

    available_dfs = [df for df in sources.values() if df is not None]
    if not available_dfs:
        print("No data found from any source.")
        return None

    combined = pd.concat(available_dfs, axis=1)
    df_cleaned = post_process_combined_df(combined)
    df_cleaned.to_csv(r"C:\Users\Vince\master\Betting\mlb\Scrape\combined_picks_with_matchups.csv", index=False)
    print(df_cleaned)
    return df_cleaned

# Example usage
if __name__ == "__main__":
    chromedriver_path = r"C:\\Users\\Vince\\Downloads\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe"
    df_combined = combine_dfs(chromedriver_path)

