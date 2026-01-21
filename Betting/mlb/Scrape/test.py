from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd

import pandas as pd
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def get_today_mlb_matchups_df(chromedriver_path):
    chrome_options = Options()
    chrome_options.add_argument("--headless")  # Run headless (no GUI)
    chrome_options.add_argument("--ignore-certificate-errors")
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
        df_matchups = pd.DataFrame(index=matchups)
        return df_matchups

    except Exception as e:
        print(f"Error while fetching matchups: {e}")
        return pd.DataFrame()
    finally:
        driver.quit()

# Example usage:
chromedriver_path = r"C:\Users\Vince\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"
today_matchups = get_today_mlb_matchups(chromedriver_path)
print(today_matchups)