from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import time
from selenium.common.exceptions import TimeoutException

# Set the path to your ChromeDriver, making sure to include 'chromedriver.exe'
driver_path = 'C:\\Users\\Vince\\chromedriver-win64\\chromedriver-win64\\chromedriver.exe'

# Set up Chrome options to run in headless mode
chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--disable-gpu")  # Optional: Disables GPU usage (may increase performance)

# Set up the Service object for ChromeDriver
service = Service(driver_path)

# Launch Chrome browser in headless mode
driver = webdriver.Chrome(service=service, options=chrome_options)

# URL of the page you want to scrape
url = 'https://sportsbook.draftkings.com/nba-player-props?category=player-rebounds&subcategory=rebounds-o%2Fu'

# Open the URL
try:
    driver.get(url)
    # Wait for the page to load and wait for a specific element (adjust the class as needed)
    WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.CLASS_NAME, 'odds-container')))  # Change class name if necessary
    print("Page loaded and elements found")
except TimeoutException:
    print("Timed out while waiting for page to load")
    driver.quit()

# JavaScript to scroll the page to the bottom (to ensure all content loads)
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# Additional scroll to ensure content loads if required
time.sleep(3)  # Wait to let the page load more content, if necessary
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

# List to hold the scraped data
data = []

# Find all relevant player prop elements (make sure to adjust this selector if the class names are different)
player_elements = driver.find_elements(By.CLASS_NAME, 'odds-container')  #
