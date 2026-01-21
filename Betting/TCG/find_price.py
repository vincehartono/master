import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
import time

# Path to ChromeDriver
chrome_driver_path = r"C:\Users\Vince\Downloads\chromedriver-win64\chromedriver-win64\chromedriver.exe"

# Load CSV and read the necessary columns
csv_path = r"C:\Users\Vince\master\Betting\TCG\inventory.csv"
df = pd.read_csv(csv_path)

# Ensure 'Price Now', 'Name', and 'Check Price' columns exist
if 'Price Now' not in df.columns:
    df['Price Now'] = None
if 'Name' not in df.columns:
    df['Name'] = None
if 'Check Price' not in df.columns:
    df['Check Price'] = None

# Set up Selenium
service = Service(executable_path=chrome_driver_path)
options = webdriver.ChromeOptions()
# proxy = "47.251.122.81:8888"
# options.add_argument(f'--proxy-server=http://{proxy}')
driver = webdriver.Chrome(service=service, options=options)

# Visit each URL and scrape the product name and price
for index, row in df[df['Check Price'] == 'Y'].iterrows():
    url = row['URL']
    driver.get(url)
    time.sleep(5)  # Give the page time to load

    try:
        # Scrape product name
        name_element = driver.find_element(By.CLASS_NAME, "product-details__name")
        product_name = name_element.text.strip()
        df.at[index, 'Name'] = product_name
        
        # Scrape price
        price_element = driver.find_element(By.CLASS_NAME, "spotlight__price")
        price_text = price_element.text.strip()
        
        # Convert price_text to float by removing '$'
        price_text = price_text.replace('$', '').strip()
        price = float(price_text)
        
        df.at[index, 'Price Now'] = price
        
        print(f"{url} → {product_name} → ${price}")
    except Exception as e:
        print(f"Error processing {url}: {e}")
        df.at[index, 'Name'] = "Error"
        df.at[index, 'Price Now'] = "Error"

# Quit browser
driver.quit()

# Save updated CSV file
df.to_csv(csv_path, index=False)
print("✅ Updated CSV file saved.")
