import random
import time
from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.common.exceptions import ElementClickInterceptedException, NoSuchElementException

# Generate random keywords for searching
def generate_random_keywords(n=8):
    keywords = [
        "python", "selenium", "automation", "technology", "weather", 
        "sports", "news", "entertainment", "science", "history", 
        "movies", "books", "travel", "food", "health", "music", 
        "art", "education", "games", "software", "hardware", "coding", 
        "AI", "machine learning", "space", "universe", "biology", 
        "physics", "chemistry", "mathematics", "fashion", "design", 
        "architecture", "finance", "economy", "stock market", "real estate", 
        "cars", "motorcycles", "aviation", "wildlife", "forests", 
        "oceans", "mountains", "deserts", "holidays", "events", 
        "workshops", "tutorials", "recipes", "languages", "literature", 
        "poetry", "politics", "government", "social media", "branding", 
        "marketing", "entrepreneurship", "startup", "business", "investments", 
        "cryptocurrency", "blockchain", "virtual reality", "augmented reality", 
        "photography", "videography", "fitness", "yoga", "meditation", 
        "mindfulness", "self-help", "mental health", "relationships", "parenting", 
        "pets", "gardening", "DIY", "crafts", "paintings", "sculptures", 
        "classic literature", "modern art", "philosophy", "astronomy", "geography", 
        "climate change", "sustainability", "renewable energy", "ethics", 
        "culture", "heritage", "traditions", "festivals", "urban life", 
        "rural life", "cosmetics", "skincare", "haircare", "robotics", 
        "nanotechnology", "cybersecurity", "programming languages", "cybernetics", 
        "quantum computing", "3D printing", "wearable technology", "genetics", 
        "gene therapy", "epigenetics", "neuroscience", "psychology", "sociology", 
        "anthropology", "paleontology", "archaeology", "linguistics", "ethnomusicology", 
        "traditional medicine", "alternative medicine", "global warming", "biodiversity", 
        "renewable resources", "ecotourism", "space exploration", "mars rover", 
        "planetary science", "deep space", "hubble telescope", "artificial intelligence ethics", 
        "AI applications", "neural networks", "big data", "data analytics", "blockchain applications", 
        "cryptocurrency trading", "NFTs", "web3", "cloud computing", "virtual meetings", 
        "remote work", "hybrid work", "electric vehicles", "autonomous cars", "biotechnology", 
        "smart cities", "e-commerce", "digital marketing", "content creation", "streaming platforms"
    ]
    return random.sample(keywords, n)

# Perform random searches on Bing
def perform_bing_searches(keywords):
    # Set up Edge WebDriver
    edge_service = Service(r"C:\Users\Vince\Downloads\edgedriver_win64\msedgedriver.exe")
    driver = webdriver.Edge(service=edge_service)

    # Open Bing
    driver.get("https://www.bing.com")
    print("[INFO] Opened Bing")

    try:
        for keyword in keywords:
            print(f"[INFO] Searching for keyword: {keyword}")
            
            try:
                # Locate the search bar
                time.sleep(5)
                search_box = driver.find_element(By.NAME, "q")
                search_box.clear()
                search_box.send_keys(keyword)
                search_box.send_keys(Keys.RETURN)  # Press Enter
                
                # Ensure search is submitted
                time.sleep(5 + random.uniform(1, 2))
                if driver.current_url == "https://www.bing.com":
                    print("[WARN] Search didn't submit via Enter key. Retrying using JS.")
                    driver.execute_script("document.querySelector('form').submit()")
                    time.sleep(5)
                    
            except NoSuchElementException:
                print("[ERROR] Search box not found. Skipping this keyword.")
                continue
            
            # Wait for search results to load
            print(f"[INFO] Successfully searched for: {keyword}")
            time.sleep(5 + random.uniform(1, 3))

            # Find all clickable links
            links = driver.find_elements(By.XPATH, "//a[@href]")
            valid_links = [link for link in links if link.is_displayed()]
            if valid_links:
                link_to_click = random.choice(valid_links)
                print(f"[INFO] Clicking a random link for: {keyword}")
                try:
                    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", link_to_click)
                    link_to_click.click()
                    time.sleep(10 + random.uniform(5, 10))
                    driver.back()
                except ElementClickInterceptedException:
                    print("[WARN] Link click intercepted, skipping.")

            # Scroll to simulate interaction
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(5 + random.uniform(2, 4))

    finally:
        print("[INFO] Closing browser...")
        driver.quit()

# Main function
if __name__ == "__main__":
    keywords = generate_random_keywords()
    perform_bing_searches(keywords)
