from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

def init_driver():
    options = Options()
    options.add_argument("--headless")  # Run Chrome in headless mode
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    return driver

def scrape_amazon(product_search_query):
    driver = init_driver()
    amazon_url = "https://www.amazon.com/"
    
    # Open Amazon and search for the product
    driver.get(amazon_url)
    
    try:
        # Wait for the search box to load
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "twotabsearchtextbox"))
        )
        search_box.send_keys(product_search_query)
        search_box.send_keys(Keys.RETURN)
    except Exception as e:
        print(f"Error locating search box: {e}")
        driver.quit()
        return []
    
    time.sleep(2)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    product_links = [a['href'] for a in soup.select('a.a-link-normal.s-no-outline')[:5]]  # Top 5 products
    
    products = []
    for link in product_links:
        product_url = f"https://www.amazon.com{link}"
        driver.get(product_url)
        time.sleep(2)
        
        product_soup = BeautifulSoup(driver.page_source, "html.parser")
        title = product_soup.select_one('#productTitle').get_text(strip=True) if product_soup.select_one('#productTitle') else "N/A"
        price = product_soup.select_one('.a-price .a-offscreen').get_text(strip=True) if product_soup.select_one('.a-price .a-offscreen') else "N/A"
        rating = product_soup.select_one('.a-icon-alt').get_text(strip=True) if product_soup.select_one('.a-icon-alt') else "N/A"
        description = product_soup.select_one('#productDescription').get_text(strip=True) if product_soup.select_one('#productDescription') else "N/A"
        
        product_data = {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Description": description,
            "URL": product_url
        }
        products.append(product_data)
    
    driver.quit()
    return products


# Function to save data to Excel
def save_to_excel(data, file_name):
    df = pd.DataFrame(data)
    df.to_excel(file_name, index=False)
    print(f"Data successfully saved to {file_name}")

# Main function to trigger scraping and saving
if __name__ == "__main__":
    query = "Protein Powder"
    file_name = "amazon_products.xlsx"
    
    print("Scraping Amazon for products...")
    product_data = scrape_amazon(query)
    
    print("Saving data to Excel...")
    save_to_excel(product_data, file_name)
    print("Process complete!")
