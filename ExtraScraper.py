'''
Web scraper using Selenium to extract textual content from dynamic websites (handling JavaScript),
chunk it by topic (heuristically), and save it to a JSON file.
'''
import time
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# --- Configuration ---
# !!! USER: Please replace this with the starting URL of the website you want to scrape !!!
START_URL = "https://docs.crewai.com"  # Replace this!

# !!! USER: Update this path if chromedriver.exe is not in your PATH !!!
# Example: CHROME_DRIVER_PATH = "C:/path/to/your/chromedriver.exe"
CHROME_DRIVER_PATH = r"chromedriver.exe" # Set to None if chromedriver is in PATH

# Tags that might contain primary content. Adjust if needed for specific sites.
CONTENT_SELECTORS = ['article', 'main', 'section', 'div.content', 'div.post', 'div.entry']
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

MAX_PAGES_TO_SCRAPE = 25 # Reduced default for potentially slower Selenium scrapes
REQUEST_DELAY_SECONDS = 2 # Selenium interactions can take longer
PAGE_LOAD_TIMEOUT_SECONDS = 30 # Max time to wait for a page to load

def get_website_name(url):
    """Extracts a usable filename component (e.g., example_com) from a URL."""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace('www.', '').replace('.', '_')
    except Exception:
        return "unknown_website"

def init_driver():
    """Initializes the Selenium WebDriver."""
    options = webdriver.ChromeOptions()
    options.add_argument("--headless") # Run Chrome in headless mode (no GUI)
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    try:
        if CHROME_DRIVER_PATH:
            service = ChromeService(executable_path=CHROME_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
        else:
            # Assumes chromedriver is in PATH
            driver = webdriver.Chrome(options=options)
        return driver
    except WebDriverException as e:
        print(f"Error initializing WebDriver: {e}")
        print("Please ensure ChromeDriver is installed and its path is correctly configured (either in PATH or via CHROME_DRIVER_PATH variable in the script).")
        return None

def fetch_page_with_selenium(driver, url):
    """Fetches the page content using Selenium, allowing JavaScript to render."""
    try:
        driver.get(url)
        # Wait for a basic element like body to be present, or a more specific one if known
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT_SECONDS).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        # You might want to add a small explicit wait if content loads very late
        # time.sleep(3) # Uncomment and adjust if necessary
        return driver.page_source
    except TimeoutException:
        print(f"Timeout loading page: {url}")
        return None
    except WebDriverException as e:
        print(f"WebDriver error fetching {url}: {e}")
        return None

def find_internal_links(soup, base_url, current_page_url):
    """Finds all unique internal links on a page, ensuring they are absolute."""
    links = set()
    base_domain = urlparse(base_url).netloc
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(current_page_url, href)
        parsed_full_url = urlparse(full_url)
        
        if parsed_full_url.scheme in ['http', 'https'] and parsed_full_url.netloc == base_domain:
            clean_url = parsed_full_url._replace(query='', fragment='').geturl()
            links.add(clean_url)
    return list(links)

def get_topic_for_element(element, default_topic="General Content"):
    """
    Tries to find a heading associated with the element to determine its topic.
    Searches for preceding sibling headings or parent's headings.
    """
    current_element = element
    for _ in range(5):
        prev_sibling = current_element.find_previous_sibling()
        if not prev_sibling: break
        if prev_sibling.name in HEADING_TAGS:
            return prev_sibling.get_text(strip=True)
        current_element = prev_sibling

    parent = element.parent
    for _ in range(5):
        if not parent or parent.name == 'body': break
        found_heading = parent.find(HEADING_TAGS)
        if found_heading:
            return found_heading.get_text(strip=True)
        parent = parent.parent
    return default_topic

def extract_chunks_from_page(soup, page_url, website_name_meta, page_number_counter):
    """
    Extracts textual content from the page and breaks it into chunks.
    """
    chunks = []
    page_title = soup.title.string.strip() if soup.title else "Untitled Page"

    main_content_area = None
    for selector in CONTENT_SELECTORS:
        # BeautifulSoup's select_one can take CSS selectors directly
        if soup.select_one(selector):
            main_content_area = soup.select_one(selector)
            break
    
    target_elements_source = main_content_area if main_content_area else soup.body
    if not target_elements_source: return []

    elements_to_process = target_elements_source.find_all('p')

    if not elements_to_process and main_content_area:
        elements_to_process = [main_content_area]
    elif not elements_to_process:
        body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        if body_text:
            chunks.append({
                "content": body_text,
                "meta_data": {
                    "web_site_name": website_name_meta,
                    "web_page_url": page_url,
                    "topic": page_title,
                    "page_number": page_number_counter
                }
            })
        return chunks

    for element in elements_to_process:
        text = element.get_text(separator=' ', strip=True)
        if not text or len(text.split()) < 5:
            continue
        topic = get_topic_for_element(element, page_title)
        chunks.append({
            "content": text,
            "meta_data": {
                "web_site_name": website_name_meta,
                "web_page_url": page_url,
                "topic": topic,
                "page_number": page_number_counter
            }
        })
    return chunks

def save_to_json(data, website_name_file):
    """Saves the data to a JSON file named websitename.json."""
    filename = f"{website_name_file}.json"
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"Data saved to {filename}")
    except IOError as e:
        print(f"Error saving data to JSON: {e}")

def main_selenium_scraper(start_url):
    """Main function to crawl and scrape the website using Selenium."""
    if not start_url or not (start_url.startswith("http://") or start_url.startswith("https://")):
        print("Invalid START_URL. Please provide a valid http/https URL.")
        return

    website_name_file = get_website_name(start_url)
    website_name_meta = urlparse(start_url).netloc

    if (website_name_file == "example_com" or website_name_file == "unknown_website") and "example.com" in start_url:
        print("Please update the START_URL in the script with the actual website you want to scrape.")
        return

    driver = init_driver()
    if not driver:
        return
    
    print(f"Starting Selenium scrape for website: {website_name_meta} (from URL: {start_url})")

    queue = [start_url]
    visited_urls = set()
    all_scraped_chunks = []
    page_counter = 0

    try:
        while queue and page_counter < MAX_PAGES_TO_SCRAPE:
            current_url = queue.pop(0)
            if current_url in visited_urls:
                continue
            
            if urlparse(current_url).netloc != urlparse(start_url).netloc:
                print(f"Skipping external or subdomain link: {current_url}")
                continue

            print(f"Scraping ({page_counter + 1}/{MAX_PAGES_TO_SCRAPE}): {current_url}")
            visited_urls.add(current_url)
            page_counter += 1

            html_content = fetch_page_with_selenium(driver, current_url)
            if html_content:
                soup = BeautifulSoup(html_content, 'html.parser')
                page_chunks = extract_chunks_from_page(soup, current_url, website_name_meta, page_counter)
                if page_chunks:
                    all_scraped_chunks.extend(page_chunks)
                else:
                    print(f"No content chunks extracted from {current_url}")

                internal_links = find_internal_links(soup, start_url, current_url)
                for link in internal_links:
                    if link not in visited_urls and link not in queue:
                        queue.append(link)
            else:
                print(f"Failed to fetch content for {current_url}")
            
            time.sleep(REQUEST_DELAY_SECONDS) # Be polite
    finally:
        if driver:
            driver.quit()
            
    if not all_scraped_chunks:
        print("No data was scraped.")
        return

    save_to_json(all_scraped_chunks, website_name_file)
    print(f"Scraping finished. Visited {page_counter} pages. Total chunks: {len(all_scraped_chunks)}")

if __name__ == "__main__":
    if START_URL == "http://example.com" or not START_URL:
        print("ERROR: Please edit the ExtraScraper.py script and set the START_URL variable.")
        print("Also, ensure CHROME_DRIVER_PATH is set correctly if chromedriver is not in your system PATH.")
    else:
        main_selenium_scraper(START_URL)
