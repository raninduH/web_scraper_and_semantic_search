import time
import json
import os
import re
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup, NavigableString, Tag
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

# --- Configuration ---
START_URL = "https://docs.crewai.com"  # Replace this!
CHROME_DRIVER_PATH = r"chromedriver.exe" # Set to None if chromedriver is in PATH

CONTENT_SELECTORS = ['article', 'main', 'section', 'div.content', 'div.post', 'div.entry', 'div.main-content'] # Added common class
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
# Selectors for code blocks. Add more specific ones if needed.
CODE_SELECTORS = ['pre', 'code', 'div.codeblock', 'div.highlight', 'div.prism-code'] 

MAX_PAGES_TO_SCRAPE = 25 
REQUEST_DELAY_SECONDS = 2 
PAGE_LOAD_TIMEOUT_SECONDS = 30 

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
    options.add_argument("--headless") 
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

    try:
        if CHROME_DRIVER_PATH:
            service = ChromeService(executable_path=CHROME_DRIVER_PATH)
            driver = webdriver.Chrome(service=service, options=options)
        else:
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
        WebDriverWait(driver, PAGE_LOAD_TIMEOUT_SECONDS).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
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
    # Check preceding siblings
    for _ in range(5): # Look back up to 5 siblings
        prev_sibling = current_element.find_previous_sibling()
        if not prev_sibling:
            break
        if prev_sibling.name in HEADING_TAGS:
            return prev_sibling.get_text(strip=True)
        # Check for headings within the previous sibling if it's a container
        if prev_sibling.find(HEADING_TAGS):
             heading = prev_sibling.find(HEADING_TAGS)
             if heading: return heading.get_text(strip=True)
        current_element = prev_sibling

    # Check parents
    parent = element.parent
    for _ in range(5): # Look up to 5 parent levels
        if not parent or parent.name == 'body':
            break
        # Check direct children headings of the parent that appear before the current element's original branch
        # This is complex; a simpler approach is to find any heading in the parent
        found_heading = parent.find(HEADING_TAGS) # Find first heading in parent
        if found_heading:
            # Check if this heading is an ancestor or closely related, not a sub-heading of a different section
            # This heuristic can be improved. For now, any heading in parent is a candidate.
            return found_heading.get_text(strip=True)
        parent = parent.parent
    return default_topic

def extract_code_from_element(code_element):
    """Extracts text from a code element, preserving line breaks."""
    # For <pre> tags, often the formatting is important.
    # For <code> tags, they might be inline or part of a block.
    # This tries to get text content, handling common structures.
    
    # If it's a pre tag, get_text with a separator is usually good.
    if code_element.name == 'pre':
        return code_element.get_text(separator='\\n', strip=False) # Keep leading/trailing whitespace for pre

    # For other code tags, iterate through NavigableString to build content
    # This helps with nested tags inside <code> like <span> for syntax highlighting
    lines = []
    for elem in code_element.descendants:
        if isinstance(elem, NavigableString):
            lines.append(str(elem))
        elif elem.name == 'br': # Handle <br> tags for line breaks
            lines.append('\\n')
    
    # Join lines and then strip leading/trailing whitespace from the whole block,
    # but preserve internal newlines.
    full_code = "".join(lines)
    return full_code.strip()


def extract_chunks_from_page(soup, url, web_site_name, page_counter):
    chunks = []
    current_topic = "Introduction"  # Default topic
    
    # Try to find a main content area to reduce noise, common selectors
    main_content = soup.find('main') or soup.find('article') or \
                   soup.find('div', class_=['content', 'main-content', 'post-content', 'article-body']) or \
                   soup
    
    # Find all relevant elements for chunking (paragraphs, headings, list items, code blocks)
    # elements_for_chunking = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'div'], recursive=True)
    # More robust: find all text-bearing elements and also specific code block containers
    elements_for_chunking = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'pre', 'code', 'div', NavigableString], recursive=True)


    for element in elements_for_chunking:
        if isinstance(element, NavigableString):
            text_content = str(element).strip()
            if text_content and not (isinstance(element.parent, Tag) and element.parent.name in ['script', 'style', 'pre', 'code']):
                 # Add NavigableString if it's not essentially empty and not part of already handled tags
                chunks.append({
                    "content": text_content,
                    "meta_data": {
                        "web_site_name": web_site_name,
                        "web_page_url": url,
                        "topic": current_topic,
                        "page_number": page_counter,
                        "type": "paragraph" # Or determine more specifically if possible
                    }
                })
            continue # Move to next element

        if not isinstance(element, Tag): # Should not happen if NavigableString is handled above
            continue

        # Skip script/style tags and their content early
        if element.name in ['script', 'style']:
            continue
            
        # Update topic if a heading is encountered
        if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            current_topic = element.get_text(separator=' ', strip=True) or current_topic
            # Add heading as a chunk as well
            text_content = element.get_text(separator=' ', strip=True)
            if text_content:
                chunks.append({
                    "content": text_content,
                    "meta_data": {
                        "web_site_name": web_site_name,
                        "web_page_url": url,
                        "topic": current_topic, # Use the new topic
                        "page_number": page_counter,
                        "type": "heading"
                    }
                })
            continue # Headings are processed, move to next element

        # --- Refined code block identification logic ---
        is_element_chosen_for_code_extraction = False
        
        # Priority 1: 'pre' tags
        if element.name == 'pre':
            is_element_chosen_for_code_extraction = True
        
        # Priority 2: Specified 'div' wrappers, but only if they don't contain a 'pre'
        elif element.name == 'div':
            matches_div_code_selector = False
            for selector_str in CODE_SELECTORS:
                if selector_str.startswith('div'):
                    try:
                        if element.matches(selector_str):
                            matches_div_code_selector = True
                            break
                    except (AttributeError, TypeError): # Fallback for .matches
                        parts = selector_str.split('.')
                        tag_name_from_selector = parts[0]
                        required_classes = parts[1:]
                        if element.name == tag_name_from_selector and \
                           (not required_classes or all(cls in element.get('class', []) for cls in required_classes)):
                            matches_div_code_selector = True
                            break
            
            if matches_div_code_selector:
                if not element.find('pre'): # Only if no 'pre' child, as 'pre' is preferred
                    is_element_chosen_for_code_extraction = True
        
        # Priority 3: 'code' tags, but only if not inside a 'pre' and 'code' is in CODE_SELECTORS
        elif element.name == 'code':
            is_code_tag_in_code_selectors = False
            for selector_str in CODE_SELECTORS: # Check if 'code' itself is a target selector
                if selector_str == 'code': # Simple name check for 'code' tag
                     is_code_tag_in_code_selectors = True
                     break
            
            if is_code_tag_in_code_selectors:
                parent = element.parent
                if not (isinstance(parent, Tag) and parent.name == 'pre'):
                    is_element_chosen_for_code_extraction = True
        # --- End of refined code block identification logic ---

        if is_element_chosen_for_code_extraction:
            code_content = extract_code_from_element(element)
            if code_content:
                chunks.append({
                    "content": code_content,
                    "meta_data": {
                        "web_site_name": web_site_name,
                        "web_page_url": url,
                        "topic": current_topic,
                        "page_number": page_counter,
                        "type": "code_block"
                    }
                })
        else:
            # Process as paragraph or other text element if not chosen as a code block
            # This applies to 'p', 'li', and 'div'/'code' tags not meeting the prioritized code block criteria.
            text_content = ""
            if element.name in ['p', 'li']: # h1-h6 are handled above
                text_content = element.get_text(separator=' ', strip=True)
            elif element.name == 'div': 
                # Handle text within divs that were not chosen as code blocks
                # Avoid extracting text from divs that are just containers for other processed elements (like 'pre' that was skipped here)
                is_likely_container_only = element.find(['p', 'li', 'ul', 'ol', 'table', 'pre', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']) is not None
                if not is_likely_container_only:
                    div_text_parts = []
                    for child_node in element.children:
                        if isinstance(child_node, NavigableString):
                            t = str(child_node).strip()
                            if t: div_text_parts.append(t)
                        elif isinstance(child_node, Tag) and child_node.name not in ['script', 'style', 'pre', 'code', 'p', 'li', 'ul', 'ol', 'table', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            # Get text from simple inline tags if any, avoiding block children handled elsewhere
                            if child_node.name in ['span', 'a', 'em', 'strong', 'b', 'i', 'u', 'font', 'mark', 'small', 'sub', 'sup', 'abbr', 'cite', 'dfn', 'kbd', 'q', 'samp', 'var', 'time', 'data', 'address']:
                                t = child_node.get_text(separator=' ', strip=True)
                                if t: div_text_parts.append(t)
                    if div_text_parts:
                        text_content = " ".join(div_text_parts)
            # Note: 'code' tags not chosen as code blocks (e.g., inline code inside <p>) will have their text
            # extracted as part of their parent's get_text(). We are not adding them as separate chunks here.

            if text_content:
                chunks.append({
                    "content": text_content,
                    "meta_data": {
                        "web_site_name": web_site_name,
                        "web_page_url": url,
                        "topic": current_topic,
                        "page_number": page_counter,
                        "type": "paragraph" 
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

    if (website_name_file == "example_com" or website_name_file == "unknown_website") and "example.com" not in start_url : # Make sure it's not the default example.com
        print("Please update the START_URL in the script with the actual website you want to scrape.")
        return

    driver = init_driver()
    if not driver:
        return
    
    print(f"Starting Enhanced Selenium scrape for website: {website_name_meta} (from URL: {start_url})")

    queue = [start_url]
    visited_urls = set()
    all_scraped_chunks = []
    page_counter = 0

    try:
        while queue and page_counter < MAX_PAGES_TO_SCRAPE:
            current_url = queue.pop(0)
            if current_url in visited_urls:
                continue
            
            # Simple domain check to avoid crawling external sites
            if urlparse(current_url).netloc != urlparse(start_url).netloc:
                print(f"Skipping external or subdomain link: {current_url}")
                continue

            print(f"Scraping ({page_counter + 1}/{MAX_PAGES_TO_SCRAPE}): {current_url}")
            visited_urls.add(current_url)
            
            html_content = fetch_page_with_selenium(driver, current_url)
            if html_content:
                page_number_counter = page_counter + 1 # Use a 1-based counter for page number meta
                soup = BeautifulSoup(html_content, 'html.parser')
                page_chunks = extract_chunks_from_page(soup, current_url, website_name_meta, page_number_counter)
                if page_chunks:
                    all_scraped_chunks.extend(page_chunks)
                    print(f"  Extracted {len(page_chunks)} chunks from {current_url}")
                else:
                    print(f"  No content chunks extracted from {current_url}")

                internal_links = find_internal_links(soup, start_url, current_url)
                for link in internal_links:
                    if link not in visited_urls and link not in queue:
                        queue.append(link)
            else:
                print(f"Failed to fetch content for {current_url}")
            
            page_counter += 1 # Increment after processing the page
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
    script_name = os.path.basename(__file__)

    main_selenium_scraper(START_URL)
