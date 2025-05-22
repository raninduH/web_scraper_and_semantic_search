'''
Web scraper to extract textual content from a website, chunk it by topic (heuristically),
and save it to a JSON file.
'''
import requests
from bs4 import BeautifulSoup
import json
from urllib.parse import urljoin, urlparse
import time

# --- Configuration ---
# !!! USER: Please replace this with the starting URL of the website you want to scrape !!!
START_URL = "http://example.com" # Replace this!

# Tags that might contain primary content. Adjust if needed for specific sites.
CONTENT_SELECTORS = ['article', 'main', 'section', 'div.content', 'div.post', 'div.entry']
# If the above selectors don't yield good results, the script will fall back to <p> tags.
HEADING_TAGS = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# Safety limit to prevent overly long scrapes or getting stuck.
MAX_PAGES_TO_SCRAPE = 50
# Delay between requests to be polite to the server.
REQUEST_DELAY_SECONDS = 1

def get_website_name(url):
    """Extracts a usable filename component (e.g., example_com) from a URL."""
    try:
        netloc = urlparse(url).netloc
        return netloc.replace('www.', '').replace('.', '_')
    except Exception:
        return "unknown_website"

def fetch_page(url):
    """Fetches the content of a URL with a user-agent and timeout."""
    try:
        # Using a common user-agent can help avoid being blocked by some sites.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, timeout=15, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def find_internal_links(soup, base_url, current_page_url):
    """Finds all unique internal links on a page, ensuring they are absolute."""
    links = set()
    base_domain = urlparse(base_url).netloc
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(current_page_url, href) # Handles relative URLs
        parsed_full_url = urlparse(full_url)
        
        # Ensure it's an HTTP/HTTPS link and belongs to the same domain
        if parsed_full_url.scheme in ['http', 'https'] and parsed_full_url.netloc == base_domain:
            clean_url = parsed_full_url._replace(query='', fragment='').geturl() # Remove query params and fragments
            links.add(clean_url)
    return list(links)

def get_topic_for_element(element, default_topic="General Content"):
    """
    Tries to find a heading associated with the element to determine its topic.
    Searches for preceding sibling headings or parent's headings.
    """
    # Look for preceding sibling headings that are actual heading tags
    current_element = element
    for _ in range(5): # Check up to 5 previous siblings
        prev_sibling = current_element.find_previous_sibling()
        if not prev_sibling: break
        if prev_sibling.name in HEADING_TAGS:
            return prev_sibling.get_text(strip=True)
        current_element = prev_sibling

    # Look for headings in parent elements up the tree
    parent = element.parent
    for _ in range(5): # Check up to 5 levels of parents
        if not parent or parent.name == 'body': break
        # Find the closest heading tag within this parent that appears before the element
        # This is complex to do perfectly without knowing element order, so we simplify:
        # Find any heading in the parent as a fallback topic for this section.
        found_heading = parent.find(HEADING_TAGS)
        if found_heading:
            return found_heading.get_text(strip=True)
        parent = parent.parent
        
    return default_topic

def extract_chunks_from_page(soup, page_url, website_name_meta, page_number_counter):
    """
    Extracts textual content from the page and breaks it into chunks.
    Each chunk is associated with a topic, heuristically determined from headings.
    """
    chunks = []
    page_title = soup.title.string.strip() if soup.title else "Untitled Page"

    # Attempt to find main content blocks first
    main_content_area = None
    for selector in CONTENT_SELECTORS:
        if soup.select_one(selector):
            main_content_area = soup.select_one(selector)
            break
    
    target_elements_source = main_content_area if main_content_area else soup.body
    if not target_elements_source: return []

    # Process paragraphs within the identified content area or body
    # This aims for finer-grained chunking.
    elements_to_process = target_elements_source.find_all('p')

    # If no paragraphs found in main content (or body), try to get text from larger semantic blocks
    if not elements_to_process and main_content_area:
        elements_to_process = [main_content_area] # Treat the whole block as one chunk source
    elif not elements_to_process: # Still no paragraphs, and no specific main_content_area
        # Fallback: get all text from body if no p tags and no specific content selectors worked
        body_text = soup.body.get_text(separator='\n', strip=True) if soup.body else ""
        if body_text:
            chunks.append({
                "content": body_text,
                "meta_data": {
                    "web_site_name": website_name_meta,
                    "web_page_url": page_url,
                    "topic": page_title, # Use page title as topic for the whole page
                    "page_number": page_number_counter
                }
            })
        return chunks

    for element in elements_to_process:
        text = element.get_text(separator=' ', strip=True)
        if not text or len(text.split()) < 5: # Skip very short or empty texts
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

def main_scraper(start_url):
    """Main function to crawl and scrape the website."""
    if not start_url or not (start_url.startswith("http://") or start_url.startswith("https://")):
        print("Invalid START_URL. Please provide a valid http/https URL in the script.")
        return

    website_name_file = get_website_name(start_url) # For filename
    website_name_meta = urlparse(start_url).netloc # For metadata

    if (website_name_file == "example_com" or website_name_file == "unknown_website") and "example.com" in start_url:
        print("Please update the START_URL in the script with the actual website you want to scrape.")
        return
    
    print(f"Starting scrape for website: {website_name_meta} (from URL: {start_url})")

    queue = [start_url]
    visited_urls = set()
    all_scraped_chunks = []
    page_counter = 0 # This will be the unique page number for metadata

    while queue and page_counter < MAX_PAGES_TO_SCRAPE:
        current_url = queue.pop(0)

        if current_url in visited_urls:
            continue
        
        # Ensure we stay on the same domain as the start_url
        if urlparse(current_url).netloc != urlparse(start_url).netloc:
            print(f"Skipping external or subdomain link: {current_url}")
            continue

        print(f"Scraping ({page_counter + 1}/{MAX_PAGES_TO_SCRAPE}): {current_url}")
        visited_urls.add(current_url)
        page_counter += 1 # Increment for each new page processed

        html_content = fetch_page(current_url)
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
            
        time.sleep(REQUEST_DELAY_SECONDS) # Be polite to the server
            
    if not all_scraped_chunks:
        print("No data was scraped.")
        return

    save_to_json(all_scraped_chunks, website_name_file)
    print(f"Scraping finished. Visited {page_counter} pages. Total chunks: {len(all_scraped_chunks)}")

if __name__ == "__main__":
    if START_URL == "http://example.com" or not START_URL:
        print("ERROR: Please edit the scraper.py script and set the START_URL variable to the website you wish to scrape.")
    else:
        main_scraper(START_URL)
