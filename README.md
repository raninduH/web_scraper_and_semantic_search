'''# Web Scrapers (`scraper.py` and `ExtraScraper.py`)

This project contains two Python scripts for scraping textual content from websites:

1.  `scraper.py`: Uses `requests` and `BeautifulSoup` to scrape static HTML content.
2.  `ExtraScraper.py`: Uses `Selenium` and `BeautifulSoup` to scrape websites that load content dynamically using JavaScript.

## Setup Instructions

### 1. Python and Pip

Ensure you have Python installed on your system. Pip (Python's package installer) usually comes with Python.

### 2. Install Required Python Libraries

Open your terminal (PowerShell) and run the following command to install the necessary libraries for both scrapers:

```powershell
pip install requests beautifulsoup4 selenium
```

### 3. Set Up ChromeDriver (for `ExtraScraper.py`)

`ExtraScraper.py` uses Selenium to control a Chrome browser, which requires ChromeDriver.

**a. Check Your Chrome Browser Version:**
   - Open Chrome.
   - Go to `chrome://settings/help` (type this into the address bar and press Enter).
   - Note down your Chrome version (e.g., `125.0.6422.60`).

**b. Download ChromeDriver:**
   - Go to the official ChromeDriver download page: [https://chromedriver.chromium.org/downloads](https://chromedriver.chromium.org/downloads) OR the newer Chrome for Testing availability dashboard: [https://googlechromelabs.github.io/chrome-for-testing/](https://googlechromelabs.github.io/chrome-for-testing/)
   - Find and download the ChromeDriver version that **matches your Chrome browser version**.
   - Download the zip file for Windows (e.g., `chromedriver_win32.zip` or `chromedriver-win64.zip`).

**c. Extract and Place `chromedriver.exe`:**
   - Extract the `chromedriver.exe` file from the downloaded zip archive.
   - You have a few options for making `chromedriver.exe` accessible to `ExtraScraper.py`:
      *   **Option 1 (Recommended for this project):** Place `chromedriver.exe` in the same directory as the `ExtraScraper.py` script (`c:\Users\raninduh\Documents\Web scraper\`). The script is configured to look here by default if the `CHROME_DRIVER_PATH` variable is not set.
      *   **Option 2 (Add to System PATH):**
          1.  Move `chromedriver.exe` to a preferred directory (e.g., `C:\WebDriver\bin`).
          2.  Add this directory to your system's PATH environment variable. This makes `chromedriver.exe` accessible from any command prompt.
      *   **Option 3 (Specify Path in Script):**
          1.  Open `ExtraScraper.py` in a text editor.
          2.  Find the line: `CHROME_DRIVER_PATH = None`
          3.  Change `None` to the full path of your `chromedriver.exe`, using double backslashes for Windows paths. For example:
              ```python
              CHROME_DRIVER_PATH = "C:\\path\\to\\your\\chromedriver.exe"
              ```

### 4. Configure the Target Website URL

For both `scraper.py` and `ExtraScraper.py`, you need to specify the website you want to scrape.

-   Open the script you intend to use (`scraper.py` or `ExtraScraper.py`).
-   Find the line:
    ```python
    START_URL = "http://example.com" # Replace this!
    ```
-   Replace `"http://example.com"` with the full starting URL of the website you wish to scrape.

## Running the Scrapers

Once the setup is complete and you have configured the `START_URL`:

1.  Navigate to the project directory in your terminal:
    ```powershell
    cd "c:\Users\raninduh\Documents\Web scraper"
    ```

2.  To run the static scraper:
    ```powershell
    python scraper.py
    ```

3.  To run the dynamic (JavaScript-enabled) scraper:
    ```powershell
    python ExtraScraper.py
    ```

The scraped data will be saved in a `.json` file named after the website (e.g., `example_com.json`) in the same directory.

## Notes

*   **Politeness:** The scrapers include a delay between requests (`REQUEST_DELAY_SECONDS`). Be mindful of the website's terms of service and robots.txt file. Scraping too aggressively can overload a server or get your IP address blocked.
*   **Dynamic Content:** `scraper.py` will not capture content loaded by JavaScript. Use `ExtraScraper.py` for such sites.
*   **Error Handling:** Basic error handling is included, but web scraping can be fragile due to website structure changes. You might need to adjust selectors (`CONTENT_SELECTORS`, `HEADING_TAGS`) in the scripts for optimal results on different websites.
*   **`MAX_PAGES_TO_SCRAPE`:** Both scripts have a limit on the number of pages to scrape to prevent overly long runs. You can adjust this variable if needed.
'''
