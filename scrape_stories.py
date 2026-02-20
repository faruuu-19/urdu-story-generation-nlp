import csv
import time
import re
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import undetected_chromedriver as uc

def setup_driver():
    """Setup Chrome driver with undetected_chromedriver"""
    options = uc.ChromeOptions()
    
    # Basic options
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--lang=en-US')
    
    # Optional: Run headless (comment out to see browser)
    # options.add_argument('--headless=new')
    
    # Initialize undetected chrome driver
    driver = uc.Chrome(options=options, version_main=144)
    
    return driver

def clean_filename(text):
    """Remove special characters from filename"""
    clean = re.sub(r'[^\w\s-]', '', text)
    clean = re.sub(r'[-\s]+', '_', clean)
    return clean.strip('_')

def scrape_story(driver, url):
    """Scrape a single story from the URL using Selenium"""
    try:
        print(f"  Loading URL: {url}")
        driver.get(url)
        
        # Wait longer for page to fully load
        time.sleep(5)
        
        # Wait for main content
        wait = WebDriverWait(driver, 15)
        try:
            wait.until(EC.presence_of_element_located((By.CLASS_NAME, "txt_detail")))
        except TimeoutException:
            print("  ✗ Content not loaded in time")
            return None
        
        data = {}
        
        # Extract Title
        try:
            title_element = driver.find_element(By.CSS_SELECTOR, 'h1.phead')
            data['title'] = title_element.text.strip()
            print(f"  Found title: {data['title'][:50]}...")
        except NoSuchElementException:
            print("  ⚠ Title not found")
            data['title'] = ''
        
        # Extract Urdu Title
        try:
            urdu_title_element = driver.find_element(By.CSS_SELECTOR, 'h2.urdu.fs24')
            data['urdu_title'] = urdu_title_element.text.strip()
            print(f"  Found Urdu title: {data['urdu_title'][:30]}...")
        except NoSuchElementException:
            print("  ⚠ Urdu title not found")
            data['urdu_title'] = ''
        
        # Extract Summary
        try:
            summary_element = driver.find_element(By.CSS_SELECTOR, 'p.urdu.fs20')
            data['summary'] = summary_element.text.strip()
        except NoSuchElementException:
            data['summary'] = ''
        
        # Extract Date
        try:
            date_element = driver.find_element(By.CSS_SELECTOR, 'p.art_info_bar')
            data['date'] = date_element.text.strip()
        except NoSuchElementException:
            data['date'] = ''
        
        # Extract Main Content
        try:
            content_div = driver.find_element(By.CSS_SELECTOR, 'div.txt_detail')
            
            # Get all text content
            full_text = content_div.text
            
            # Clean up the text
            lines = full_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                line = line.strip()
                # Skip ads, empty lines, and "continue reading" markers
                if line and '(جاری ہے)' not in line and 'googletag' not in line and len(line) > 5:
                    cleaned_lines.append(line)
            
            data['content'] = '\n\n'.join(cleaned_lines)
            print(f"  Found content: {len(data['content'])} characters")
            
        except NoSuchElementException:
            print("  ✗ Content div not found")
            data['content'] = ''
        
        return data
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)[:100]}")
        return None

def format_story_text(data):
    """Format the story data into readable text"""
    text = f"""Title: {data['title']}
Urdu Title: {data['urdu_title']}
Date: {data['date']}

Summary:
{data['summary']}

Story:
{data['content']}
"""
    return text

def main():
    # Create output directory
    output_dir = Path('stories')
    output_dir.mkdir(exist_ok=True)
    
    # Read CSV file
    csv_file = 'urdu_stories_all.csv'
    
    driver = None
    
    try:
        print("Setting up browser (this may take a moment)...")
        driver = setup_driver()
        print("✓ Browser ready\n")
        
        # Read CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            stories = list(reader)
        
        print(f"Found {len(stories)} stories to scrape\n")
        
        success_count = 0
        fail_count = 0
        
        # Process each story
        for idx, row in enumerate(stories, 1):
            if len(row) < 4:
                print(f"[{idx}/{len(stories)}] Skipping invalid row")
                continue
            
            category, url, urdu_title, english_title = row
            
            print(f"\n[{idx}/{len(stories)}] Scraping: {english_title}")
            
            # Scrape the story
            story_data = scrape_story(driver, url)
            
            if story_data and story_data['content']:
                # Format the story text
                story_text = format_story_text(story_data)
                
                # Create filename: story_01_Barr_e_Azam_Antarctica.txt
                filename = f"story_{idx:02d}_{clean_filename(english_title)}.txt"
                filepath = output_dir / filename
                
                # Save to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(story_text)
                
                print(f"  ✓ Saved to: {filename}")
                success_count += 1
            else:
                print(f"  ✗ Failed to scrape or no content found")
                fail_count += 1
            
            # Longer delay between requests
            if idx < len(stories):
                delay = 5 + (idx % 3)
                print(f"  Waiting {delay}s before next request...")
                time.sleep(delay)
        
        print(f"\n{'='*50}")
        print(f"✓ Completed!")
        print(f"  Success: {success_count}")
        print(f"  Failed: {fail_count}")
        print(f"  Stories saved in '{output_dir}' directory")
        
    except FileNotFoundError:
        print(f"Error: Could not find '{csv_file}'")
        print("Make sure your CSV file is in the same directory as this script")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if driver:
            print("\nClosing browser...")
            driver.quit()

if __name__ == "__main__":
    main()