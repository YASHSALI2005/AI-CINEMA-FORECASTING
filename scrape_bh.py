import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
import os

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def get_movie_links(year):
    # Using the more robust filterbycountry URL pattern provided by the user
    url = f"https://www.bollywoodhungama.com/box-office-collections/filterbycountry/IND/{year}"
    print(f"🚀 Fetching movie list for {year} via {url}...")
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"❌ Failed to fetch year {year}: Status {response.status_code}")
            return {}
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', href=True)
        
        movie_links = {}
        for a in links:
            href = a['href']
            # Pick links that go to specific movie box office pages
            if '/movie/' in href and '/box-office/' in href:
                # Normalize URL by removing fragments
                clean_href = href.split('#')[0].rstrip('/') + '/'
                
                title = a.text.strip()
                # Exclude links that are just amounts (figures) or ranks
                if title and not re.match(r'^[\d.,\u20b9\s]+$', title):
                    movie_links[title] = clean_href
        
        print(f"✅ Found {len(movie_links)} movies for {year}")
        return movie_links
    except Exception as e:
        print(f"❌ Error fetching year {year}: {e}")
        return {}

def generate_bh_slug(movie_name):
    # Normalize movie name to generate potential URL slug
    slug = movie_name.lower()
    # Remove things in brackets
    slug = re.sub(r'\(.*?\)', '', slug)
    # Remove special characters
    slug = re.sub(r'[^a-z0-9\s]', '', slug)
    # Replace spaces with hyphens
    slug = slug.strip().replace('  ', ' ').replace(' ', '-')
    return slug

def scrape_by_name(movie_name):
    slug = generate_bh_slug(movie_name)
    url = f"https://www.bollywoodhungama.com/movie/{slug}/box-office/"
    print(f"🎯 Attempting targeted scrape for: {movie_name} at {url}")
    return scrape_movie_details(movie_name, url)

def scrape_movie_details(movie_name, url):
    """
    Scrapes detailed box office tables for a specific movie.
    """
    print(f"🔍 Scraping details for: {movie_name}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            print(f"⚠️ Page not found ({response.status_code}) for: {url}")
            return None
        
        soup = BeautifulSoup(response.text, 'html.parser')
        tables = soup.find_all('table')
        
        details = {
            'original_name': movie_name,
            'bh_url': url,
            'summary': {},
            'daily': [],
            'cinemas': {}
        }
        
        if not tables:
            print(f"⚠️ No tables found for: {movie_name}")
            return details

        # Table 0: Summary
        summary_table = tables[0]
        for tr in summary_table.find_all('tr'):
            cols = [td.text.strip() for td in tr.find_all(['td', 'th'])]
            if len(cols) == 2:
                key = cols[0].replace('End of ', '').replace(' Collection', '').lower().replace(' ', '_')
                details['summary'][key] = cols[1]

        # Table 1: Day Wise
        if len(tables) > 1:
            day_wise_table = tables[1]
            rows = day_wise_table.find_all('tr')
            if rows:
                headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
                if 'Day' in headers and 'Date' in headers:
                    for tr in rows[1:]:
                        cols = [td.text.strip() for td in tr.find_all(['td'])]
                        if len(cols) >= 3:
                            details['daily'].append({
                                'day': cols[0],
                                'date': cols[1],
                                'amount': cols[2]
                            })

        # Table 5: Cinema Chain (Checking if it's the right table)
        if len(tables) > 5:
            cinema_table = tables[5]
            rows = cinema_table.find_all('tr')
            if rows:
                headers = [th.text.strip() for th in rows[0].find_all(['th', 'td'])]
                if 'Cinema' in headers and 'Collection' in headers:
                    for tr in rows[1:]:
                        cols = [td.text.strip() for td in tr.find_all(['td'])]
                        if len(cols) >= 2:
                            details['cinemas'][cols[0]] = cols[1]
        
        return details
    except Exception as e:
        print(f"❌ Error scraping {movie_name}: {e}")
        return None

def run_scraper(years=[2024, 2025, 2026], output_file='bh_box_office_data.json'):
    all_data = []
    
    # Load existing if available to avoid redundant work
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_data = json.load(f)
    
    scraped_urls = {m['bh_url'] for m in all_data}
    
    for year in years:
        movie_links = get_movie_links(year)
        
        for name, url in movie_links.items():
            if url in scraped_urls:
                continue
                
            details = scrape_movie_details(name, url)
            if details:
                all_data.append(details)
                scraped_urls.add(url)
                
                # Save after each movie for safety
                with open(output_file, 'w') as f:
                    json.dump(all_data, f, indent=4)
                    
            time.sleep(1) # Polite scraping
            
    print(f"🎉 Successfully scraped {len(all_data)} movies. Saved to {output_file}")

if __name__ == "__main__":
    run_scraper()
