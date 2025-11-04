import pandas as pd
from bs4 import BeautifulSoup
from cloudscraper import create_scraper

def fetch_manchester_united_stats():
    url = 'https://fbref.com/en/squads/19538871/2024-2025/Manchester-United-Stats#all_stats_standard'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://fbref.com/en/squads/19538871/2024-2025/Manchester-United-Stats',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }
    
    print("Fetching Manchester United data...")
    
    # Use cloudscraper instead of requests
    scraper = create_scraper()
    response = scraper.get(url, headers=headers)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    tab_html = str(table)
    
    print("Parsing table...")
    df = pd.read_html(tab_html)[0]
    
    print("Saving to man_utd_output.csv...")
    df.to_csv('man_utd_stats.csv', index=False)
    
    print(f"Saved {len(df)} players!")
    return df


def fetch_chelsea_stats():
    url = 'https://fbref.com/en/squads/cff3d9bb/2024-2025/Chelsea-Stats'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://fbref.com/en/squads/cff3d9bb/2024-2025/Chelsea-Stats',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Pragma': 'no-cache',
        'Cache-Control': 'no-cache'
    }
    
    print("Fetching Chelsea data...")
    
    # Use cloudscraper instead of requests
    scraper = create_scraper()
    response = scraper.get(url, headers=headers)
    
    soup = BeautifulSoup(response.content, 'html.parser')
    table = soup.find('table')
    tab_html = str(table)
    
    print("Parsing table...")
    df = pd.read_html(tab_html)[0]
    
    print("Saving to chelsea_output.csv...")
    df.to_csv('chelsea_stats.csv', index=False)
    
    print(f"Saved {len(df)} players!")
    return df


# Call both functions
fetch_manchester_united_stats()
fetch_chelsea_stats()


