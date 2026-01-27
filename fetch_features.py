import pandas as pd
import requests
import time
import csv
import os
import urllib3
from concurrent.futures import ThreadPoolExecutor, as_completed
from sqlalchemy import create_engine
from urllib.parse import quote_plus
from tqdm import tqdm

# ==========================================
# 0. NETWORK FIX (The "Battering Ram")
# ==========================================
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Force-clear proxies
for var in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy', 'ALL_PROXY']:
    if var in os.environ:
        del os.environ[var]

# ==========================================
# 1. CONFIGURATION
# ==========================================
TMDB_API_KEY = "cfc42745859368e3d9c8252b457b09fb"
DB_PASSWORD  = "Yashsali@*2005"        # <--- PASTE YOUR SQL PASSWORD HERE
DB_USER      = "root"
DB_HOST      = "localhost"
DB_NAME      = "movie"
OUTPUT_FILE  = "movie_features_safe.csv"

# ==========================================
# 2. WORKER FUNCTION
# ==========================================
def get_movie_details(movie_name):
    if not movie_name: return None
    clean_name = movie_name.split('(')[0].strip()
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    with requests.Session() as session:
        try:
            # 1. Search
            search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_name}"
            response = session.get(search_url, headers=headers, timeout=10, verify=False)
            
            if response.status_code != 200: return None
            
            data = response.json()
            if not data.get('results'): return None
                
            top_result = data['results'][0]
            movie_id = top_result['id']
            
            # 2. Details
            details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&append_to_response=credits"
            details = session.get(details_url, headers=headers, timeout=10, verify=False).json()
            
            return {
                'original_name': movie_name,
                'tmdb_title': top_result['title'],
                'genres': "|".join([g['name'] for g in details.get('genres', [])]),
                'budget': details.get('budget', 0),
                'revenue': details.get('revenue', 0),
                'runtime': details.get('runtime', 0),
                'release_date': details.get('release_date', ''),
                'popularity': details.get('popularity', 0),
                'vote_count': details.get('vote_count', 0),
                'vote_average': details.get('vote_average', 0),
                'director': next((crew['name'] for crew in details.get('credits', {}).get('crew', []) if crew['job'] == 'Director'), "Unknown"),
                'top_cast': "|".join([cast['name'] for cast in details.get('credits', {}).get('cast', [])[:3]])
            }
        except Exception:
            return None

# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_pipeline():
    print("🚀 Connecting to DB...")
    encoded_password = quote_plus(DB_PASSWORD)
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
    
    df_movies = pd.read_sql("SELECT DISTINCT movie_name FROM m_grouped_transactions", engine)
    all_movies = df_movies['movie_name'].tolist()
    
    if os.path.exists(OUTPUT_FILE):
        try:
            existing_df = pd.read_csv(OUTPUT_FILE)
            existing_movies = set(existing_df['original_name'].tolist())
            movies_to_process = [m for m in all_movies if m not in existing_movies]
            print(f"🔄 Resuming... Found {len(existing_movies)} done. {len(movies_to_process)} left.")
        except:
            movies_to_process = all_movies
    else:
        movies_to_process = all_movies
        pd.DataFrame(columns=['original_name', 'tmdb_title', 'genres', 'budget', 'revenue', 
                              'runtime', 'release_date', 'popularity', 'vote_count', 
                              'vote_average', 'director', 'top_cast']).to_csv(OUTPUT_FILE, index=False)
        print(f"🆕 Starting fresh. {len(movies_to_process)} movies to go.")

    print("⚡ Starting REAL DATA Fetch...")

    with open(OUTPUT_FILE, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['original_name', 'tmdb_title', 'genres', 'budget', 
                                              'revenue', 'runtime', 'release_date', 'popularity', 
                                              'vote_count', 'vote_average', 'director', 'top_cast'])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_movie = {executor.submit(get_movie_details, m): m for m in movies_to_process}
            
            # Using tqdm to show progress bar
            for future in tqdm(as_completed(future_to_movie), total=len(movies_to_process)):
                result = future.result()
                if result:
                    writer.writerow(result)
                    f.flush()
                    
                    # ✅ VISUAL CONFIRMATION: Prints every single success
                    tqdm.write(f"✅ Fetched: {result['tmdb_title']}")

    print("\n🎉 DONE! Real Data Saved.")

if __name__ == "__main__":
    run_pipeline()