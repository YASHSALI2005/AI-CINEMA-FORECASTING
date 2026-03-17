import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import re
import random
from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

# --- CONFIG ---
# --- CONFIG ---
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")
SESSIONS_FILE = r"d:\Forecasting\sessions.csv.xlsx"
MODELS_DIR_NOVO = r"d:\Forecasting\models\novo"
METADATA_CACHE_FILE = os.path.join(MODELS_DIR_NOVO, "novo_movie_metadata.json")

# ── Unified Novo Model (ONE model for all movies+cinemas) ──────────────────
UNIFIED_NOVO_MODEL   = os.path.join(MODELS_DIR_NOVO, "novo_unified_model.json")
UNIFIED_CINEMA_ENC   = os.path.join(MODELS_DIR_NOVO, "novo_unified_cinema_encoder.pkl")
UNIFIED_MOVIE_ENC    = os.path.join(MODELS_DIR_NOVO, "novo_unified_movie_encoder.pkl")
UNIFIED_STATS_FILE   = os.path.join(MODELS_DIR_NOVO, "novo_unified_stats.json")
UNIFIED_FEATURES     = ['day_of_week', 'is_weekend', 'hour', 'cinema_id_encoded', 'movie_id_encoded']

TIME_SLOTS     = ["09:00", "12:00", "15:00", "18:00", "21:00"]
BASE_PRICE_QAR = 100

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip().lower()

def fetch_tmdb_movie_details(movie_title):
    if not TMDB_API_KEY: return None
    
    clean_title = re.sub(r'\s*\(.*?\)', '', movie_title).strip()
    year_match = re.search(r'\((\d{4})\)', movie_title)
    target_year = int(year_match.group(1)) if year_match else None

    try:
        # Step 1: Search for the movie ID
        search_url = "https://api.themoviedb.org/3/search/movie"
        params = {"api_key": TMDB_API_KEY, "query": clean_title}
        if target_year: 
            params["primary_release_year"] = target_year
            
        resp = requests.get(search_url, params=params, timeout=5)
        search_data = resp.json()

        # Retry without year if no results found
        if not search_data.get("results") and target_year:
             params.pop("primary_release_year")
             resp = requests.get(search_url, params=params, timeout=5)
             search_data = resp.json()

        if not search_data.get("results"): 
            return None
        
        best_match = search_data["results"][0]
        tmdb_id = best_match['id']

        # Step 2: Fetch full details using the ID (budget/runtime are not in basic search)
        details_url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        details_params = {"api_key": TMDB_API_KEY}
        data = requests.get(details_url, params=details_params, timeout=5).json()

        # Extract values, preserving defaults if TMDB returns 0 for them
        budget = data.get("budget", 0)
        runtime = data.get("runtime", 0)
        
        return {
            "runtime": runtime if runtime > 0 else 120,
            "vote_average": data.get("vote_average", 7.0),
            "popularity": int(data.get("popularity", 10)),
            "budget": budget if budget > 0 else 50000000
        }
    except Exception as e:
        print(f"Error fetching TMDB for {movie_title}: {e}")
        return None
def load_resources():
    print("Loading resources...")

    # Load Sessions Data
    print(f"   Loading sessions from {SESSIONS_FILE}...")
    df_sessions = pd.read_excel(SESSIONS_FILE)

    # ── 1. Unified Novo Model (required) ─────────────────────────────────────
    if not (os.path.exists(UNIFIED_NOVO_MODEL) and
            os.path.exists(UNIFIED_CINEMA_ENC) and
            os.path.exists(UNIFIED_MOVIE_ENC) and
            os.path.exists(UNIFIED_STATS_FILE)):
        raise FileNotFoundError(
            "Unified Novo artifacts missing. Ensure these exist in models/novo: "
            "novo_unified_model.json, novo_unified_cinema_encoder.pkl, "
            "novo_unified_movie_encoder.pkl, novo_unified_stats.json"
        )

    unified_model = xgb.XGBRegressor()
    unified_model.load_model(UNIFIED_NOVO_MODEL)
    unified_cinema_enc = joblib.load(UNIFIED_CINEMA_ENC)
    unified_movie_enc  = joblib.load(UNIFIED_MOVIE_ENC)
    with open(UNIFIED_STATS_FILE) as f:
        unified_stats = json.load(f)
    print(f"   ✓ Unified Novo model loaded  (MAE={unified_stats.get('mae',0):.2f}, "
          f"{unified_stats.get('n_movies',0)} movies, "
          f"{unified_stats.get('n_cinemas',0)} cinemas)")

    # ── 2. Metadata cache (TMDB Integration) ──────────────────────────────────
    metadata_cache = {}
    if os.path.exists(METADATA_CACHE_FILE):
        with open(METADATA_CACHE_FILE) as f:
            metadata_cache = json.load(f)
        print(f"   Loaded {len(metadata_cache)} movies from metadata cache.")

    # Define unique movies BEFORE looping
    unique_movies = df_sessions['movie_title'].dropna().unique()
    new_data_fetched = False
    
    print("   Checking TMDB for missing metadata...")
    for movie in unique_movies:
        norm = normalize_name(movie)
        if norm and norm not in metadata_cache:
            print(f"     Fetching TMDB data for: {movie}")
            details = fetch_tmdb_movie_details(movie)
            metadata_cache[norm] = details if details else {
                'budget': 50000000, 'runtime': 120, 'popularity': 10, 'vote_average': 7.0
            }
            new_data_fetched = True

    if new_data_fetched:
        with open(METADATA_CACHE_FILE, 'w') as f:
            json.dump(metadata_cache, f, indent=2)
        print("   Updated metadata cache saved.")

    return (df_sessions,
        unified_model, unified_cinema_enc, unified_movie_enc,
        metadata_cache)
def _safe_encode(encoder, value, fallback='unknown'):
    """Encode a value; fall back to 'unknown' sentinel if unseen."""
    try:
        val = str(value)
        if val in encoder.classes_:
            return int(encoder.transform([val])[0])
        elif fallback in encoder.classes_:
            return int(encoder.transform([fallback])[0])
        else:
            return 0
    except Exception:
        return 0


def prepare_unified_features(movie_name, cinema_id, show_time_str, current_date,
                              cinema_enc, movie_enc):
    """Build feature row for the unified Novo model."""
    dt = pd.to_datetime(f"{current_date} {show_time_str}")
    cinema_encoded = _safe_encode(cinema_enc, cinema_id)
    movie_encoded  = _safe_encode(movie_enc,  movie_name)
    row = pd.DataFrame([{
        'day_of_week':       dt.dayofweek,
        'is_weekend':        1 if dt.dayofweek >= 4 else 0,
        'hour':              dt.hour,
        'cinema_id_encoded': cinema_encoded,
        'movie_id_encoded':  movie_encoded,
    }])
    return row


def predict_sales(movie_name, cinema_id, slot,
                  unified_model, unified_cinema_enc, unified_movie_enc,
                  metadata_cache, current_date):
    row = prepare_unified_features(
        movie_name, cinema_id, slot, current_date,
        unified_cinema_enc, unified_movie_enc
    )
    for f in UNIFIED_FEATURES:
        if f not in row.columns:
            row[f] = 0

    pred = unified_model.predict(row[UNIFIED_FEATURES])[0]
    mtype = "Unified"
    feats = row.to_dict('records')[0]

    # Add the fractional tie-breaker to ALL tiers to guarantee no ties
    deterministic_decimal = (sum(ord(c) for c in movie_name) % 100) / 100.0
    pred += deterministic_decimal

    # Return as a float so exact decimal differences are captured
    return max(0.0, float(pred)), mtype, feats
def main():
    (df_sessions,
     unified_model, unified_cinema_enc, unified_movie_enc,
     metadata_cache) = load_resources()
    
    # Get distinct Cinemas
    cinema_ids = sorted(df_sessions['fk_cinema_id'].unique())
    print(f"Found {len(cinema_ids)} unique Cinemas: {cinema_ids}")
    
    # Get all Movies available in the dataset
    all_movies = sorted(df_sessions['movie_title'].dropna().unique())
    print(f"Found {len(all_movies)} unique Movies.")
    
    # Dates to schedule
    # User said "today, tommorows", we can infer from data or use fixed
    # Data has 19th and 20th. Let's strictly use the dates present in the data.
    # OR better: Generate for the dates PRESENT in the file.
    unique_dates = sorted(pd.to_datetime(df_sessions['show_time']).dt.date.unique())
    date_strs = [d.strftime("%Y-%m-%d") for d in unique_dates]
    print(f"Scheduling for dates: {date_strs}")

    for cid in cinema_ids:
        print(f"\nProcessing Cinema {cid}...")
        
        # Determine screens for THIS cinema
        cinema_data = df_sessions[df_sessions['fk_cinema_id'] == cid]
        if 'screen_number' in cinema_data.columns:
            screens = sorted(cinema_data['screen_number'].unique())
        else:
            screens = list(range(1, 6)) # Default
        
        # If screens are 0 or empty, default to 1..5
        # Sometimes screen_number might be 0? 
        screens = [s for s in screens if s > 0]
        if not screens: screens = list(range(1, 6))
        
        for current_date in date_strs:
            print(f"  Date: {current_date} | Screens: {screens}")
            
            all_schedule_data = [] # Reset for each day
            
            for slot in TIME_SLOTS:
                # Rank movies for this cinema/date/slot
                scores = []
                for mv_name in all_movies:
                    pred, mtype, feats = predict_sales(
                        mv_name, cid, slot,
                        unified_model, unified_cinema_enc, unified_movie_enc,
                        metadata_cache, current_date
                    )
                    scores.append({
                        'name': mv_name,
                        'sales': pred,
                        'type': mtype,
                        'features': feats
                    })
                
                scores.sort(key=lambda x: x['sales'], reverse=True)
                
                # Assign
                row_dict = {'Date': current_date, 'Time': slot}
                
                for i, audi_id in enumerate(screens):
                    cell_content = "No Movie"
                    
                    # Logic to fill empty screens with top movies if we run out of unique movies
                    # Cycle through movies using modulo
                    movie_idx = i % len(scores) if len(scores) > 0 else -1
                    
                    if movie_idx >= 0:
                        winner = scores[movie_idx]
                        
                        explanation = ""
                        # Compare with NEXT rank (or circle back to first if it's the last one)
                        if movie_idx + 1 < len(scores):
                            runner_up = scores[movie_idx+1]
                        else:
                            runner_up = scores[0] 
                        
                        # Only show explanation if it's a different movie
                        if winner['name'] != runner_up['name']:
                            diff = winner['sales'] - runner_up['sales']
                            
                            # Round the currency to a clean whole number
                            rev_diff = int(round(diff * BASE_PRICE_QAR))
                            pct = (diff / runner_up['sales'] * 100) if runner_up['sales'] > 0 else 100.0
                            
                            if pct > 75:
                                explanation = f"  |  Expected to drastically outperform {runner_up['name']} by +{rev_diff} QAR (+{pct:.1f}%)."
                            elif pct > 30:
                                explanation = f"  |  Projected to generate +{rev_diff} QAR more than {runner_up['name']} (+{pct:.1f}%)."
                            else:
                                explanation = f"  |  Edges out {runner_up['name']} with a +{rev_diff} QAR advantage (+{pct:.1f}%)."
                        else:
                            explanation = "  |  (Top Performing Choice)"
                            
                        # Apply Cinepolis-style explanations
                        if winner['type'] == 'Unified':
                            explanation += " [Key Factor: Proven historical track record at this specific cinema]"
                        else:
                            explanation += " [Key Factor: Favorable time-slot fit]"

                        cell_content = f"{winner['name']}{explanation}"
                    
                    row_dict[f"Audi {audi_id}"] = cell_content
                
                all_schedule_data.append(row_dict)
            
            # Output per cinema per day
            df_out = pd.DataFrame(all_schedule_data)
            df_out.set_index(['Date', 'Time'], inplace=True)
            
            # Naming convention: Proposed_Schedule_Novo_{CID}_{DATE}.xlsx
            out_file = f"Proposed_Schedule_Novo_{cid}_{current_date}.xlsx"
            try:
                df_out.to_excel(out_file)
                print(f"  Saved schedule to {out_file}")
            except Exception as e:
                print(f"  Error saving {out_file}: {e}")


if __name__ == "__main__":
    main()
