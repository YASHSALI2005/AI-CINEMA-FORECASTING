import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import re
import random
from datetime import datetime, timedelta

# --- CONFIG ---
CINEMA_ID = 890
AUDITORIUMS = list(range(1, 12)) # Screens 1 to 11
TIME_SLOTS = ["09:00", "12:00", "15:00", "18:00", "21:00"]

MODELS_DIR = "models"
UNIFIED_MODEL_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_model.json")
UNIFIED_CINEMA_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_cinema_encoder.pkl")
UNIFIED_MOVIE_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_encoder.pkl")
UNIFIED_MOVIE_MAP_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_map.json")
UNIFIED_STATS_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_stats.json")
BMS_FILE = "bms_currently_playing.json"
GENERIC_MODEL_FILE = "xgb_cinema_model_v5.json"
GENERIC_ENCODER_FILE = "cinema_encoder_v5.pkl"
TRAINING_DATA_FILE = "final_training_data_feb18.csv"

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip().lower()

def load_resources():
    print(f"Loading resources for Cinema {CINEMA_ID}...")

    unified_model = xgb.XGBRegressor()
    unified_model.load_model(UNIFIED_MODEL_FILE)
    unified_cinema_encoder = joblib.load(UNIFIED_CINEMA_ENCODER_FILE)
    unified_movie_encoder = joblib.load(UNIFIED_MOVIE_ENCODER_FILE)
    with open(UNIFIED_MOVIE_MAP_FILE) as f:
        unified_movie_map = json.load(f)

    unified_features = [
        'day_of_week', 'is_weekend', 'hour',
        'log_days_since_release',
        'cinema_id_encoded', 'movie_id_encoded',
        'competitors_on_screen',
        'cinema_avg_daily', 'cinema_hour_avg',
        'budget', 'runtime', 'popularity', 'vote_average',
        'movie_trend_7d', 'cinema_trend_7d'
    ]
    if os.path.exists(UNIFIED_STATS_FILE):
        try:
            with open(UNIFIED_STATS_FILE) as f:
                stats = json.load(f)
            unified_features = stats.get('features', unified_features)
        except Exception:
            pass
    
    with open(BMS_FILE) as f:
        bms_movies = json.load(f)
    
    generic_model = xgb.XGBRegressor()
    generic_model.load_model(GENERIC_MODEL_FILE)
    generic_encoder = joblib.load(GENERIC_ENCODER_FILE)
    
    print("   Loading metadata from training data...")
    metadata_cache = {}
    
    try:
        cols = ['movie_name', 'budget', 'runtime', 'popularity', 'vote_average', 'show_time']
        for chunk in pd.read_csv(TRAINING_DATA_FILE, usecols=cols, chunksize=100000):
            chunk['show_time'] = pd.to_datetime(chunk['show_time'], errors='coerce')
            chunk.sort_values('show_time', ascending=True, inplace=True)
            for _, row in chunk.iterrows():
                norm = normalize_name(row['movie_name'])
                if norm:
                    metadata_cache[norm] = {
                        'budget': row.get('budget', 0),
                        'runtime': row.get('runtime', 0),
                        'popularity': row.get('popularity', 0),
                        'vote_average': row.get('vote_average', 0)
                    }
    except Exception as e:
        print(f"   Warning: Could not load metadata: {e}")

    print("   Loaded unified model, generic model, encoders, and metadata.")
    return (
        unified_model,
        unified_cinema_encoder,
        unified_movie_encoder,
        unified_movie_map,
        unified_features,
        bms_movies,
        generic_model,
        generic_encoder,
        metadata_cache,
    )

def get_movie_metadata(movie_name, cache):
    norm = normalize_name(movie_name)
    if norm in cache:
        return cache[norm]
    
    # Defaults
    return {
        'budget': 50000000,
        'runtime': 150,
        'popularity': 10,
        'vote_average': 7.0
    }

def prepare_features(movie_name, cinema_id, show_time_str, metadata, current_date, encoder=None):
    # Now uses the current_date from the loop so day_of_week and is_weekend update correctly
    dt = pd.to_datetime(f"{current_date} {show_time_str}")
    
    features = {
        'day_of_week': dt.dayofweek,
        'is_weekend': 1 if dt.dayofweek >= 4 else 0,
        'hour': dt.hour,
        'log_days_since_release': 2.0, 
        'cinema_id': cinema_id,
        'budget': float(metadata.get('budget', 0)),
        'runtime': float(metadata.get('runtime', 0)),
        'popularity': float(metadata.get('popularity', 0)),
        'vote_average': float(metadata.get('vote_average', 0)),
        'competitors_on_screen': 5,
        'movie_trend_7d': 0,
        'cinema_trend_7d': 0,
        'cinema_avg_daily': 500,
        'cinema_hour_avg': 50,
        'holiday_weight': 0,
        'bh_opening_day': 0,
        'bh_verdict_score': 0
    }
    
    df = pd.DataFrame([features])
    
    if encoder:
        c_val = str(cinema_id)
        if c_val in encoder.classes_:
            df['cinema_id_encoded'] = encoder.transform([c_val])[0]
        else:
            df['cinema_id_encoded'] = 0
            
    return df


def resolve_primary_movie_name(movie_name, unified_movie_map):
    movie_name = str(movie_name)
    if movie_name in unified_movie_map:
        return movie_name

    n_input = normalize_name(movie_name)
    for primary, variants in unified_movie_map.items():
        p = normalize_name(primary)
        if n_input == p or n_input in p or p in n_input:
            return primary
        for v in variants if isinstance(variants, list) else []:
            nv = normalize_name(v)
            if n_input == nv or n_input in nv or nv in n_input:
                return primary
    return None

def predict_sales(movie_name, cinema_id, slot,
                  unified_model, unified_cinema_encoder, unified_movie_encoder,
                  unified_movie_map, unified_features,
                  generic_model, generic_encoder,
                  metadata_cache, current_date):
    metadata = get_movie_metadata(movie_name, metadata_cache)

    primary_name = resolve_primary_movie_name(movie_name, unified_movie_map)

    # Pass the current_date to prepare_features
    row = prepare_features(movie_name, cinema_id, slot, metadata, current_date, encoder=generic_encoder)

    if primary_name and str(primary_name) in set(unified_movie_encoder.classes_):
        c_val = str(cinema_id)
        row['cinema_id_encoded'] = int(unified_cinema_encoder.transform([c_val])[0]) if c_val in set(unified_cinema_encoder.classes_) else 0
        row['movie_id_encoded'] = int(unified_movie_encoder.transform([str(primary_name)])[0])
        row['cinema_avg_daily'] = row.get('cinema_avg_daily', 500)
        row['cinema_hour_avg'] = row.get('cinema_hour_avg', 50)

        for f in unified_features:
            if f not in row.columns: row[f] = 0
        X = row[unified_features]
        pred = unified_model.predict(X)[0]
        model_type = "Unified"
        used_features = row.to_dict('records')[0]
    else:
        gen_features = [
            'budget', 'runtime', 'popularity', 'vote_average',
            'day_of_week', 'is_weekend', 'hour', 'holiday_weight',
            'competitors_on_screen', 'log_days_since_release',
            'cinema_id_encoded', 'movie_trend_7d', 'cinema_trend_7d',
            'bh_opening_day', 'bh_verdict_score'
        ]
        for f in gen_features:
            if f not in row.columns: row[f] = 0
        X = row[gen_features]
        pred = generic_model.predict(X)[0]
        model_type = "Generic"
        used_features = row.to_dict('records')[0]

    return max(0, int(pred)), model_type, used_features

def main():
    (unified_model, unified_cinema_encoder, unified_movie_encoder,
     unified_movie_map, unified_features,
     bms_movies, gen_model, gen_encoder, metadata_cache) = load_resources()
    
    # Generate a list of dates: Past 10 days + Today + Next 10 days
    start_date = datetime.now() - timedelta(days=10)
    date_list = [(start_date + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(21)]
    
    all_schedule_data = [] # Master list for all days
    
    print(f"\nOptimizing Schedule for {len(date_list)} dates (Past 10 days to Next 10 days)...")
    
    import random # Ensures varied text generation
    
    for current_date in date_list:
        print(f"  Generating schedule for {current_date}...")
        for slot in TIME_SLOTS:
            # Score all movies
            scores = []
            for mv in bms_movies:
                name = mv['name']
                # Pass current_date into the prediction
                pred, mtype, feats = predict_sales(
                    name,
                    CINEMA_ID,
                    slot,
                    unified_model,
                    unified_cinema_encoder,
                    unified_movie_encoder,
                    unified_movie_map,
                    unified_features,
                    gen_model,
                    gen_encoder,
                    metadata_cache,
                    current_date,
                )
                scores.append({
                    'name': name,
                    'sales': pred,
                    'type': mtype,
                    'features': feats
                })
            
            scores.sort(key=lambda x: x['sales'], reverse=True)
            
            # Assign to Audis
            row_dict = {'Date': current_date, 'Time': slot}
            
            for i, audi_id in enumerate(AUDITORIUMS):
                cell_content = "No Movie"
                
                if i < len(scores):
                    winner = scores[i]
                    
                    explanation = ""
                    if i + 1 < len(scores):
                        runner_up = scores[i+1]
                        diff = winner['sales'] - runner_up['sales']
                        rev_diff = diff * 100
                        pct = (diff / runner_up['sales'] * 100) if runner_up['sales'] > 0 else 100.0
                        
                        # Added explicit spacing "  |  " to prevent squishing
                        if pct > 75:
                            explanation = f"  |  Expected to drastically outperform {runner_up['name']} by +{rev_diff} Rs (+{pct:.1f}%)."
                        elif pct > 30:
                            explanation = f"  |  Projected to generate +{rev_diff} Rs more than {runner_up['name']} (+{pct:.1f}%)."
                        else:
                            explanation = f"  |  Edges out {runner_up['name']} with a +{rev_diff} Rs advantage (+{pct:.1f}%)."
                        
                        extras = []
                        w_feat = winner['features']
                        r_feat = runner_up['features']
                        
                        if w_feat.get('popularity', 0) > r_feat.get('popularity', 0) * 1.15:
                            extras.append("stronger widespread popularity")
                        if w_feat.get('vote_average', 0) > r_feat.get('vote_average', 0) + 0.3:
                            extras.append("superior audience ratings")
                        if w_feat.get('budget', 0) > r_feat.get('budget', 0) * 1.5:
                            extras.append("massive production scale (mega-budget pull)")
                        if w_feat.get('log_days_since_release', 10) < r_feat.get('log_days_since_release', 10) - 0.2:
                            extras.append("fresher release momentum")
                        if winner.get('type') == 'Specific' and runner_up.get('type') != 'Specific':
                            extras.append("proven historical track record at this specific cinema")
                        if w_feat.get('movie_trend_7d', 0) > r_feat.get('movie_trend_7d', 0) + 0.1:
                            extras.append("better 7-day booking trends")
                        
                        if extras:
                            random.shuffle(extras)
                            selected_extras = extras[:2]
                            reason_text = " and ".join(selected_extras).capitalize()
                            explanation += f" [Key Factor: {reason_text}]"
                        else:
                            explanation += " [Key Factor: Favorable time-slot fit]"
                    else:
                        explanation = "  |  (Best remaining option for this slot)"
                    
                    cell_content = f"{winner['name']}{explanation}"
                
                row_dict[f"Audi {audi_id}"] = cell_content
                
            all_schedule_data.append(row_dict)
            
        # ---> NEW: Visual Separator Row <---
        # This adds an empty line after the 21:00 slot before the next day starts
        separator_row = {'Date': current_date, 'Time': '----------'}
        for audi_id in AUDITORIUMS:
            separator_row[f"Audi {audi_id}"] = "" # Keeps the audi cells completely blank
        all_schedule_data.append(separator_row)

    # Output
    df = pd.DataFrame(all_schedule_data)
    # Group by Date and Time for a clean Excel layout
    df.set_index(['Date', 'Time'], inplace=True)
    
    print("\n" + "="*80)
    print(f"OPTIMAL SCHEDULE FOR CINEMA {CINEMA_ID} (21-DAY SPREAD)")
    print("="*80)
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"Proposed_Schedule_{CINEMA_ID}_MultiDate_{timestamp}.xlsx"
    
    try:
        df.to_excel(out_file)
        print(f"\nMulti-date schedule successfully saved to {out_file}")
    except PermissionError:
        out_file_fallback = f"Proposed_Schedule_{CINEMA_ID}_MultiDate_{timestamp}_fallback.xlsx"
        df.to_excel(out_file_fallback)
        print(f"\nWarning: Could not open primary file. Saved to {out_file_fallback} instead.")
if __name__ == "__main__":
    main()