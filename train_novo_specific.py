import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

# --- CONFIG ---
SESSIONS_FILE = r"d:\Forecasting\sessions.csv.xlsx"
MODELS_DIR = r"d:\Forecasting\models\novo"
MODEL_MAP_FILE = os.path.join(MODELS_DIR, "model_map_novo.json")
MIN_ROWS = 5 # Lower threshold since data might be sparse

def normalize_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip().lower()

def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading sessions data...")
    df = pd.read_excel(SESSIONS_FILE)
    
    # 1. Infer Capacity & Calculate Sold Tickets
    # Assumption: For a given Cinema + Screen Name/Number, the MAX 'seats_available' seen is the Capacity?
    # OR: Maybe 'seats_available' starts at capacity? 
    # Let's assume Capacity = Max(seats_available) for that screen group.
    
    # Create unique screen identifier
    df['screen_id'] = df['fk_cinema_id'].astype(str) + "_" + df['screen_number'].astype(str)
    
    # Infer capacity
    screen_caps = df.groupby('screen_id')['seats_available'].max().to_dict()
    df['capacity'] = df['screen_id'].map(screen_caps)
    
    # Calculate sold
    # sold = capacity - available
    df['sold_tickets'] = df['capacity'] - df['seats_available']
    df['sold_tickets'] = df['sold_tickets'].clip(lower=0) # Sanity check
    
    # 2. Feature Engineering
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int) # Fri-Sat-Sun logic?
    df['hour'] = df['show_time'].dt.hour
    
    # Encoder for Cinema ID
    le_cinema = LabelEncoder()
    df['cinema_id_encoded'] = le_cinema.fit_transform(df['fk_cinema_id'])
    
    # Save the encoder for general use
    joblib.dump(le_cinema, os.path.join(MODELS_DIR, "cinema_encoder_novo.pkl"))
    
    # 3. Train Per Movie
    unique_movies = df['movie_title'].dropna().unique()
    model_map = {}
    
    print(f"Found {len(unique_movies)} movies. Training models...")
    
    for movie in unique_movies:
        movie_df = df[df['movie_title'] == movie].copy()
        
        if len(movie_df) < MIN_ROWS:
            print(f"  Skipping {movie}: {len(movie_df)} rows (threshold {MIN_ROWS})")
            continue
            
        print(f"  Training {movie} ({len(movie_df)} rows)...")
        
        features = ['day_of_week', 'is_weekend', 'hour', 'cinema_id_encoded']
        X = movie_df[features]
        y = movie_df['sold_tickets']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6)
        model.fit(X_train, y_train)
        
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # Save
        clean_name = normalize_name(movie)[:50]
        model_path = os.path.join(MODELS_DIR, f"{clean_name}.json")
        model.save_model(model_path)
        
        model_map[movie] = {
            'model_path': model_path,
            'features': features,
            'mae': mae
        }
        print(f"    Saved model. MAE: {mae:.2f}")

    # Save Map
    with open(MODEL_MAP_FILE, 'w') as f:
        json.dump(model_map, f, indent=2)
        
    print(f"\nTraining complete. Map saved to {MODEL_MAP_FILE}")

if __name__ == "__main__":
    train_model()
