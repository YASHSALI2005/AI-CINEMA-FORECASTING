"""
train_novo_unified.py
---------------------
Trains a SINGLE unified XGBoost model for Novo cinemas.
Features: cinema_id_encoded + movie_id_encoded + day_of_week + is_weekend + hour
This replaces the per-movie model approach (30+ JSON files → 1 model).
New/unseen movies use the 'unknown' fallback encoding.
"""

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
SESSIONS_FILE   = r"d:\Forecasting\sessions.csv.xlsx"
MODELS_DIR      = r"d:\Forecasting\models\novo"
UNIFIED_MODEL   = os.path.join(MODELS_DIR, "novo_unified_model.json")
CINEMA_ENC_FILE = os.path.join(MODELS_DIR, "novo_unified_cinema_encoder.pkl")
MOVIE_ENC_FILE  = os.path.join(MODELS_DIR, "novo_unified_movie_encoder.pkl")
STATS_FILE      = os.path.join(MODELS_DIR, "novo_unified_stats.json")
MIN_ROWS        = 3   # minimum rows overall to proceed

FEATURES = ['day_of_week', 'is_weekend', 'hour', 'cinema_id_encoded', 'movie_id_encoded']


def normalize_name(name):
    if not isinstance(name, str): return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip().lower()


def train_model():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading sessions data...")
    df = pd.read_excel(SESSIONS_FILE)
    print(f"  Loaded {len(df)} rows.")

    # --- Infer capacity & sold tickets ---
    df['screen_id'] = df['fk_cinema_id'].astype(str) + "_" + df['screen_number'].astype(str)
    screen_caps = df.groupby('screen_id')['seats_available'].max().to_dict()
    df['capacity'] = df['screen_id'].map(screen_caps)
    df['sold_tickets'] = (df['capacity'] - df['seats_available']).clip(lower=0)

    # --- Feature Engineering ---
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend']  = df['day_of_week'].isin([4, 5, 6]).astype(int)
    df['hour']        = df['show_time'].dt.hour

    # Drop rows with no movie title
    df = df.dropna(subset=['movie_title'])

    # --- Encode cinema_id ---
    le_cinema = LabelEncoder()
    # Add a sentinel 'unknown' so unseen cinemas don't crash
    all_cinema_ids = list(df['fk_cinema_id'].astype(str).unique()) + ['unknown']
    le_cinema.fit(all_cinema_ids)
    df['cinema_id_encoded'] = le_cinema.transform(df['fk_cinema_id'].astype(str))

    # --- Encode movie_title ---
    le_movie = LabelEncoder()
    # Normalize movie names for matching, keep the raw list for encoder
    all_movies = list(df['movie_title'].unique()) + ['unknown']
    le_movie.fit(all_movies)
    df['movie_id_encoded'] = le_movie.transform(df['movie_title'])

    print(f"  {df['fk_cinema_id'].nunique()} unique cinemas, {df['movie_title'].nunique()} unique movies.")

    if len(df) < MIN_ROWS:
        print("Not enough data to train. Exiting.")
        return

    # --- Train / Test split ---
    X = df[FEATURES]
    y = df['sold_tickets']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"  Training: {len(X_train)} rows | Test: {len(X_test)} rows")

    # --- Train ---
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=50)

    preds = model.predict(X_test)
    mae   = mean_absolute_error(y_test, preds)
    r2    = r2_score(y_test, preds)
    print(f"\n  MAE: {mae:.2f}  |  R²: {r2:.3f}")

    # --- Save ---
    model.save_model(UNIFIED_MODEL)
    joblib.dump(le_cinema, CINEMA_ENC_FILE)
    joblib.dump(le_movie,  MOVIE_ENC_FILE)

    # Save stats (movie list, cinema list, metrics) for the scheduler to use
    stats = {
        "mae": mae,
        "r2":  r2,
        "features": FEATURES,
        "n_movies":  int(df['movie_title'].nunique()),
        "n_cinemas": int(df['fk_cinema_id'].nunique()),
        "known_movies":  list(df['movie_title'].unique()),
        "known_cinemas": [str(c) for c in df['fk_cinema_id'].unique()],
        "trained_at": datetime.now().isoformat()
    }
    with open(STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nUnified model saved → {UNIFIED_MODEL}")
    print(f"Cinema encoder  → {CINEMA_ENC_FILE}")
    print(f"Movie encoder   → {MOVIE_ENC_FILE}")
    print(f"Stats           → {STATS_FILE}")
    print("\nDone! You now have ONE model for all movies + cinemas.")


if __name__ == "__main__":
    train_model()
