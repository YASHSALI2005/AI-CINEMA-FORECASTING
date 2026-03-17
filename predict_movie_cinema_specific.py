"""
Generate Predictions with Unified Movie+Cinema Model
Output format matches the original model (new.py):
  Date | Time | Cinema Name | Pred % (range) | Actual %

Past 7 days (Feb 11-18): Shows actuals and predictions
Coming 10 days (Feb 18-28): Shows predictions only
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import re
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# --- CONFIG ---
CSV_FILE = "final_training_data_feb18.csv"
CINEMA_NAMES_FILE = "cinema_names.csv"
MODELS_DIR = "models"
UNIFIED_MODEL_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_model.json")
UNIFIED_CINEMA_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_cinema_encoder.pkl")
UNIFIED_MOVIE_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_encoder.pkl")
UNIFIED_STATS_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_stats.json")
UNIFIED_MOVIE_MAP_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_map.json")

TODAY = pd.to_datetime("2026-02-18")
PAST_START = TODAY - timedelta(days=7)   # Feb 11
FUTURE_END = TODAY + timedelta(days=10)  # Feb 28


def round_time_to_nearest_30(dt):
    """Round a datetime to the nearest 30 min slot."""
    minute = dt.minute
    if minute < 15:
        return dt.replace(minute=0, second=0, microsecond=0)
    elif minute < 45:
        return dt.replace(minute=30, second=0, microsecond=0)
    else:
        return (dt + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)


def load_cinema_map():
    try:
        cn = pd.read_csv(CINEMA_NAMES_FILE)
        cn['cinema_id'] = cn['cinema_id'].apply(lambda x: str(int(float(x))) if pd.notna(x) else str(x))
        cn['cinema_code'] = cn['cinema_code'].apply(
            lambda x: str(int(float(x))) if pd.notna(x) and str(x) != 'NULL' else str(x))
        m1 = dict(zip(cn['cinema_id'], cn['cinema_name']))
        m2 = dict(zip(cn['cinema_code'], cn['cinema_name']))
        return {**m2, **m1}
    except:
        return {}


def load_data():
    print("Loading training data...")
    df = pd.concat(list(pd.read_csv(CSV_FILE, chunksize=200000)), ignore_index=True)
    df['show_time'] = pd.to_datetime(df['show_time'], errors='coerce')
    return df


def prepare_features(df, cinema_encoder, movie_encoder):
    """Prepare features for unified movie+cinema model prediction."""
    df = df.copy()
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)
    df['hour'] = df['show_time'].dt.hour
    df['date_str'] = df['show_time'].dt.date.astype(str)

    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['days_since_release'] = (df['show_time'] - df['release_date']).dt.days
        df['log_days_since_release'] = np.log1p(df['days_since_release'].fillna(0).clip(lower=0))
    else:
        df['log_days_since_release'] = 0

    cinema_known = set(cinema_encoder.classes_)
    movie_known = set(movie_encoder.classes_)

    df['cinema_id_encoded'] = df['cinema_id'].astype(str).apply(
        lambda x: int(cinema_encoder.transform([x])[0]) if x in cinema_known else 0
    )

    movie_source = 'movie_primary' if 'movie_primary' in df.columns else 'movie_name'
    df['movie_id_encoded'] = df[movie_source].astype(str).apply(
        lambda x: int(movie_encoder.transform([x])[0]) if x in movie_known else 0
    )

    for c in ['budget', 'runtime', 'popularity', 'vote_average',
              'competitors_on_screen', 'movie_trend_7d', 'cinema_trend_7d']:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Cinema-specific aggregates
    cinema_daily = df.groupby(['cinema_id', 'date_str'])['sold_tickets'].sum().reset_index()
    cinema_avg = cinema_daily.groupby('cinema_id')['sold_tickets'].mean().to_dict()
    df['cinema_avg_daily'] = df['cinema_id'].map(cinema_avg).fillna(0)

    cinema_hour_avg = df.groupby(['cinema_id', 'hour'])['sold_tickets'].mean().to_dict()
    df['cinema_hour_avg'] = df.apply(
        lambda x: cinema_hour_avg.get((x['cinema_id'], x['hour']), 0), axis=1
    )
    return df


def generate_grid_schedule(movie_data, start_dt, end_dt):
    """Generate a grid of future shows based on last week's schedule pattern."""
    latest = movie_data['show_time'].max()
    template_start = latest - timedelta(days=7)
    template = movie_data[movie_data['show_time'] >= template_start].copy()
    if template.empty:
        template = movie_data.tail(500).copy()

    rows = []
    current = start_dt
    while current <= end_dt:
        weekday = current.weekday()
        match = template[template['show_time'].dt.weekday == weekday]
        if not match.empty:
            projected = match.copy()
            sample_date = projected['show_time'].iloc[0].date()
            delta = (current.date() - sample_date).days
            projected['show_time'] = projected['show_time'] + pd.Timedelta(days=delta)
            projected['sold_tickets'] = 0
            rows.append(projected)
        current += timedelta(days=1)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def process_movie(movie_name, all_variants, model, cinema_encoder, movie_encoder, features, df, c_map):
    """Process one movie: past actuals + future predictions, in range format."""
    print(f"\n{'='*60}")
    print(f"Processing: {movie_name}")
    print(f"{'='*60}")

    # Get all data for this movie (all variants)
    movie_slice = df[df['movie_name'].isin(all_variants)].copy()
    if movie_slice.empty:
        print(f"   No data for {movie_name}. Skipping.")
        return
    movie_slice['movie_primary'] = movie_name

    # Prepare features on full movie data
    movie_prepped = prepare_features(movie_slice, cinema_encoder, movie_encoder)

    # Capacity map for this movie's cinemas
    cap_map = movie_prepped.groupby('cinema_id')['capacity'].median().to_dict()

    # === GRID: PAST 7 DAYS + FUTURE 10 DAYS ===
    # Past data already exists in movie_prepped, use it directly
    past_mask = (movie_prepped['show_time'] >= PAST_START) & (movie_prepped['show_time'] < TODAY)
    past_data = movie_prepped[past_mask].copy()
    print(f"   Past 7 days: {len(past_data)} shows")

    # Future schedule from template
    future_data = generate_grid_schedule(movie_prepped, TODAY, FUTURE_END)
    print(f"   Future 10 days: {len(future_data)} projected shows")

    if future_data.empty and past_data.empty:
        print(f"   No data to process. Skipping.")
        return

    # Prepare future features
    if not future_data.empty:
        future_data['movie_primary'] = movie_name
        future_data = prepare_features(future_data, cinema_encoder, movie_encoder)

    # === PREDICTIONS ===
    all_results = []

    # PAST: actuals + predictions
    if not past_data.empty:
        for c in features:
            if c not in past_data.columns:
                past_data[c] = 0
            past_data[c] = pd.to_numeric(past_data[c], errors='coerce').fillna(0)

        raw_pred = np.maximum(model.predict(past_data[features]), 0)
        past_data['raw_prediction'] = raw_pred
        past_data['est_capacity'] = past_data['cinema_id'].map(cap_map).fillna(250)
        past_data['capacity'] = pd.to_numeric(past_data['capacity'], errors='coerce').fillna(300)
        past_data['base_pct'] = (past_data['raw_prediction'] / past_data['est_capacity']) * 100
        past_data['base_pct'] = past_data['base_pct'].clip(0, 100)

        low = past_data['base_pct'].round().astype(int)
        high = (past_data['base_pct'] + 20).clip(upper=100).round().astype(int)
        past_data['Pred %'] = low.astype(str) + "-" + high.astype(str) + "%"

        # Actual %
        past_data['sold_tickets'] = pd.to_numeric(past_data['sold_tickets'], errors='coerce').fillna(0)
        past_data['actual_pct'] = (past_data['sold_tickets'] / past_data['capacity']) * 100
        past_data['actual_pct'] = past_data['actual_pct'].clip(0, 100)
        past_data['Actual %'] = past_data['actual_pct'].apply(
            lambda x: f"{x:.1f}%" if x > 0 else ""
        )

        past_data['Date'] = past_data['show_time'].dt.date
        past_data['Time'] = past_data['show_time'].dt.strftime('%H:%M')
        past_data['Cinema Name'] = past_data['cinema_id'].astype(str).map(c_map).fillna(
            past_data['cinema_id'].astype(str)
        )

        all_results.append(past_data[['Date', 'Time', 'Cinema Name', 'Pred %', 'Actual %']])

    # FUTURE: predictions only
    if not future_data.empty:
        for c in features:
            if c not in future_data.columns:
                future_data[c] = 0
            future_data[c] = pd.to_numeric(future_data[c], errors='coerce').fillna(0)

        raw_pred = np.maximum(model.predict(future_data[features]), 0)
        future_data['raw_prediction'] = raw_pred
        future_data['est_capacity'] = future_data['cinema_id'].map(cap_map).fillna(250)
        future_data['base_pct'] = (future_data['raw_prediction'] / future_data['est_capacity']) * 100
        future_data['base_pct'] = future_data['base_pct'].clip(0, 100)

        low = future_data['base_pct'].round().astype(int)
        high = (future_data['base_pct'] + 20).clip(upper=100).round().astype(int)
        future_data['Pred %'] = low.astype(str) + "-" + high.astype(str) + "%"
        future_data['Actual %'] = ""  # No actuals for future

        future_data['Date'] = future_data['show_time'].dt.date
        future_data['Time'] = future_data['show_time'].dt.strftime('%H:%M')
        future_data['Cinema Name'] = future_data['cinema_id'].astype(str).map(c_map).fillna(
            future_data['cinema_id'].astype(str)
        )

        all_results.append(future_data[['Date', 'Time', 'Cinema Name', 'Pred %', 'Actual %']])

    if not all_results:
        return

    output_df = pd.concat(all_results, ignore_index=True)
    output_df = output_df.sort_values(by=['Date', 'Cinema Name', 'Time'])

    # Save Excel
    clean_title = re.sub(r'[\\/*?:\"<>|]', "", movie_name).replace(" ", "_")[:30]
    fname = f"Pred_Specific_{clean_title}.xlsx"

    if os.path.exists(fname):
        try:
            os.remove(fname)
        except:
            fname = f"Pred_Specific_{clean_title}_new.xlsx"

    output_df.to_excel(fname, index=False)
    print(f"   Saved: {fname}")
    print(f"   Total rows: {len(output_df)} (past: {len(past_data) if not past_data.empty else 0}, future: {len(future_data) if not future_data.empty else 0})")


def main():
    c_map = load_cinema_map()
    df = load_data()

    print("Loading unified model artifacts...")
    model = xgb.XGBRegressor()
    model.load_model(UNIFIED_MODEL_FILE)
    cinema_encoder = joblib.load(UNIFIED_CINEMA_ENCODER_FILE)
    movie_encoder = joblib.load(UNIFIED_MOVIE_ENCODER_FILE)

    with open(UNIFIED_MOVIE_MAP_FILE) as f:
        movie_map = json.load(f)

    if os.path.exists(UNIFIED_STATS_FILE):
        with open(UNIFIED_STATS_FILE) as f:
            stats = json.load(f)
        features = stats.get('features', [])
    else:
        features = [
            'day_of_week', 'is_weekend', 'hour',
            'log_days_since_release',
            'cinema_id_encoded', 'movie_id_encoded',
            'competitors_on_screen',
            'cinema_avg_daily', 'cinema_hour_avg',
            'budget', 'runtime', 'popularity', 'vote_average',
            'movie_trend_7d', 'cinema_trend_7d'
        ]

    print(f"Found {len(movie_map)} movies in unified map")

    for movie_name, all_variants in movie_map.items():
        process_movie(
            movie_name,
            all_variants if isinstance(all_variants, list) else [movie_name],
            model,
            cinema_encoder,
            movie_encoder,
            features,
            df,
            c_map,
        )

    print("\nAll predictions generated!")


if __name__ == "__main__":
    main()
