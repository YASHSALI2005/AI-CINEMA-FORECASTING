"""
Train ONE Unified Movie+Cinema XGBoost Model
- Loads BMS currently playing movies
- Matches them to SQL dump movie names (picks version with most data)
- Trains one XGBoost model on broad Cinepolis history (all eligible movies + cinemas)
- Keeps a BMS movie-variant map for current prediction/reporting targets
- Saves unified model + encoders + stats + movie variant map
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import re
import warnings
from difflib import get_close_matches
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- CONFIG ---
CSV_FILE = "final_training_data_feb18.csv"
BMS_FILE = "bms_currently_playing.json"
CINEMA_NAMES_FILE = "cinema_names.csv"
MODELS_DIR = "models"
LOGS_DIR = "logs"
GENERIC_MODEL_FILE = "xgb_cinema_model_v5.json"
GENERIC_ENCODER_FILE = "cinema_encoder_v5.pkl"
LOG_FILE = os.path.join(LOGS_DIR, "movie_cinema_unified_training_log.csv")

UNIFIED_MODEL_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_model.json")
UNIFIED_CINEMA_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_cinema_encoder.pkl")
UNIFIED_MOVIE_ENCODER_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_encoder.pkl")
UNIFIED_STATS_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_stats.json")
UNIFIED_MOVIE_MAP_FILE = os.path.join(MODELS_DIR, "movie_cinema_unified_movie_map.json")

MIN_ROWS = 50
MIN_MOVIE_ROWS_FOR_UNIFIED = 30

TODAY = pd.to_datetime("2026-02-18")


def normalize_name(name):
    if not isinstance(name, str):
        return ""
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^a-zA-Z0-9 ]', '', name)
    return name.strip().lower()


def match_bms_to_db(bms_names, db_names, movie_counts):
    """
    Match BMS movie names to database names.
    Picks the variant with the MOST rows and returns ALL related variants.
    """
    db_clean = {normalize_name(m): m for m in db_names}
    matches = {}

    for bms_name in bms_names:
        clean = normalize_name(bms_name)

        # Find ALL db names that contain the BMS name as a substring
        related = []
        for db_clean_name, db_original in db_clean.items():
            if clean in db_clean_name or db_clean_name in clean:
                row_count = movie_counts.get(db_original, 0)
                related.append((db_original, row_count))

        if not related:
            close = get_close_matches(clean, list(db_clean.keys()), n=5, cutoff=0.5)
            for c in close:
                db_original = db_clean[c]
                row_count = movie_counts.get(db_original, 0)
                related.append((db_original, row_count))

        if related:
            related.sort(key=lambda x: x[1], reverse=True)
            primary = related[0][0]
            all_names = [r[0] for r in related]
            matches[bms_name] = {
                'primary': primary,
                'all_variants': all_names,
                'total_rows': sum(r[1] for r in related)
            }
        else:
            print(f"   No match for '{bms_name}'")

    return matches


def load_data():
    print("Loading training data...")
    chunks = pd.read_csv(CSV_FILE, chunksize=200000)
    df = pd.concat(list(chunks), ignore_index=True)
    df['show_time'] = pd.to_datetime(df['show_time'], errors='coerce')
    print(f"   Loaded {len(df)} rows, {df['show_time'].min()} to {df['show_time'].max()}")
    return df


def encode_with_unknown(series, encoder):
    known = set(encoder.classes_)
    return series.astype(str).apply(
        lambda x: int(encoder.transform([x])[0]) if x in known else 0
    )


def prepare_features_unified(df_input, cinema_encoder=None, movie_encoder=None):
    df = df_input.copy()
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([4, 5, 6]).astype(int)
    df['hour'] = df['show_time'].dt.hour
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['date_str'] = df['show_time'].dt.date.astype(str)

    if 'release_date' in df.columns:
        df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
        df['days_since_release'] = (df['show_time'] - df['release_date']).dt.days
        df['days_since_release'] = df['days_since_release'].fillna(0).clip(lower=0)
        df['log_days_since_release'] = np.log1p(df['days_since_release'])
    else:
        df['log_days_since_release'] = 0

    if cinema_encoder is None:
        cinema_encoder = LabelEncoder()
        df['cinema_id_encoded'] = cinema_encoder.fit_transform(df['cinema_id'].astype(str))
    else:
        df['cinema_id_encoded'] = encode_with_unknown(df['cinema_id'], cinema_encoder)

    if movie_encoder is None:
        movie_encoder = LabelEncoder()
        df['movie_id_encoded'] = movie_encoder.fit_transform(df['movie_name'].astype(str))
    else:
        df['movie_id_encoded'] = encode_with_unknown(df['movie_name'], movie_encoder)

    num_cols = ['popularity', 'vote_average', 'competitors_on_screen']
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # Cinema-specific aggregates (KEY per-movie+cinema features)
    df = df.sort_values('show_time')
    cinema_daily = df.groupby(['cinema_id', 'date_str'])['sold_tickets'].sum().reset_index()
    cinema_avg = cinema_daily.groupby('cinema_id')['sold_tickets'].mean().to_dict()
    df['cinema_avg_daily'] = df['cinema_id'].map(cinema_avg).fillna(0)

    cinema_hour_avg = df.groupby(['cinema_id', 'hour'])['sold_tickets'].mean().to_dict()
    df['cinema_hour_avg'] = df.apply(
        lambda x: cinema_hour_avg.get((x['cinema_id'], x['hour']), 0), axis=1
    )

    return df, cinema_encoder, movie_encoder


def train_unified_model(df_train):
    print(f"\nTraining unified model on {len(df_train)} rows...")
    if len(df_train) < MIN_ROWS:
        print(f"   Too few rows ({len(df_train)} < {MIN_ROWS}).")
        return None, None, None, None, None

    df_prepped, cinema_encoder, movie_encoder = prepare_features_unified(df_train)

    features = [
        'day_of_week', 'is_weekend', 'hour_cos',
        'log_days_since_release',
        'cinema_id_encoded', 'movie_id_encoded',
        'competitors_on_screen',
        'cinema_avg_daily', 'cinema_hour_avg',
        'popularity', 'vote_average'
    ]

    target = 'sold_tickets'
    valid = df_prepped[df_prepped[target].notna()].copy()
    for c in features:
        if c not in valid.columns:
            valid[c] = 0
        valid[c] = pd.to_numeric(valid[c], errors='coerce').fillna(0)

    if len(valid) < MIN_ROWS:
        print("   Not enough valid rows.")
        return None, None, None, None, None

    idx_train, idx_test = train_test_split(valid.index, test_size=0.2, random_state=42)
    X_train = valid.loc[idx_train, features]
    y_train = valid.loc[idx_train, target]
    X_test = valid.loc[idx_test, features]
    y_test = valid.loc[idx_test, target]

    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=600,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=0)

    predictions = np.maximum(model.predict(X_test), 0)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    metrics = {
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'r2': round(float(r2), 4),
        'mae': round(float(mae), 2),
        'rmse': round(float(rmse), 2),
        'features': features
    }
    print(f"   Unified R2={r2:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")
    return model, cinema_encoder, movie_encoder, metrics, valid.loc[idx_test].copy()


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)

    # 1. Load BMS movie list
    print("Loading BookMyShow movie list...")
    with open(BMS_FILE) as f:
        bms_movies = json.load(f)
    bms_names = [m['name'] for m in bms_movies]
    print(f"   Found {len(bms_names)} BMS movies")

    # 2. Load training data
    df = load_data()

    # 3. Get all unique DB movie names (recent) + counts
    recent = df[df['show_time'] >= '2025-12-01']
    db_movies = recent['movie_name'].unique().tolist()
    movie_counts = df.groupby('movie_name')['sold_tickets'].count().to_dict()

    # 4. Match BMS to DB names
    print("\nMatching BMS movies to database...")
    matches = match_bms_to_db(bms_names, db_movies, movie_counts)

    print(f"\n{'BMS Name':40s} | {'Primary DB Name':45s} | {'Variants':>8s} | {'Total Rows':>10s}")
    print("-" * 110)
    for bms, info in matches.items():
        print(f"{bms:40s} | {info['primary']:45s} | {len(info['all_variants']):>8d} | {info['total_rows']:>10d}")

    # 5. Build movie-variant map for current BMS targets (prediction/output scope)
    movie_map = {}
    for _, info in matches.items():
        primary_name = info['primary']
        all_variants = info['all_variants']
        movie_map[primary_name] = all_variants

    # 6. Build unified TRAINING set from broad Cinepolis history
    #    Keep only movies with at least MIN_MOVIE_ROWS_FOR_UNIFIED rows to reduce extreme sparsity noise.
    train_df = df.copy()
    train_df = train_df[train_df['movie_name'].notna() & train_df['show_time'].notna()]

    movie_row_counts = train_df.groupby('movie_name')['sold_tickets'].count()
    eligible_movies = set(movie_row_counts[movie_row_counts >= MIN_MOVIE_ROWS_FOR_UNIFIED].index)
    train_df = train_df[train_df['movie_name'].isin(eligible_movies)].copy()

    # Canonicalize currently-playing matched variants into primary names
    variant_to_primary = {}
    for primary_name, variants in movie_map.items():
        for v in variants:
            variant_to_primary[v] = primary_name
    train_df['movie_name'] = train_df['movie_name'].map(lambda x: variant_to_primary.get(x, x))

    if train_df.empty:
        print("No eligible Cinepolis rows found for unified training.")
        return

    unified_df = train_df
    print(f"\nUnified training dataset: {len(unified_df)} rows")
    print(f"Eligible movies for training: {unified_df['movie_name'].nunique()} (min rows/movie={MIN_MOVIE_ROWS_FOR_UNIFIED})")
    print(f"BMS target movies in map: {len(movie_map)}")

    # 7. Train one unified model
    model, cinema_encoder, movie_encoder, metrics, unified_test = train_unified_model(unified_df)
    if model is None:
        print("Unified training failed.")
        return

    # 8. Save unified artifacts
    model.save_model(UNIFIED_MODEL_FILE)
    joblib.dump(cinema_encoder, UNIFIED_CINEMA_ENCODER_FILE)
    joblib.dump(movie_encoder, UNIFIED_MOVIE_ENCODER_FILE)
    with open(UNIFIED_MOVIE_MAP_FILE, 'w') as f:
        json.dump(movie_map, f, indent=2)

    stats = {
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'n_rows': int(len(unified_df)),
        'n_movies': int(unified_df['movie_name'].nunique()),
        'n_cinemas': int(unified_df['cinema_id'].astype(str).nunique()),
        'n_bms_target_movies': int(len(movie_map)),
        'min_movie_rows_for_unified': int(MIN_MOVIE_ROWS_FOR_UNIFIED),
        'metrics': metrics,
        'features': metrics['features'],
        'model_file': UNIFIED_MODEL_FILE,
        'cinema_encoder_file': UNIFIED_CINEMA_ENCODER_FILE,
        'movie_encoder_file': UNIFIED_MOVIE_ENCODER_FILE,
        'movie_map_file': UNIFIED_MOVIE_MAP_FILE
    }
    with open(UNIFIED_STATS_FILE, 'w') as f:
        json.dump(stats, f, indent=2)

    log_df = pd.DataFrame([{
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'unified_movie_cinema',
        'train_rows': metrics['train_rows'],
        'test_rows': metrics['test_rows'],
        'r2': metrics['r2'],
        'mae': metrics['mae'],
        'rmse': metrics['rmse']
    }])
    
    # Append if exists, write with header if new
    log_exists = os.path.isfile(LOG_FILE)
    log_df.to_csv(LOG_FILE, mode='a', header=not log_exists, index=False)

    print("\n" + "=" * 80)
    print("UNIFIED MOVIE+CINEMA MODEL TRAINING SUMMARY")
    print("=" * 80)
    print(f"Rows: {len(unified_df)} | Movies: {unified_df['movie_name'].nunique()} | Cinemas: {unified_df['cinema_id'].nunique()}")
    print(f"BMS target movies: {len(movie_map)}")
    print(f"R2={metrics['r2']:.4f} | MAE={metrics['mae']:.2f} | RMSE={metrics['rmse']:.2f}")
    print(f"Saved model: {UNIFIED_MODEL_FILE}")
    print(f"Saved encoders: {UNIFIED_CINEMA_ENCODER_FILE}, {UNIFIED_MOVIE_ENCODER_FILE}")
    print(f"Saved movie map: {UNIFIED_MOVIE_MAP_FILE}")
    print(f"Saved stats: {UNIFIED_STATS_FILE}")
    print(f"Saved log: {LOG_FILE}")

    # 9. a unified with generic model
    print("\nComparing with Generic Model...")
    try:
        generic_model = xgb.XGBRegressor()
        generic_model.load_model(GENERIC_MODEL_FILE)
        generic_encoder = joblib.load(GENERIC_ENCODER_FILE)

        generic_features = [
            'budget', 'runtime', 'popularity', 'vote_average',
            'day_of_week', 'is_weekend', 'hour', 'holiday_weight',
            'competitors_on_screen', 'log_days_since_release',
            'cinema_id_encoded', 'movie_trend_7d', 'cinema_trend_7d',
            'bh_opening_day', 'bh_verdict_score'
        ]

        generic_eval = unified_test.copy()
        generic_eval['holiday_weight'] = 0
        generic_eval['cinema_id_encoded'] = generic_eval['cinema_id'].astype(str).apply(
            lambda x: int(generic_encoder.transform([x])[0]) if x in set(generic_encoder.classes_) else 0
        )

        for c in generic_features:
            if c not in generic_eval.columns:
                generic_eval[c] = 0
            generic_eval[c] = pd.to_numeric(generic_eval[c], errors='coerce').fillna(0)

        X_generic = generic_eval[generic_features]
        y_generic = pd.to_numeric(generic_eval['sold_tickets'], errors='coerce').fillna(0)

        gen_preds = np.maximum(generic_model.predict(X_generic), 0)
        gen_r2 = r2_score(y_generic, gen_preds)
        gen_mae = mean_absolute_error(y_generic, gen_preds)
        print(f"   Generic  -> R2={gen_r2:.4f}, MAE={gen_mae:.2f}")
        print(f"   Unified  -> R2={metrics['r2']:.4f}, MAE={metrics['mae']:.2f}")
        improvement = ((gen_mae - metrics['mae']) / gen_mae) * 100 if gen_mae > 0 else 0
        print(f"   MAE Improvement: {improvement:+.1f}%")
    except Exception as e:
        print(f"   Generic model comparison failed: {e}")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
