import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
from datetime import datetime, timedelta
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# --- CONFIG ---
CSV_FILE = "final_training_data_from_dump.csv"
CINEMA_NAMES_FILE = "cinema_names.csv"
MODEL_FILE = "xgb_cinema_model_v5.json"
ENCODER_FILE = "cinema_encoder_v5.pkl"

# --- MANUAL RELEASE DATES (STRICT OVERRIDE) ---
# Format: "YYYY-MM-DD"
MANUAL_RELEASE_DATES = {
    "BORDER 2 (HINDI)": "2026-01-23",
    "DHURANDHAR (HINDI)": "2025-12-05"
}

# --- 100% DATA-DRIVEN TREND LOGIC ---
def calculate_data_driven_trends(df, reference_date):
    # Define the 7-day window BEFORE the reference date
    start_window = reference_date - timedelta(days=7)
    end_window = reference_date
    
    # Filter data for the last 7 days
    mask = (df['show_time'] >= start_window) & (df['show_time'] < end_window)
    recent_data = df[mask].copy()
    
    if recent_data.empty:
        recent_data = df.copy()

    movie_hype = recent_data.groupby('movie_name')['sold_tickets'].mean().to_dict()
    cinema_status = recent_data.groupby('cinema_id')['sold_tickets'].mean().to_dict()
    global_avg = recent_data['sold_tickets'].mean()
    
    return movie_hype, cinema_status, global_avg

def load_resources():
    print("Loading model and resources...")
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_FILE)
        encoder = joblib.load(ENCODER_FILE)
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return None, None, None, None

    try:
        c_names = pd.read_csv(CINEMA_NAMES_FILE)
        def clean_code(x):
            try: return str(int(float(x)))
            except: return str(x)

        c_names['cinema_id'] = c_names['cinema_id'].apply(clean_code)
        c_names['cinema_code'] = c_names['cinema_code'].apply(clean_code)
        
        map_id = dict(zip(c_names['cinema_id'], c_names['cinema_name']))
        map_code = dict(zip(c_names['cinema_code'], c_names['cinema_name']))
        c_map = {**map_code, **map_id} 
    except:
        c_map = {}
        
    return model, encoder, df, c_map

def get_holiday_weight(d):
    return 0.0

def round_time_to_nearest_30(t):
    if pd.isnull(t): return t
    minute = t.minute
    if minute < 15: minute = 0
    elif minute < 45: minute = 30
    else:
        minute = 0
        t += timedelta(hours=1)
    return t.replace(minute=minute, second=0, microsecond=0)

def prepare_features(df, encoder):
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['date_obj'] = df['show_time'].dt.date
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour'] = df['show_time'].dt.hour
    
    unique_dates = df['date_obj'].unique()
    hol_map = {d: get_holiday_weight(d) for d in unique_dates}
    df['holiday_weight'] = df['date_obj'].map(hol_map)
    
    unique_cids = df['cinema_id'].astype(str).unique()
    cid_map = {}
    for cid in unique_cids:
        try: cid_map[cid] = encoder.transform([cid])[0]
        except: cid_map[cid] = 0
            
    df['cinema_id_encoded'] = df['cinema_id'].astype(str).map(cid_map).fillna(0)
    
    cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
            'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
            'log_days_since_release', 'movie_trend_7d', 'cinema_trend_7d',
            'bh_opening_day', 'bh_verdict_score']
            
    for c in cols:
        if c not in df.columns: df[c] = 0
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    return df

def generate_grid_schedule_vectorized(movie_name, movie_meta, unique_cinemas, start_dt, end_dt):
    # Ensure inputs are Datetime objects
    start_dt = pd.to_datetime(start_dt)
    end_dt = pd.to_datetime(end_dt)

    dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
    times = pd.date_range("2022-01-01 09:00", "2022-01-01 23:30", freq="30min").time
    
    full_timestamps = [datetime.combine(d, t) for d in dates for t in times]
    df_time = pd.DataFrame({'show_time': full_timestamps})
    df_cinema = pd.DataFrame({'cinema_id': unique_cinemas})
    
    grid = df_cinema.merge(df_time, how='cross')

    grid['movie_name'] = movie_name
    for col, val in movie_meta.items():
        if col != 'release_date':
            grid[col] = val
            
    if movie_meta.get('release_date'):
        rel_dt = pd.to_datetime(movie_meta['release_date'])
        grid['days_valid'] = (grid['show_time'] - rel_dt).dt.days
        grid['log_days_since_release'] = np.log1p(grid['days_valid'].clip(lower=0))
    else:
        grid['log_days_since_release'] = 0
        
    return grid

def process_movie_excel(movie_name, df_full, model, encoder, c_map):
    print(f"Processing Excel for: {movie_name} ...")
    
    movie_slice = df_full[df_full['movie_name'] == movie_name]
    # Even if movie not in CSV, we might want to generate hypothetical forecast
    # But we need metadata. If empty, we can't do much.
    if movie_slice.empty: 
        print(f"   [!] Movie '{movie_name}' not found in data. Skipping.")
        return
    
    meta_row = movie_slice.sort_values('show_time', ascending=False).iloc[0]
    
    # --- DATE SELECTION LOGIC ---
    if movie_name in MANUAL_RELEASE_DATES:
        start_date = pd.to_datetime(MANUAL_RELEASE_DATES[movie_name])
        print(f"   -> !!! USING MANUAL OVERRIDE DATE: {start_date.date()} !!!")
    else:
        release_date_raw = meta_row.get('release_date', None)
        if pd.isna(release_date_raw):
            start_date = pd.to_datetime(datetime.now().date())
            print(f"   -> [!] No date found. Defaulting to Today: {start_date.date()}")
        else:
            start_date = pd.to_datetime(release_date_raw)
            print(f"   -> Using CSV Date: {start_date.date()}")

    end_date = start_date + timedelta(days=15)
    print(f"   -> Grid Range: {start_date.date()} to {end_date.date()}")

    # --- TRENDS ---
    movie_hype, cinema_status, global_avg = calculate_data_driven_trends(df_full, reference_date=start_date)
    specific_movie_trend = movie_hype.get(movie_name, global_avg)

    meta_dict = {
        'budget': meta_row.get('budget', 0),
        'runtime': meta_row.get('runtime', 0),
        'popularity': meta_row.get('popularity', 0),
        'vote_average': meta_row.get('vote_average', 0),
        'release_date': start_date, # Explicitly passed
        'competitors_on_screen': meta_row.get('competitors_on_screen', 0),
        'movie_trend_7d': specific_movie_trend, 
        'bh_opening_day': meta_row.get('bh_opening_day', 0),
        'bh_verdict_score': meta_row.get('bh_verdict_score', 0)
    }
    
    unique_cinemas = movie_slice['cinema_id'].unique()
    cinema_capacity_map = df_full.groupby('cinema_id')['capacity'].median().to_dict()
    
    # 2. Grid Generation
    grid_df = generate_grid_schedule_vectorized(movie_name, meta_dict, unique_cinemas, start_date, end_date)
    
    grid_df['cinema_trend_7d'] = grid_df['cinema_id'].map(cinema_status).fillna(global_avg)
    
    # 3. Prediction
    grid_df = prepare_features(grid_df, encoder)
    feature_cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
                    'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
                    'log_days_since_release', 'cinema_id_encoded', 'movie_trend_7d', 
                    'cinema_trend_7d', 'bh_opening_day', 'bh_verdict_score']
    
    grid_df['Raw Prediction'] = model.predict(grid_df[feature_cols])
    
    grid_df['est_capacity'] = grid_df['cinema_id'].map(cinema_capacity_map).fillna(250)
    grid_df['base_pct'] = (grid_df['Raw Prediction'] / grid_df['est_capacity']) * 100
    grid_df['base_pct'] = grid_df['base_pct'].clip(0, 100)
    
    low_val = grid_df['base_pct'].round().astype(int)
    high_val = (grid_df['base_pct'] + 20).clip(upper=100).round().astype(int)
    grid_df['Pred Range'] = low_val.astype(str) + "-" + high_val.astype(str) + "%"

    # 4. Actuals Merge
    actuals = movie_slice.copy()
    actuals['show_time'] = pd.to_datetime(actuals['show_time'])
    actuals['rounded_time'] = actuals['show_time'].apply(round_time_to_nearest_30)
    
    actuals['capacity'] = pd.to_numeric(actuals['capacity'], errors='coerce').fillna(300)
    actuals['sold_tickets'] = pd.to_numeric(actuals['sold_tickets'], errors='coerce').fillna(0)
    actuals['actual_pct'] = (actuals['sold_tickets'] / actuals['capacity']) * 100
    actuals['actual_pct'] = actuals['actual_pct'].clip(0, 100)
    
    actuals_subset = actuals[['rounded_time', 'cinema_id', 'actual_pct']].copy()
    actuals_subset.columns = ['show_time', 'cinema_id', 'Actual %']
    actuals_subset = actuals_subset.groupby(['show_time', 'cinema_id'], as_index=False).max()
    
    final_df = pd.merge(grid_df, actuals_subset, on=['show_time', 'cinema_id'], how='left')
    
    def format_actual(x):
        if pd.isna(x): return ""
        return f"{x:.1f}%"

    final_df['Actual Display'] = final_df['Actual %'].apply(format_actual)
    
    # 5. Final Output
    final_df['Date'] = final_df['show_time'].dt.date
    final_df['Time'] = final_df['show_time'].dt.strftime('%H:%M')
    final_df['Cinema Name'] = final_df['cinema_id'].astype(str).map(c_map).fillna(final_df['cinema_id'])
    
    output_cols = ['Date', 'Time', 'Cinema Name', 'Pred Range', 'Actual Display']
    output_df = final_df[output_cols].rename(columns={'Pred Range': 'Pred %', 'Actual Display': 'Actual %'})
    output_df = output_df.sort_values(by=['Date', 'Cinema Name', 'Time'])
    
    clean_title = re.sub(r'[\\/*?:\'\"<>|]', "", movie_name).replace(" ", "_")[:30]
    fname = f"Pred_{clean_title}.xlsx"
    
    # Auto-delete old file to avoid confusion
    if os.path.exists(fname):
        try:
            os.remove(fname)
            print(f"   -> Deleted old file: {fname}")
        except PermissionError:
            print(f"   [!] WARNING: Cannot delete {fname}. It is OPEN.")
            input("   >>> Please CLOSE the Excel file and press ENTER...")

    while True:
        try:
            output_df.to_excel(fname, index=False)
            print(f"   -> Saved: {fname}")
            break
        except PermissionError:
            print(f"   [!] ERROR: Could not save '{fname}'. It is currently open.")
            input("   >>> Please CLOSE the Excel file and press ENTER to retry...")

def main():
    model, encoder, df, c_map = load_resources()
    if df is None: return

    df['show_time'] = pd.to_datetime(df['show_time'])
    
    # Keys must match MANUAL_RELEASE_DATES exactly
    target_movies = ["DHURANDHAR (HINDI)", "BORDER 2 (HINDI)"]
    
    print(f"Targets: {target_movies}")
    print("-" * 50)
    
    for m in target_movies:
        process_movie_excel(m, df, model, encoder, c_map)

if __name__ == "__main__":
    main()