import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import requests
import os
import re
from datetime import datetime, timedelta, time
from dotenv import load_dotenv

# Load Environment Variables
load_dotenv()
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# --- CONFIGURATION ---
MODEL_FILE = "xgb_cinema_model_v5.json"
ENCODER_FILE = "cinema_encoder_v5.pkl"
BH_DATA_FILE = "bh_box_office_data.json"
HISTORY_CSV = "final_training_data_from_dump.csv"
CINEMA_NAMES_CSV = "cinema_names.csv"
OUTPUT_FILE = "New_Movies_Predictions_Report_v10.xlsx"

# Simulation Cinemas (Codes)
# Picking Top 5 diverse cinemas for simulation
SIM_CINEMAS = [754, 621, 368, 1326, 740] # Andheri, Inorbit, Viviana, Airia, DB City

START_TIME = time(9, 0) # 9:00 AM
END_TIME = time(23, 30) # 11:30 PM
INTERVAL_MINUTES = 30

def get_date_ranges():
    """Dynamic Date Ranges based on Today."""
    today = datetime.now().date()
    forecast_end = today + timedelta(days=5)
    
    # Range 1: Historical / Recents (Fixed in logic or dynamic?)
    # Reference script used Jan 25 - Feb 05. We can keep that for context.
    # Range 2: Forecast (From "Tomorrow" or "Today" -> Today+5)
    
    # Let's align with user request: "date range same as report... and for future take date from today till 5 days more"
    # Report ranges:
    r1_start = pd.to_datetime("2026-01-25").date()
    r1_end = pd.to_datetime("2026-02-05").date()
    
    r2_start = pd.to_datetime("2026-02-06").date()
    # If today is later than Feb 6, extend end to today+5
    # If today is earlier, stick to Feb 15?
    # User said: "from today till 5 days more"
    
    r2_end_base = pd.to_datetime("2026-02-15").date()
    r2_end = max(r2_end_base, forecast_end)
    
    return [
        (r1_start, r1_end, "Jan 25 - Feb 05"),
        (r2_start, r2_end, f"Feb 06 - {r2_end.strftime('%b %d')} (Forecast)")
    ]

def load_resources():
    print("Loading model and resources...")
    try:
        model = xgb.XGBRegressor()
        model.load_model(MODEL_FILE)
        encoder = joblib.load(ENCODER_FILE)
        
        # Load Cinema Names
        c_names = pd.read_csv(CINEMA_NAMES_CSV)
        # Map Code -> Name (and handle NaNs)
        c_map = dict(zip(c_names['cinema_code'].fillna(0).astype(int).astype(str), c_names['cinema_name']))
        
        # Load History for Actuals Lookup
        # Columns in dump: movie_name, cinema_id (which is code), show_time, sold_tickets, capacity
        # We need check which columns physically exist
        # Using 'cinema_id' as matches the dump
        history_cols = ['cinema_id', 'show_time', 'sold_tickets', 'original_name', 'capacity']
        try:
            history_df = pd.read_csv(HISTORY_CSV, usecols=history_cols)
            history_df['show_time'] = pd.to_datetime(history_df['show_time'])
            # Normalize names
            history_df['clean_name'] = history_df['original_name'].astype(str).apply(lambda x: re.sub(r'\s*\(.*?\)', '', x).strip().lower())
            
            # Ensure cinema_id is str for matching
            history_df['cinema_id'] = history_df['cinema_id'].astype(str).str.replace('.0', '')
            
        except Exception as e:
            print(f"Warning: Could not load history CSV fully ({e}). Actuals will be empty.")
            history_df = pd.DataFrame()

        with open(BH_DATA_FILE, 'r', encoding='utf-8') as f:
            bh_data = json.load(f)
            
        return model, encoder, history_df, bh_data, c_map
    except Exception as e:
        print(f"Error loading resources: {e}")
        return None, None, None, None, None

def get_holiday_weight(date_obj):
    try:
        from holiday_utils import get_holiday_weight
        return get_holiday_weight(date_obj)
    except:
        if date_obj.weekday() >= 5: return 0.5
        return 0.0

def fetch_omdb_details(movie_name):
    clean_title = re.sub(r'\s*\(.*?\)', '', movie_name).strip()
    defaults = {
        "budget": 100000000,
        "runtime": 120,
        "popularity": 50,
        "vote_average": 7.0,
        "genres": "Drama"
    }
    
    if not OMDB_API_KEY:
        return defaults

    try:
        url = f"http://www.omdbapi.com/?t={clean_title}&apikey={OMDB_API_KEY}"
        resp = requests.get(url, timeout=5)
        data = resp.json()
        
        if data.get("Response") == "True":
            box_office = data.get("BoxOffice", "N/A")
            budget_val = int(box_office.replace("$", "").replace(",", "")) if box_office != "N/A" else 100000000
            
            runtime_str = data.get("Runtime", "120 min")
            runtime_val = int(runtime_str.split(" ")[0]) if "min" in runtime_str else 120
            
            votes_str = data.get("imdbVotes", "5,000")
            votes_val = int(votes_str.replace(",", "")) if votes_str != "N/A" else 5000
            popularity_val = min(votes_val / 1000, 100) 

            rating_str = data.get("imdbRating", "7.0")
            rating_val = float(rating_str) if rating_str != "N/A" else 7.0

            return {
                "budget": budget_val,
                "runtime": runtime_val,
                "popularity": popularity_val,
                "vote_average": rating_val,
                "genres": data.get("Genre", "Unknown")
            }
            
    except Exception as e:
        print(f"OMDb Fetch Error for {movie_name}: {e}")
        
    return defaults

def generate_slots_for_range(start_date, end_date):
    slots = []
    current_date = start_date
    while current_date <= end_date:
        current_time = datetime.combine(current_date, START_TIME)
        end_time_dt = datetime.combine(current_date, END_TIME)
        
        while current_time <= end_time_dt:
            slots.append(current_time)
            current_time += timedelta(minutes=INTERVAL_MINUTES)
            
        current_date += timedelta(days=1)
    return slots

def find_actuals(history_df, movie_name, slot_dt, cinema_code):
    """Find actual sales if available in history for this specific slot (+/- 15 mins) AND Cinema."""
    if history_df.empty:
        return None
        
    clean_name = re.sub(r'\s*\(.*?\)', '', movie_name).strip().lower()
    c_code_str = str(cinema_code)
    
    start_window = slot_dt - timedelta(minutes=15)
    end_window = slot_dt + timedelta(minutes=15)
    
    mask = (
        (history_df['clean_name'] == clean_name) & 
        (history_df['cinema_id'] == c_code_str) &
        (history_df['show_time'] >= start_window) &
        (history_df['show_time'] <= end_window)
    )
    
    matches = history_df[mask]
    if not matches.empty:
        total_sold = matches['sold_tickets'].sum()
        total_cap = matches['capacity'].sum()
        if total_cap > 0:
            return (total_sold / total_cap) * 100
        else:
            return 0
            
    return None

def main():
    model, encoder, history_df, bh_data, c_map = load_resources()
    if not model:
        return

    # 1. Identify Top 10 Movies
    movies_list = bh_data[:10]
    print(f"Generating reports for {len(movies_list)} movies...")

    writer = pd.ExcelWriter(OUTPUT_FILE, engine='openpyxl')
    
    date_ranges = get_date_ranges()
    
    first_movie = True
    start_row = 0
    
    sheet_name = "Predictions"
    
    # --- Trend Pre-Calculation ---
    print("Calculating Data-Driven Trends from History...")
    if not history_df.empty:
        # Group by movie (clean_name)
        movie_trend_map = history_df.groupby('clean_name')['sold_tickets'].mean().to_dict()
        
        # Group by cinema (cinema_id string)
        cinema_trend_map = history_df.groupby('cinema_id')['sold_tickets'].mean().to_dict()
        
        global_avg_tickets = history_df['sold_tickets'].mean()
    else:
        movie_trend_map = {}
        cinema_trend_map = {}
        global_avg_tickets = 0
        
    print(f"Global Avg Tickets/Show: {global_avg_tickets:.2f}")

    for movie_entry in movies_list:
        original_name = movie_entry.get('original_name', 'Unknown')
        print(f"Processing: {original_name}")
        
        clean_name = re.sub(r'\s*\(.*?\)', '', original_name).strip().lower()
        
        # Pre-filter history for this movie
        sim_cinema_strs = [str(c) for c in SIM_CINEMAS]
        movie_history = history_df[
            (history_df['clean_name'] == clean_name) & 
            (history_df['cinema_id'].isin(sim_cinema_strs))
        ].copy()
        
        if not movie_history.empty:
            movie_history = movie_history.sort_values('show_time')
            movie_history['Cinema Code'] = movie_history['cinema_id'].astype(int)
        
        meta = fetch_omdb_details(original_name)
        
        bh_opening = movie_entry.get('summary', {}).get('opening_day', '0')
        try:
             bh_opening_val = float(re.sub(r'[^\d.]', '', bh_opening))
        except:
             bh_opening_val = 0
        
        all_period_data = []

        for start_dt, end_dt, label in date_ranges:
            slots = generate_slots_for_range(start_dt, end_dt)
            
            for slot_dt in slots:
                hour = slot_dt.hour
                day_of_week = slot_dt.weekday()
                is_weekend = 1 if day_of_week >= 5 else 0
                holiday_w = get_holiday_weight(slot_dt.date())
                
                for c_code in SIM_CINEMAS:
                    # Encoder transform
                    try:
                        c_encoded = encoder.transform([c_code])[0]
                    except:
                        try:
                            c_encoded = encoder.transform([str(c_code)])[0]
                        except:
                            c_encoded = 0
                    
                    # Trends Calculation
                    # Cinema Trend: Average tickets per show for this cinema in last 7 days (or all history provided)
                    # The history_df is likely full history.
                    # We can pre-calculate global dictionaries outside loops for speed.
                    
                    # For New Movies (Mardaani 3), movie_trend_7d will be NaN -> fill with Global Avg?
                    # User said: "Calculated as Average Tickets Sold Per Show for that specific movie over the last 7 days."
                    # If 0 days released, trend is 0? Or maybe assume average movie hype?
                    # The reference script fills NaN with Global Avg.
                    
                    p_movie_trend = movie_trend_map.get(clean_name, global_avg_tickets)
                    p_cinema_trend = cinema_trend_map.get(str(c_code), global_avg_tickets)
                    
                    # Clip as per reference
                    p_movie_trend = min(p_movie_trend, 400)
                    p_cinema_trend = min(p_cinema_trend, 400)

                    row = {
                        'budget': meta['budget'],
                        'runtime': meta['runtime'],
                        'popularity': meta['popularity'],
                        'vote_average': meta['vote_average'],
                        'day_of_week': day_of_week,
                        'is_weekend': is_weekend,
                        'hour': hour,
                        'holiday_weight': holiday_w,
                        'competitors_on_screen': 5, 
                        'log_days_since_release': np.log1p(2),
                        'cinema_id_encoded': c_encoded,
                        'movie_trend_7d': p_movie_trend,
                        'cinema_trend_7d': p_cinema_trend,
                        'bh_opening_day': bh_opening_val,
                        'bh_verdict_score': 0,
                        
                        'Showtime': slot_dt.strftime("%Y-%m-%d %H:%M"),
                        'Cinema Name': c_map.get(str(c_code), f"Cinema {c_code}"),
                        'Period': label,
                        'Slot Datetime': slot_dt,
                        'Cinema Code': c_code
                    }
                    all_period_data.append(row)

        if not all_period_data:
            continue
            
        df_batch = pd.DataFrame(all_period_data)
        
        # Predict
        feature_cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
                        'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
                        'log_days_since_release', 'cinema_id_encoded', 'movie_trend_7d', 
                        'cinema_trend_7d', 'bh_opening_day', 'bh_verdict_score']
        
        X = df_batch[feature_cols]
        preds = model.predict(X)
        
        df_batch['Raw Pred'] = preds
        
        # Prediction Logic with 20% Gap
        # formula: pred_low = (raw * 0.55 / 300) * 100
        # range: low to low + 20
        
        def calculate_range(raw_pred):
            pred_low_val = max((raw_pred * 0.55 / 300) * 100, 0)
            pred_high_val = min(pred_low_val + 20, 100) # 20% Gap
            
            if pred_high_val >= 100:
                pred_high_val = 100
                pred_low_val = max(100 - 20, 0)
                
            return f"{round(pred_low_val)}-{round(pred_high_val)}%"

        df_batch['Pred %'] = df_batch['Raw Pred'].apply(calculate_range)
        
        # Merge Actuals
        df_batch['Actual %'] = np.nan
        
        if not movie_history.empty:
            df_batch = df_batch.sort_values('Slot Datetime')
            merged = pd.merge_asof(
                df_batch, 
                movie_history[['show_time', 'Cinema Code', 'sold_tickets', 'capacity']], 
                left_on='Slot Datetime', 
                right_on='show_time', 
                by='Cinema Code', 
                tolerance=pd.Timedelta(minutes=15),
                direction='nearest'
            )
            merged['Actual %'] = (merged['sold_tickets'] / merged['capacity']) * 100
            df_batch = merged
        
        
        # Sort Logic: Group by Cinema (9am-11:30pm), Priority slightly different
        # 1. Cinemas WITH Actuals should appear first (their entire schedule block).
        # 2. Then Cinemas WITHOUT Actuals. (Alphabetical or Code order)
        # 3. Within each Cinema block, sort by Showtime (asc).
        
        # Calculate 'Has_Data' per cinema
        # Group by Cinema Code to check if any 'Actual %' is valid
        # We can use transform on the boolean mask
        
        merged['is_valid_actual'] = merged['Actual %'].notna() & (merged['Actual %'] != "")
        # Note: 'Actual %' is float or NaN before formatting? No, we merge as float/NaN.
        # Wait, merged['Actual %'] is float/NaN.
        
        # Calculate flag per cinema
        merged['Cinema_Has_Data'] = merged.groupby('Cinema Code')['Actual %'].transform(lambda x: x.notna().any().astype(int))
        
        # Sort
        # First: Cinema_Has_Data (Desc -> 1s first)
        # Second: Cinema Name (Asc -> Alphabetical Grouping)
        # Third: Slot Datetime (Asc -> Chronological)
        
        merged = merged.sort_values(
            by=['Cinema_Has_Data', 'Cinema Name', 'Slot Datetime'], 
            ascending=[False, True, True]
        )
        
        df_batch = merged
        
        # Formatting
        def safe_format_pct(x):
            try:
                if pd.isna(x) or x == "":
                    return ""
                return f"{float(x):.1f}%"
            except:
                return str(x)

        df_batch['Actual %'] = df_batch['Actual %'].apply(safe_format_pct)
        
        # Select Final Columns (drop temp cols implicitly by selection)
        final_df = df_batch[['Showtime', 'Cinema Name', 'Pred %', 'Actual %']]
        
        # Write to Excel with Header
        # If first movie, start at row 0? No, let's use pandas startrow
        
        # We need to write the Movie Title first
        # We can create a 1-row dataframe or use openpyxl directly?
        # Using pandas with startrow is easiest.
        
        # Header DF
        header_df = pd.DataFrame({f"MOVIE: {original_name}": []})
        
        # Write Header
        header_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        start_row += 1
        
        # Write Data
        final_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=False)
        
        # Update start_row for next movie
        start_row += len(final_df) + 3 # +1 for header, +2 for spacing
        
    writer.close()

    print(f"Report saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
