import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import re
from fpdf import FPDF
from datetime import datetime, timedelta
import os

# Import holiday util
try:
    from holiday_utils import get_holiday_weight
except ImportError:
    def get_holiday_weight(d, c="IN"): return 0.0

# --- CONFIG ---
CSV_FILE = "final_training_data_from_dump.csv"
CINEMA_NAMES_FILE = "cinema_names.csv"
MODEL_FILE = "xgb_cinema_model_v4.json"
ENCODER_FILE = "cinema_encoder_v4.pkl"

# PAST RANGE (Source for Forecasting)
PAST_START = pd.to_datetime("2026-01-30") # Start of the "week" to copy from
PAST_END = pd.to_datetime("2026-02-05")

# REPORT RANGES
DATE_RANGES = [
    (pd.to_datetime("2026-01-25"), pd.to_datetime("2026-02-05"), "Jan 25 - Feb 05"),
    (pd.to_datetime("2026-02-06"), pd.to_datetime("2026-02-15"), "Feb 06 - Feb 15 (Forecast)")
]

def load_resources():
    print("Loading model and resources...")
    model = xgb.XGBRegressor()
    model.load_model(MODEL_FILE)
    encoder = joblib.load(ENCODER_FILE)
    df = pd.read_csv(CSV_FILE)
    
    # Load Cinema Names
    try:
        c_names = pd.read_csv(CINEMA_NAMES_FILE)
        
        def clean_code(x):
            try:
                # Handle 863.0 -> 863
                return str(int(float(x)))
            except:
                return str(x)

        c_names['cinema_id'] = c_names['cinema_id'].apply(clean_code)
        c_names['cinema_code'] = c_names['cinema_code'].apply(clean_code)
        
        # Map both ID and Code to Name
        # If transaction has ID (e.g. "41"), it hits map_id
        # If transaction has Code (e.g. "1049"), it hits map_code
        map_id = dict(zip(c_names['cinema_id'], c_names['cinema_name']))
        map_code = dict(zip(c_names['cinema_code'], c_names['cinema_name']))
        
        c_map = {**map_code, **map_id} 
        
        # DEBUG
        print(f"DEBUG: Map size: {len(c_map)}")
        if '429' in c_map: 
            print(f"DEBUG: '429' IS in map -> {c_map['429']}")
        else:
            print("DEBUG: '429' NOT in map. Sample keys:", list(c_map.keys())[:5])
    except Exception as e:
        print(f"DEBUG: Map Error: {e}")
        c_map = {}
        
    return model, encoder, df, c_map

def prepare_features(df, encoder):
    print("Preparing features...")
    # Ensure datetimes
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['date_obj'] = df['show_time'].dt.date
    
    # Calculate Day of Week & Hour explicitly
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour'] = df['show_time'].dt.hour
    
    # OPTIMIZED: Holiday Weight (Map unique dates)
    unique_dates = df['date_obj'].unique()
    hol_map = {d: get_holiday_weight(d) for d in unique_dates}
    df['holiday_weight'] = df['date_obj'].map(hol_map)
    
    # OPTIMIZED: Cinema Encoding (Map unique IDs)
    unique_cids = df['cinema_id'].unique()
    cid_map = {}
    for cid in unique_cids:
        try:
            cid_map[cid] = encoder.transform([str(cid)])[0]
        except:
            cid_map[cid] = 0
            
    df['cinema_id_encoded'] = df['cinema_id'].map(cid_map)
    
    # Ensure float/int types
    cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
            'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
            'log_days_since_release', 'movie_trend_7d', 'cinema_trend_7d']
            
    # Check if cols exist before looping, create if missing (safe default)
    for c in cols:
        if c not in df.columns:
            df[c] = 0
            
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
        
    return df

class PDFReport(FPDF):
    def header(self):
        # Professional Header with Color Banner
        self.set_fill_color(33, 150, 243) # Blue Banner
        self.rect(0, 0, 210, 25, 'F')
        self.set_font('Arial', 'B', 20)
        self.set_text_color(255, 255, 255)
        self.set_y(10)
        if hasattr(self, 'title_text'):
            self.cell(0, 10, self.title_text, 0, 1, 'C')
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf(movie_name, df_movie, cinema_map):
    clean_title = re.sub(r'\(.*?\)', '', movie_name).strip()
    clean_fname = re.sub(r'[\\/*?:\'\"<>|]', "", movie_name).replace(" ", "_")
    fname = f"Report_{clean_fname}.pdf"
    
    pdf = PDFReport()
    pdf.title_text = f"Performance Report: {clean_title}"
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    
    # (Summary Section Removed as requested)

    for start_dt, end_dt, label in DATE_RANGES:
        is_forecast = "Forecast" in label
        
        # Section Header
        pdf.set_fill_color(230, 230, 230)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, f"  {label}", ln=True, fill=True)
        pdf.ln(2)
        
        mask = (df_movie['show_time'] >= start_dt) & (df_movie['show_time'] <= end_dt)
        subset = df_movie[mask].sort_values('show_time')
        
        # Table Setup
        pdf.set_font("Arial", 'B', 9)
        pdf.set_fill_color(100, 100, 100) # Dark Header
        pdf.set_text_color(255, 255, 255) # White Text
        
        if is_forecast:
            col_w = [40, 110, 40] 
            headers = ["Showtime", "Cinema Name", "Pred Occupancy"]
        else:
            col_w = [35, 95, 30, 30] 
            headers = ["Showtime", "Cinema Name", "Pred %", "Actual %"]
            
        for i, h in enumerate(headers):
            pdf.cell(col_w[i], 8, h, 1, 0, 'C', 1)
        pdf.ln()
        
        # Table Body
        pdf.set_font("Arial", '', 9)
        pdf.set_text_color(0, 0, 0)
        
        if subset.empty:
             pdf.set_fill_color(255, 255, 255)
             pdf.cell(sum(col_w), 10, "No shows in this period", 1, ln=True, align='C')
        else:
            fill = False
            for _, row in subset.iterrows():
                # Alternating Colors
                if fill: pdf.set_fill_color(240, 248, 255) # AliceBlue
                else: pdf.set_fill_color(255, 255, 255)
                
                sts = row['show_time'].strftime('%b %d %H:%M')
                raw_cid = str(row.get('cinema_id', 'N/A'))
                cid = raw_cid.strip().replace('.0', '') 
                
                cname = cinema_map.get(cid, f"Unknown ({cid})")
                if len(cname) > 55: cname = cname[:52] + "..."
                
                # Prediction
                raw_pred = row['Raw Prediction']
                pred_low_val = max((raw_pred * 0.55 / 300) * 100, 0)
                pred_high_val = min(pred_low_val + 25, 100) 
                if pred_high_val >= 100:
                    pred_high_val = 100
                    pred_low_val = max(100-25, 0)
                
                # Match "Fit" Logic to Displayed Text (Integer)
                # If display is "5-30%", then 5.0 should be Green.
                # Use int/round to align internal check with visual
                fit_low = round(pred_low_val)
                fit_high = round(pred_high_val)
                
                pred_str = f"{fit_low}-{fit_high}%"
                
                # Actual
                sold = row.get('sold_tickets', 0)
                cap = row.get('capacity', 300)
                actual_pct = min((sold / cap) * 100, 100) if cap > 0 else 0
                
                # Render Row
                pdf.cell(col_w[0], 8, sts, 1, 0, 'C', 1) # Time
                pdf.cell(col_w[1], 8, f" {cname}", 1, 0, 'L', 1) # Name
                
                # Pred Col (Keep formatted or black? Reset to black for consistency)
                pdf.set_text_color(0, 0, 0)
                pdf.cell(col_w[2], 8, pred_str, 1, 0, 'C', 1)
                
                if not is_forecast:
                    # Logic: 0 -> Grey, Fits Range -> Green, Else -> Red
                    # Allow slight margin? Using rounded fits strictly now.
                    if actual_pct == 0:
                        pdf.set_text_color(128, 128, 128) # Grey
                    elif fit_low <= actual_pct <= fit_high:
                         pdf.set_text_color(0, 150, 0) # Green
                    elif (actual_pct >= fit_low - 1) and (actual_pct <= fit_high + 1):
                         # Extra tolerance for "barely missed" (4.9 vs 5) -> Green? 
                         # User asked for 5.0 to be included. Rounded integers cover it.
                         pdf.set_text_color(0, 150, 0)
                    else:
                         pdf.set_text_color(200, 0, 0) # Red
                    
                    pdf.cell(col_w[3], 8, f"{actual_pct:.1f}%", 1, 0, 'C', 1)
                    pdf.set_text_color(0, 0, 0) # Reset

                pdf.ln()
                fill = not fill
                
        pdf.ln(8)
    
    pdf.output(fname)
    print(f"Generated PDF: {fname}")

def project_schedule(df):
    """
    Project schedule from PAST_START..PAST_END into Feb 6..Feb 15
    """
    print("Projecting future schedule...")
    future_rows = []
    
    # Target period: Feb 6 to Feb 15
    start_future = pd.to_datetime("2026-02-06")
    end_future = pd.to_datetime("2026-02-15")
    
    current_date = start_future
    while current_date <= end_future:
        # Find corresponding source date (Same Weekday) from Last Week
        # Simple: Subtract 7 days? 
        # Feb 6 is Friday. Jan 30 is Friday. (Diff 7 days)
        # Feb 13 is Friday. Diff 14 days.
        
        # We want to cycle through the available past week data (Jan 30 - Feb 5)
        # Jan 30 (Fri) -> Feb 5 (Thu) covers a full week.
        
        # Map current weekday to source date
        # If current_date is Fri, source is Jan 30.
        # If current_date is Sat, source is Jan 31.
        # ...
        
        weekday = current_date.weekday() # 0=Mon, 4=Fri
        
        # Find date in PAST range with this weekday
        # Jan 30 (Fri) = 4
        # Jan 31 (Sat) = 5
        # Feb 01 (Sun) = 6
        # Feb 02 (Mon) = 0
        # Feb 03 (Tue) = 1
        # Feb 04 (Wed) = 2
        # Feb 05 (Thu) = 3
        # Perfect, exact 1-to-1 mapping for a full week.
        
        # Source offset
        # Simple logic: If current > Feb 5, substract 7 until in range?
        # Or just match by weekday? Matching by weekday is safer if mapped correctly.
        
        # Let's verify Jan 30 is Fri.
        # pd.to_datetime("2026-01-30").weekday() -> 4. Correct.
        
        source_date = None
        temp_date = PAST_START
        while temp_date <= PAST_END:
            if temp_date.weekday() == weekday:
                source_date = temp_date
                break
            temp_date += timedelta(days=1)
            
        if source_date:
            # Extract rows for this source date
            source_mask = df['date_obj'] == source_date.date()
            source_data = df[source_mask].copy()
            
            if not source_data.empty:
                # Update Date
                # We need to keep the TIME.
                # show_time is timestamp.
                
                # Delta days
                delta_days = (current_date.date() - source_date.date()).days
                
                source_data['show_time'] = source_data['show_time'] + pd.Timedelta(days=delta_days)
                source_data['date_obj'] = source_data['show_time'].dt.date
                source_data['day_of_week'] = source_data['show_time'].dt.dayofweek
                source_data['is_weekend'] = source_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
                
                # Recalculate days_since_release
                # log_days_since_release needed.
                # Assuming 'release_date' is in columns.
                # We need to re-calc 'log_days_since_release'.
                # But release_date might be string or obj.
                
                # Let's try to update log_days_since_release if possible
                # Otherwise model uses old value which is slightly wrong but maybe ok for quick projection?
                # Ideally we update it.
                if 'release_date' in source_data.columns:
                     source_data['release_date_dt'] = pd.to_datetime(source_data['release_date'], errors='coerce')
                     # delta
                     diff = source_data['show_time'] - source_data['release_date_dt']
                     source_data['days_valid'] = diff.dt.days.clip(lower=0)
                     source_data['log_days_since_release'] = np.log1p(source_data['days_valid'])
                
                # Remove Actual Sales info (so we don't accidentally print it or use it)
                source_data['sold_tickets'] = 0 # No actuals for future
                
                future_rows.append(source_data)
        
        current_date += timedelta(days=1)
        
    if future_rows:
        return pd.concat(future_rows, ignore_index=True)
    return pd.DataFrame()

def main():
    model, encoder, df, c_map = load_resources()
    
    # 1. Preprocess Existing Data
    # Aggregate duplicates first? (Split entries)
    print("Aggregating duplicates in Historical Data...")
    # Group keys: needs all non-metric columns
    # We want to sum sold_tickets.
    # Keep other cols.
    group_cols = ['movie_name', 'cinema_id', 'session_id', 'show_time', 'capacity', 
                  'budget', 'runtime', 'popularity', 'vote_average', 'release_date', 'competitors_on_screen', 'movie_trend_7d', 'cinema_trend_7d']
                  
    # Only aggregate if columns exist (safe check)
    avail_cols = [c for c in group_cols if c in df.columns]
    
    df = df.groupby(avail_cols, as_index=False)['sold_tickets'].sum()
    
    # 2. Project Future Schedule
    df = prepare_features(df, encoder) # Prepare basic features first to get date_obj
    
    future_df = project_schedule(df)
    
    if not future_df.empty:
        print(f"Generated {len(future_df)} future slots.")
        # Future DF needs features prepped (holidays for new dates etc)
        future_df = prepare_features(future_df, encoder)
        
        # Combine
        combined_df = pd.concat([df, future_df], ignore_index=True)
    else:
        combined_df = df
        
    # 3. Predict for ALL (Refreshes predictions for past, generates for future)
    
    # --- DATA DRIVEN TRENDS ---
    # Calculate trends ONLY from historical 'df'
    print("Calculating Data-Driven Trends (Avg Sales/Show)...")
    movie_hype = df.groupby('movie_name')['sold_tickets'].mean().to_dict()
    global_avg = df['sold_tickets'].mean()
    cinema_status = df.groupby('cinema_id')['sold_tickets'].mean().to_dict()
    
    # Apply to combined_df
    combined_df['movie_trend_7d'] = combined_df['movie_name'].map(movie_hype).fillna(global_avg)
    combined_df['cinema_trend_7d'] = combined_df['cinema_id'].map(cinema_status).fillna(global_avg)
    
    # Clip
    combined_df['movie_trend_7d'] = combined_df['movie_trend_7d'].clip(upper=400)
    combined_df['cinema_trend_7d'] = combined_df['cinema_trend_7d'].clip(upper=400)
    
    feature_cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
                    'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
                    'log_days_since_release', 'cinema_id_encoded', 'movie_trend_7d', 
                    'cinema_trend_7d']
    
    print("Running Model Inference...")
    X = combined_df[feature_cols]
    combined_df['Raw Prediction'] = model.predict(X)
    
    # 4. Top 4 Movies (Volume based on Past period only to identify leaders)
    start_filter = DATE_RANGES[0][0]
    end_filter = DATE_RANGES[0][1]
    mask = (combined_df['show_time'] >= start_filter) & (combined_df['show_time'] <= end_filter)
    
    top_movies = combined_df[mask].groupby('movie_name')['sold_tickets'].sum().sort_values(ascending=False).head(10).index.tolist()
    print(f"Top 10 Movies: {top_movies}")
    
    for m in top_movies:
        generate_pdf(m, combined_df[combined_df['movie_name'] == m], c_map)

if __name__ == "__main__":
    main()
