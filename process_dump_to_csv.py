import pandas as pd
import numpy as np
import re
import csv
from io import StringIO
import datetime

# --- CONFIG ---
SQL_FILE = "backup_20260205.sql"
METADATA_FILE = "movie_features_safe.csv"
OUTPUT_FILE = "final_training_data_from_dump.csv"

def normalize_name(name):
    if not isinstance(name, str): return ""
    # Remove contents in brackets e.g., (Hindi), (3D)
    name = re.sub(r'\s*\(.*?\)', '', name)
    return name.strip().lower()

def parse_session_capacities(sql_file):
    print("Parsing capacities from m_session...")
    caps = {} 
    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("INSERT INTO `m_session`"):
                content = line[line.find("VALUES")+6:].strip()
                if content.endswith(';'): content = content[:-1]
                rows = content.split("),(")
                for row in rows:
                    row = row.strip("()")
                    try:
                        reader = csv.reader(StringIO(row), quotechar="'", skipinitialspace=True)
                        fields = next(reader)
                        s_id = fields[0]
                        try: cap = int(fields[-1]) 
                        except: cap = 300
                        if cap == 0: cap = 300
                        caps[s_id] = cap
                    except: pass
    return caps

def parse_transactions(sql_file):
    print("Parsing transactions from m_grouped_transactions...")
    data = []
    with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith("INSERT INTO `m_grouped_transactions`"):
                content = line[line.find("VALUES")+6:].strip()
                if content.endswith(';'): content = content[:-1]
                rows = content.split("),(")
                for row in rows:
                    row = row.strip("()")
                    try:
                        reader = csv.reader(StringIO(row), quotechar="'", skipinitialspace=True)
                        fields = next(reader)
                        if len(fields) < 11: continue
                        
                        # Schema: 0:id, 1:cinema_id, 2:session_id, 3:movie_id, 4:movie_name, 
                        # 6:show_time, 7:normal_sum, 8:normal_count, 9:exec_sum, 10:exec_count
                        
                        row_dict = {
                            'cinema_id': fields[1],
                            'session_id': fields[2],
                            'original_name': fields[4],
                            'show_time': fields[6],
                        }
                        
                        # Metrics
                        try: n_sum = float(fields[7]) 
                        except: n_sum = 0.0
                        try: n_cnt = int(fields[8])
                        except: n_cnt = 0
                        try: e_sum = float(fields[9])
                        except: e_sum = 0.0
                        try: e_cnt = int(fields[10])
                        except: e_cnt = 0
                        
                        row_dict['sold_tickets'] = n_cnt + e_cnt
                        row_dict['revenue_x'] = n_sum + e_sum
                        
                        data.append(row_dict)
                    except: pass
                    
    return pd.DataFrame(data)

def main():
    # 1. Load Data
    df_caps = parse_session_capacities(SQL_FILE)
    df_trans = parse_transactions(SQL_FILE)
    print(f"Extracted {len(df_trans)} transactions.")
    
    # 2. Preprocess Transactions
    df_trans['show_time'] = pd.to_datetime(df_trans['show_time'], errors='coerce')
    df_trans['capacity'] = df_trans['session_id'].map(lambda x: df_caps.get(str(x), 300))
    df_trans['clean_name'] = df_trans['original_name'].apply(normalize_name)
    
    # 3. Load Metadata
    print("Loading metadata...")
    try:
        meta_df = pd.read_csv(METADATA_FILE)
        # Fix: CSV has 'original_name' not 'title'
        src_col = 'original_name' if 'original_name' in meta_df.columns else 'tmdb_title'
        meta_df['clean_name'] = meta_df[src_col].apply(normalize_name) 
        # Keep relevant columns
        # Check actual cols in meta file
        # Assuming: title, budget, revenue, runtime, genres, popularity, vote_average, vote_count, release_date
        cols_to_use = ['clean_name', 'tmdb_title', 'genres', 'budget', 'runtime', 'release_date', 'popularity', 'vote_count', 'vote_average', 'director', 'top_cast']
        # Filter cols that exist
        existing_cols = [c for c in cols_to_use if c in meta_df.columns]
        meta_df = meta_df[existing_cols].drop_duplicates(subset=['clean_name'])
        
        # Merge
        df_merged = df_trans.merge(meta_df, on='clean_name', how='left')
    except Exception as e:
        print(f"Metadata error: {e}")
        df_merged = df_trans
        
    # 4. Feature Engineering (mimic final_training_data_v4)
    print("Engineering features...")
    df_merged['date_str'] = df_merged['show_time'].dt.date.astype(str)
    df_merged['day_of_week'] = df_merged['show_time'].dt.dayofweek
    df_merged['is_weekend'] = df_merged['day_of_week'].isin([4, 5, 6]).astype(int) # Fri, Sat, Sun usually
    df_merged['hour'] = df_merged['show_time'].dt.hour
    
    # Hour slot
    def get_slot(h):
        if h < 12: return 'Morning'
        elif h < 17: return 'Afternoon'
        elif h < 20: return 'Evening'
        else: return 'Night'
    df_merged['hour_slot'] = df_merged['hour'].apply(get_slot)
    
    # Days since release
    # Need release_date to be datetime
    if 'release_date' in df_merged.columns:
        df_merged['release_date'] = pd.to_datetime(df_merged['release_date'], errors='coerce')
        df_merged['days_since_release'] = (df_merged['show_time'] - df_merged['release_date']).dt.days
        df_merged['days_since_release'] = df_merged['days_since_release'].fillna(0).clip(lower=0)
        df_merged['log_days_since_release'] = np.log1p(df_merged['days_since_release'])
    else:
         df_merged['days_since_release'] = 0
         df_merged['log_days_since_release'] = 0
         
    # Competitors on screen (Simplified: Count shows in same cinema-hour)
    # Group by cinema, date, hour
    print("Calculating competitors...")
    competitor_counts = df_merged.groupby(['cinema_id', 'date_str', 'hour'])['original_name'].count().reset_index()
    competitor_counts.rename(columns={'original_name': 'competitors_on_screen'}, inplace=True)
    # Competitors = Total shows in that slot - 1 (the movie itself)
    competitor_counts['competitors_on_screen'] = competitor_counts['competitors_on_screen'] - 1
    
    df_merged = df_merged.merge(competitor_counts, on=['cinema_id', 'date_str', 'hour'], how='left')
    
    # Holiday (Placeholder)
    df_merged['is_holiday'] = 0 
    
    # Trends (Placeholder - skip for speed or use 0)
    df_merged['movie_trend_7d'] = 0
    df_merged['cinema_trend_7d'] = 0
    
    # Price
    if 'revenue_x' in df_merged.columns and 'sold_tickets' in df_merged.columns:
        df_merged['price_per_ticket'] = df_merged.apply(lambda x: x['revenue_x']/x['sold_tickets'] if x['sold_tickets'] > 0 else 0, axis=1)
    else:
        df_merged['price_per_ticket'] = 0
        
    df_merged['ticket_category'] = 'Standard' # simplified
    
    # movie_name vs original_name
    df_merged['movie_name'] = df_merged['original_name']
    
    # Final cleanup
    # Ensure columns match V4 if possible
    final_cols = ['movie_name', 'cinema_id', 'session_id', 'show_time', 'sold_tickets', 'revenue_x', 'capacity', 'original_name', 
                  'tmdb_title', 'genres', 'budget', 'runtime', 'release_date', 'popularity', 'vote_count', 
                  'vote_average', 'director', 'top_cast', 'date_str', 'day_of_week', 'is_weekend', 'hour', 
                  'is_holiday', 'movie_trend_7d', 'cinema_trend_7d', 'days_since_release', 
                  'log_days_since_release', 'hour_slot', 'competitors_on_screen', 'price_per_ticket', 
                  'ticket_category']
                  
    # Add missing cols with NaN/0
    for c in final_cols:
        if c not in df_merged.columns:
            df_merged[c] = None
            
    df_merged = df_merged[final_cols]
    
    print(f"Saving to {OUTPUT_FILE}...")
    df_merged.to_csv(OUTPUT_FILE, index=False)
    print("Done.")

if __name__ == "__main__":
    main()
