import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from urllib.parse import quote_plus

# ==========================================
# CONFIGURATION
# ==========================================
DB_PASSWORD  = "Yashsali@*2005"  # <--- UPDATE THIS
DB_USER      = "root"
DB_HOST      = "localhost"
DB_NAME      = "movie"
CSV_FILE     = "movie_features_safe.csv" 

# ==========================================
# HOLIDAY LIST (India 2023-2026)
# ==========================================
HOLIDAYS = [
    '2023-01-26', '2023-03-08', '2023-08-15', '2023-10-02', '2023-10-24', '2023-11-12', '2023-12-25',
    '2024-01-26', '2024-03-25', '2024-08-15', '2024-10-02', '2024-11-01', '2024-12-25',
    '2025-01-26', '2025-03-14', '2025-08-15', '2025-10-02', '2025-10-20', '2025-12-25'
]

def apply_final_rules():
    print("🔧 Starting FINAL Feature Engineering...")
    
    # 1. Load SQL
    encoded_password = quote_plus(DB_PASSWORD)
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
    
    print("   📥 Loading Transactions...")
    query = """
    SELECT movie_name, cinema_id, show_time, 
           total_count as sold_tickets, total_sum as revenue
    FROM m_grouped_transactions
    WHERE total_sum > 0
    """
    df = pd.read_sql(query, engine)
    
    # 2. Load TMDB
    print("   📥 Loading Movie Features...")
    tmdb_features = pd.read_csv(CSV_FILE).drop_duplicates(subset=['original_name'])
    
    # 3. Merge
    df = pd.merge(df, tmdb_features, left_on='movie_name', right_on='original_name', how='inner')
    
    # ---------------------------------------------------------
    # RULE 7: CALENDAR & HOLIDAYS (New!)
    # ---------------------------------------------------------
    print("   📅 Applying Holidays & Calendar...")
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['date_str'] = df['show_time'].dt.strftime('%Y-%m-%d')
    
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour'] = df['show_time'].dt.hour
    
    # Holiday Flag
    df['is_holiday'] = df['date_str'].isin(HOLIDAYS).astype(int)

    # ---------------------------------------------------------
    # RULE 1: DUAL LAGS (The Competitor's Secret)
    # ---------------------------------------------------------
    print("   🔄 Calculating Cinema Trends (This is the key)...")
    df = df.sort_values(['cinema_id', 'show_time'])
    
    # Feature A: How is THIS MOVIE performing? (We had this)
    df['movie_trend_7d'] = df.groupby('movie_name')['sold_tickets'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Feature B: How is THIS CINEMA performing? (NEW!)
    # If Cinema X has been empty all week, it will likely stay empty.
    df['cinema_trend_7d'] = df.groupby('cinema_id')['sold_tickets'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )

    # ---------------------------------------------------------
    # OTHER RULES (Bass Decay, Competition)
    # ---------------------------------------------------------
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['days_since_release'] = (df['show_time'] - df['release_date']).dt.days
    df['days_since_release'] = df['days_since_release'].fillna(0).apply(lambda x: max(x, 0))
    df['log_days_since_release'] = np.log1p(df['days_since_release'])

    df['hour_slot'] = df['show_time'].dt.floor('h') 
    competition = df.groupby('hour_slot')['movie_name'].nunique().reset_index()
    competition.columns = ['hour_slot', 'competitors_on_screen']
    df = pd.merge(df, competition, on='hour_slot', how='left')
    
    # Fill NaNs
    df = df.fillna(0)
    
    # Save
    df.to_csv("final_training_data_v3.csv", index=False)
    print(f"✅ FINAL DATASET READY: final_training_data_v3.csv")

if __name__ == "__main__":
    apply_final_rules()