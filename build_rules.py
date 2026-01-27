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

def apply_7_rules():
    print("🔧 Starting Feature Engineering (Applying the 7 Rules)...")
    
    # 1. Load SQL Data
    encoded_password = quote_plus(DB_PASSWORD)
    engine = create_engine(f"mysql+mysqlconnector://{DB_USER}:{encoded_password}@{DB_HOST}/{DB_NAME}")
    
    print("   📥 Loading Transactions from SQL...")
    # We load transactions where sales actually happened
    query = """
    SELECT movie_name, cinema_id, show_time, 
           total_count as sold_tickets, total_sum as revenue
    FROM m_grouped_transactions
    WHERE total_sum > 0
    """
    df = pd.read_sql(query, engine)
    print(f"   ✅ Loaded {len(df)} transactions.")
    
    # 2. Load TMDB Features
    print("   📥 Loading Movie Features from CSV...")
    try:
        tmdb_features = pd.read_csv(CSV_FILE)
        # Drop duplicates just in case
        tmdb_features = tmdb_features.drop_duplicates(subset=['original_name'])
        
        # Simple cleanup for missing values
        tmdb_features['genres'] = tmdb_features['genres'].fillna('Unknown')
        tmdb_features['budget'] = tmdb_features['budget'].fillna(0)
        tmdb_features['revenue'] = tmdb_features['revenue'].fillna(0)
    except FileNotFoundError:
        print(f"   ❌ Error: {CSV_FILE} not found. Ensure it is in the same folder!")
        return

    # 3. Merge SQL + CSV
    print("   🔄 Merging SQL Transaction Data with TMDB Features...")
    # Inner join filters out movies we couldn't find in TMDB (the bad matches)
    df = pd.merge(df, tmdb_features, left_on='movie_name', right_on='original_name', how='inner')
    print(f"   ✅ Merged Data Size: {len(df)} rows")
    
    # ---------------------------------------------------------
    # RULE 7: CALENDAR LOGIC (Weekend & Day of Week)
    # ---------------------------------------------------------
    print("   📅 Applying Rule 7: Calendar Logic...")
    df['show_time'] = pd.to_datetime(df['show_time'])
    df['day_of_week'] = df['show_time'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['hour'] = df['show_time'].dt.hour
    
    # ---------------------------------------------------------
    # RULE 4: BASS DECAY (Time since release)
    # ---------------------------------------------------------
    print("   📉 Applying Rule 4: Bass Decay Model...")
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    # Calculate days difference
    df['days_since_release'] = (df['show_time'] - df['release_date']).dt.days
    # Fix negative days (pre-bookings) and errors
    df['days_since_release'] = df['days_since_release'].fillna(0).apply(lambda x: max(x, 0))
    # Log Transform: Models the sharp drop in interest better than linear time
    df['log_days_since_release'] = np.log1p(df['days_since_release'])

    # ---------------------------------------------------------
    # RULE 6: CANNIBALIZATION (Competition)
    # ---------------------------------------------------------
    print("   ⚔️ Applying Rule 6: Competition (Cannibalization)...")
    # Create a time slot identifier (e.g., "2024-01-01 18:00")
    df['hour_slot'] = df['show_time'].dt.floor('h') 
    # Count how many UNIQUE movies are playing in that same hour slot
    competition = df.groupby('hour_slot')['movie_name'].nunique().reset_index()
    competition.columns = ['hour_slot', 'competitors_on_screen']
    # Merge back to main dataframe
    df = pd.merge(df, competition, on='hour_slot', how='left')

    # ---------------------------------------------------------
    # RULE 1: LAG FEATURES (The History Rule)
    # ---------------------------------------------------------
    print("   🔄 Applying Rule 1: Lag Features (This takes a moment)...")
    df = df.sort_values(['movie_name', 'show_time'])
    
    # Calculate the average occupancy of the last 7 shows for THIS movie
    # This teaches the model: "If the last show was full, this one likely will be too"
    df['rolling_occupancy_7d'] = df.groupby('movie_name')['sold_tickets'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    # Fill remaining NaNs with 0 (for the very first show of a movie)
    df = df.fillna(0)
    
    # 4. Save Final Training Data
    output_filename = "final_training_data.csv"
    df.to_csv(output_filename, index=False)
    print(f"\n🎉 SUCCESS! Feature Engineering Complete.")
    print(f"   📂 Saved: {output_filename}")
    print(f"   📊 Total Training Rows: {len(df)}")
    print("   👉 You are ready to run 'train_model.py'!")

if __name__ == "__main__":
    apply_7_rules()