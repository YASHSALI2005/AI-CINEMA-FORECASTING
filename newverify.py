import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from datetime import datetime

# ==========================================
# 🟢 USER CONFIGURATION (EDIT THIS PART ONLY)
# ==========================================
# 1. Identity
MOVIE_NAME = "PUSHPA (THE RULE PART -02) (2D) (HINDI)"   # Must match CSV exactly
CINEMA_ID = 890                # Your Cinema ID (e.g., 1049, 412, etc.)
SHOW_DATE = "2024-12-6"        # YYYY-MM-DD
SHOW_HOUR = 21                  # 24-hour format (e.g., 21 = 9 PM)

# 2. Scenario Parameters (The 3 Sliders)
SCENARIO_COMPETITORS = 2        # How many other movies playing?
SCENARIO_MOVIE_HYPE = 300       # 0 to 500 (380 = Super Hit)
SCENARIO_CINEMA_STATUS = 150    # 0 to 500 (100 = Normal, 150 = Busy)

# ==========================================
# 🛑 DO NOT TOUCH CODE BELOW THIS LINE
# ==========================================

def get_star_power(cast_string):
    if pd.isna(cast_string): return 0
    mega_stars = ['Shah Rukh Khan', 'Salman Khan', 'Aamir Khan', 'Prabhas',
                  'Rajinikanth', 'Vijay', 'Allu Arjun', 'Ranbir Kapoor',
                  'Hrithik Roshan', 'Yash', 'Kamal Haasan', 'Mohanlal',
                  'Mammootty', 'Deepika Padukone', 'Alia Bhatt']
    score = 0
    for actor in [x.strip() for x in str(cast_string).split('|')]:
        if actor in mega_stars: score += 100
    return score

def get_holiday_weight(date_str):
    mega_holidays = ['2024-01-01', '2024-01-26', '2023-08-15', '2024-08-15', 
                     '2023-10-02', '2024-10-02', '2023-12-25', '2024-12-25']
    major_festivals = ['2023-11-12', '2024-03-25', '2024-04-11', '2024-06-17', 
                       '2023-10-24', '2024-10-12']
    if date_str in mega_holidays: return 10.0
    if date_str in major_festivals: return 5.0
    return 1.0

def run_verification():
    print(f"\n🧠 Loading Resources to test '{MOVIE_NAME}'...")
    
    # 1. Load AI Brains
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    encoder = joblib.load("cinema_encoder_v4.pkl")
    movies_df = pd.read_csv("movie_features_safe.csv")
    history_df = pd.read_csv("final_training_data_v3.csv")

    # 2. Get Movie Metadata
    try:
        movie_data = movies_df[movies_df['original_name'] == MOVIE_NAME].iloc[0]
    except IndexError:
        print(f"❌ Error: Movie '{MOVIE_NAME}' not found in features file.")
        return

    # 3. Find ACTUAL SALES from History (Smart Search)
    # We look for a match on Name + Date + Hour (+/- 1 hour tolerance)
    history_df['show_time'] = pd.to_datetime(history_df['show_time'])
    target_date = pd.to_datetime(f"{SHOW_DATE} {SHOW_HOUR}:00:00")
    
    # Filter by Movie and Cinema first
    matches = history_df[
        (history_df['movie_name'] == MOVIE_NAME) & 
        (history_df['cinema_id'] == CINEMA_ID)
    ].copy()
    
    # Find closest show time on that date
    matches['time_diff'] = (matches['show_time'] - target_date).abs()
    closest_match = matches.nsmallest(1, 'time_diff')

    if not closest_match.empty and closest_match.iloc[0]['time_diff'].total_seconds() < 7200: # Within 2 hours
        actual_sales = int(closest_match.iloc[0]['sold_tickets'])
        real_show_time = closest_match.iloc[0]['show_time']
        print(f"✅ Found REAL Data: {real_show_time} (Sales: {actual_sales})")
    else:
        actual_sales = "N/A (Not found in History)"
        print("⚠️ Warning: Could not find this specific show in history file. Comparing against 'N/A'.")

    # 4. Feature Engineering (The AI Math)
    date_obj = datetime.strptime(SHOW_DATE, "%Y-%m-%d")
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    release_ts = pd.to_datetime(movie_data['release_date'], errors='coerce')
    selected_ts = pd.to_datetime(SHOW_DATE)
    
    if pd.isna(release_ts):
        days_since = 0
    else:
        delta = selected_ts.normalize() - release_ts.normalize()
        days_since = max(delta.days, 0)
        
    log_days_since = np.log1p(days_since)
    star_power = get_star_power(movie_data.get('top_cast', ''))
    holiday_weight = get_holiday_weight(SHOW_DATE)

    try:
        cinema_encoded = encoder.transform([str(CINEMA_ID)])[0]
    except:
        print(f"⚠️ Warning: Cinema ID {CINEMA_ID} not known to AI. Using default average.")
        cinema_encoded = 0 

    # 5. Build Input
    input_data = pd.DataFrame({
        'budget': [movie_data['budget']],
        'runtime': [movie_data['runtime']],
        'popularity': [movie_data['popularity']],
        'vote_average': [movie_data['vote_average']],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'hour': [SHOW_HOUR],
        'holiday_weight': [holiday_weight], 
        'competitors_on_screen': [SCENARIO_COMPETITORS],
        'log_days_since_release': [log_days_since],
        'cinema_id_encoded': [cinema_encoded],
        'movie_trend_7d': [SCENARIO_MOVIE_HYPE],    
        'cinema_trend_7d': [SCENARIO_CINEMA_STATUS],
        'star_power': [star_power] 
    })

    # 6. Predict
    prediction = max(int(model.predict(input_data)[0]), 0)

    # 7. Final Report
    print("\n" + "="*50)
    print(f"🎬 REPORT: {MOVIE_NAME}")
    print("="*50)
    print(f"📅 Show Date:      {SHOW_DATE} @ {SHOW_HOUR}:00")
    print(f"🧪 Scenario:       Hype={SCENARIO_MOVIE_HYPE}, Comp={SCENARIO_COMPETITORS}, Status={SCENARIO_CINEMA_STATUS}")
    print(f"⭐ Star Power:     {star_power}")
    print("-" * 50)
    print(f"🔮 AI PREDICTION:  {prediction} Tickets")
    print(f"📉 REALITY:        {actual_sales} Tickets")
    
    if isinstance(actual_sales, int):
        diff = prediction - actual_sales
        print(f"📊 Accuracy Gap:   {diff:+d} ({(1 - abs(diff)/actual_sales)*100:.1f}% Accurate)")
    print("="*50)

if __name__ == "__main__":
    run_verification()