import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from datetime import datetime

# ==========================================
# 1. HELPER FUNCTIONS
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

def load_resources():
    print("🧠 Loading V4 Resources...")
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    encoder = joblib.load("cinema_encoder_v4.pkl")
    movies_df = pd.read_csv("movie_features_safe.csv")
    return model, encoder, movies_df

# ==========================================
# 2. THE "ANIMAL" STRESS TEST
# ==========================================
def run_test():
    model, encoder, movies_df = load_resources()
    
    # --- SCENARIO: ANIMAL (Ranbir Kapoor) ---
    movie_name = "ANIMAL (HINDI)"
    cinema_id = 412                
    selected_date = "2023-12-02"   # First Saturday
    hour = 21                      # 9:00 PM (Night Show)
    
    # Real World Context for that day
    competitors = 2                # Sam Bahadur was running too
    movie_trend = 380.0            # Extremely High Hype
    cinema_trend = 110.0           
    actual_sales = 295             # It was packed
    
    # -------------------------------------------
    
    try:
        movie_data = movies_df[movies_df['original_name'] == movie_name].iloc[0]
    except IndexError:
        print(f"❌ Error: Could not find '{movie_name}' in movie_features_safe.csv")
        return

    # Feature Engineering
    date_obj = datetime.strptime(selected_date, "%Y-%m-%d")
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    
    release_date = pd.to_datetime(movie_data['release_date'])
    days_since = (pd.to_datetime(selected_date) - release_date).days
    days_since = max(days_since, 0)
    log_days_since = np.log1p(days_since)
    
    star_power = get_star_power(movie_data.get('top_cast', ''))
    holiday_weight = get_holiday_weight(selected_date)

    try:
        cinema_encoded = encoder.transform([str(cinema_id)])[0]
    except:
        cinema_encoded = 0 
        
    input_data = pd.DataFrame({
        'budget': [movie_data['budget']],
        'runtime': [movie_data['runtime']],
        'popularity': [movie_data['popularity']],
        'vote_average': [movie_data['vote_average']],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'hour': [hour],
        'holiday_weight': [holiday_weight], 
        'competitors_on_screen': [competitors],
        'log_days_since_release': [log_days_since],
        'cinema_id_encoded': [cinema_encoded],
        'movie_trend_7d': [movie_trend],    
        'cinema_trend_7d': [cinema_trend],
        'star_power': [star_power] 
    })
    
    print("\n--- 🤖 Model Input (Processed) ---")
    print(input_data.T)
    
    prediction = max(int(model.predict(input_data)[0]), 0)
    
    print(f"\n=====================================")
    print(f"🎬 Movie: {movie_name}")
    print(f"📅 Date:  {selected_date} @ {hour}:00")
    print(f"⭐ Stars: {star_power} (Ranbir Kapoor)")
    print(f"-------------------------------------")
    print(f"📉 Actual Capacity: {actual_sales}")
    print(f"🔮 AI Prediction:   {prediction}")
    print(f"=====================================")

if __name__ == "__main__":
    run_test()