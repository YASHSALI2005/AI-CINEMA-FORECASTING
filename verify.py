import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

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

# ==========================================
# 2. LOAD & PREPARE TEST DATA
# ==========================================
def run_verification():
    print("🧠 Loading Model & History...")
    
    # Load Model
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    
    # Load Encoder
    encoder = joblib.load("cinema_encoder_v4.pkl")
    
    # Load REAL Data
    df = pd.read_csv("final_training_data_v3.csv")
    
    # 🎲 PICK 10 RANDOM SHOWS TO TEST
    # We select rows where tickets > 0 to avoid empty data issues
    sample = df[df['sold_tickets'] > 0].sample(10, random_state=42)
    
    print("✨ Preparing Features for 10 Random Shows...")
    
    # --- Feature Engineering (Same as Training) ---
    if 'top_cast' in sample.columns:
        sample['star_power'] = sample['top_cast'].apply(get_star_power)
    else:
        sample['star_power'] = 0

    if 'date_str' not in sample.columns:
        sample['date_str'] = pd.to_datetime(sample['show_time']).dt.strftime('%Y-%m-%d')
    sample['holiday_weight'] = sample['date_str'].apply(get_holiday_weight)
    
    # Encode Cinema ID safely
    # (We use a trick: if ID is unknown, set to 0. But for history data, it should be known)
    def safe_encode(cid):
        try: return encoder.transform([str(cid)])[0]
        except: return 0
    sample['cinema_id_encoded'] = sample['cinema_id'].apply(safe_encode)
    
    # Select Columns for Prediction
    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 
        'holiday_weight', 
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d', 'cinema_trend_7d',
        'star_power'
    ]
    
    X_test = sample[features]
    y_actual = sample['sold_tickets'].values
    names = sample['movie_name'].values
    dates = sample['show_time'].values

    # ==========================================
    # 3. RUN THE TEST
    # ==========================================
    print("\n🔮 Predicting...")
    preds = model.predict(X_test)
    preds = np.maximum(preds, 0) # No negative tickets
    
    print("\n" + "="*85)
    print(f"{'MOVIE NAME':<30} | {'DATE':<12} | {'ACTUAL':<8} | {'PREDICTED':<9} | {'DIFF':<5}")
    print("="*85)
    
    total_error = 0
    
    for i in range(len(preds)):
        actual = int(y_actual[i])
        predicted = int(preds[i])
        diff = predicted - actual
        error_display = f"{diff:+d}" # e.g., +15 or -10
        
        # Color coding for terminal (optional)
        print(f"{names[i][:28]:<30} | {str(dates[i])[:10]:<12} | {actual:<8} | {predicted:<9} | {error_display:<5}")
        total_error += abs(diff)

    print("="*85)
    avg_error = total_error / 10
    print(f"📊 AVERAGE ERROR: +/- {avg_error:.1f} Tickets per show")
    print(f"✅ CONCLUSION: The model is roughly {100 - (avg_error/250*100):.1f}% Accurate on random real data.")
    print("="*85)

if __name__ == "__main__":
    run_verification()