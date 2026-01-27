import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================
def get_holiday_weight(date_str):
    mega_holidays = ['2024-01-01', '2024-01-26', '2023-08-15', '2024-08-15', 
                     '2023-10-02', '2024-10-02', '2023-12-25', '2024-12-25']
    major_festivals = ['2023-11-12', '2024-03-25', '2024-04-11', '2024-06-17', 
                       '2023-10-24', '2024-10-12']
    if date_str in mega_holidays: return 10.0
    if date_str in major_festivals: return 5.0
    return 1.0

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

def train_final():
    print("🧠 Loading V3 Dataset...")
    try:
        df = pd.read_csv("final_training_data_v3.csv")
    except FileNotFoundError:
        print("❌ Error: final_training_data_v3.csv not found.")
        return

    # ==========================================
    # 2. FEATURE ENGINEERING (The V4 Upgrade)
    # ==========================================
    print("✨ Applying V4 Features (Star Power & Holidays)...")
    
    # A. Star Power
    if 'top_cast' in df.columns:
        df['star_power'] = df['top_cast'].apply(get_star_power)
    else:
        df['star_power'] = 0
    
    # B. Holiday Weight
    if 'date_str' not in df.columns:
        df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
    df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)
    
    # C. Cinema Encoding
    print("🏷️ Encoding Cinema IDs...")
    le = LabelEncoder()
    df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))
    
    # Save the encoder so the App can use it
    joblib.dump(le, "cinema_encoder_v4.pkl")

    # D. Feature Selection
    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 
        'holiday_weight', 
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d', 'cinema_trend_7d',
        'star_power'
    ]
    
    target = 'sold_tickets'
    X = df[features]
    y = df[target]
    
    # ==========================================
    # 3. TRAIN WITH TUNED PARAMETERS
    # ==========================================
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 🔥 THESE ARE THE WINNING NUMBERS FROM YOUR TUNING 🔥
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,        # Tuned Value
        learning_rate=0.03,      # Tuned Value
        max_depth=7,             # Tuned Value
        subsample=0.8,           # Tuned Value
        colsample_bytree=0.8,    # Tuned Value
        n_jobs=-1,
        random_state=42
    )
    
    print(f"🔥 Training Final Optimized Model on {len(X_train)} samples...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # ==========================================
    # 4. EVALUATE & SAVE
    # ==========================================
    print("🔍 Evaluating...")
    predictions = np.maximum(model.predict(X_test), 0)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*50)
    print(f"🏆 FINAL OPTIMIZED MODEL RESULTS")
    print("="*50)
    print(f"✅ Accuracy Score (R²): {r2:.4f}")
    print(f"🎫 Avg Error (MAE):     +/- {mae:.2f} Tickets")
    print(f"📉 Error (RMSE):        {rmse:.2f}")
    print("="*50)
    
    # Save as V4 so your App picks it up
    model.save_model("xgb_cinema_model_v4.json")
    print("\n💾 Model Saved: xgb_cinema_model_v4.json")

if __name__ == "__main__":
    train_final()