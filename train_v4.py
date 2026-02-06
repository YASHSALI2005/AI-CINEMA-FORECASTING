import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

from holiday_utils import get_holiday_weight

def train_v4():
    print("🧠 Loading FINAL V3 Dataset for V4 Training...")
    try:
        df = pd.read_csv("final_training_data_v3.csv")
        # Load Safe Features for Cast Info
        movies_df = pd.read_csv("movie_features_safe.csv")
    except FileNotFoundError:
        print("❌ Error: Datasets not found.")
        return

    # ---------------------------------------------------------
    # 1. FEATURE ENGINEERING (V4 Upgrades)
    # ---------------------------------------------------------
    print("   ✨ Applying V4 Features (Holidays)...")
    
    # B. Granular Holiday Weight
    # Ensure date_str column exists, else extract from show_time
    if 'date_str' not in df.columns:
        df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
        
    df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)
    
    # C. Standard Encoding
    print("   🏷️ Encoding Cinema IDs...")
    le = LabelEncoder()
    df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))
    joblib.dump(le, "cinema_encoder_v4.pkl")

    # The V4 Feature List
    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 
        'holiday_weight', # Replaces is_holiday
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d', 'cinema_trend_7d'
    ]
    
    target = 'sold_tickets'
    X = df[features]
    y = df[target]
    
    # ---------------------------------------------------------
    # 2. TRAIN MODEL
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000,       # Increased from 700
        learning_rate=0.03,      # Slightly lower
        max_depth=10,            # Deeper trees
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    
    print(f"   🔥 Training V4 Model on {len(X_train)} samples...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # ---------------------------------------------------------
    # 3. EVALUATE
    # ---------------------------------------------------------
    print("   🔍 Evaluating V4...")
    predictions = np.maximum(model.predict(X_test), 0)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*50)
    print(f"🏆 V4 MODEL RESULTS")
    print("="*50)
    print(f"   ✅ Accuracy Score (R²): {r2:.4f}")
    print(f"   🎫 Avg Error (MAE):     +/- {mae:.2f} Tickets")
    print(f"   📉 Error (RMSE):        {rmse:.2f}")
    print("="*50)
    
    # Save Final Model
    model.save_model("xgb_cinema_model_v4.json")
    print("\n💾 Model Saved: xgb_cinema_model_v4.json")

if __name__ == "__main__":
    train_v4()
