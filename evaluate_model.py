import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def evaluate():
    print("🧠 Loading Dataset for Evaluation...")
    try:
        df = pd.read_csv("final_training_data_v3.csv")
    except FileNotFoundError:
        print("❌ Error: 'final_training_data_v3.csv' not found.")
        return

    # 1. RECREATE FEATURE ENCODING (Must match training exactly for consistency)
    print("   🏷️ Re-encoding Cinema IDs...")
    # Ideally should load the encoder, but train_v3 fit it fresh. 
    # To be safe and identical to train_v3:
    le = LabelEncoder()
    df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))

    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 'is_holiday',
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d',
        'cinema_trend_7d'
    ]
    
    target = 'sold_tickets'
    X = df[features]
    y = df[target]
    
    # 2. SPLIT (Same random_state=42 as training)
    print("   ✂️ Splitting Test Set...")
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. LOAD MODEL
    print("   📥 Loading Model...")
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_final.json")
    
    # 4. PREDICT & SCORE
    print("   🔮 Predicting on Test Set...")
    predictions = np.maximum(model.predict(X_test), 0)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*50)
    print(f"📊 CURRENT MODEL METRICS")
    print("="*50)
    print(f"   ⭐ Accuracy (R² Score):  {r2:.4f}")
    print(f"   🎫 Avg Error (MAE):      +/- {mae:.2f} Tickets")
    print(f"   📉 Root Mean Sq Error:   {rmse:.2f}")
    print("="*50)

if __name__ == "__main__":
    evaluate()
