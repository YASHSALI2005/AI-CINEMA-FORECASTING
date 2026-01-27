import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder

def train_final():
    print("🧠 Loading FINAL V3 Dataset...")
    try:
        # Load the file you just created
        df = pd.read_csv("final_training_data_v3.csv")
        print(f"   📊 Loaded {len(df)} rows.")
    except FileNotFoundError:
        print("❌ Error: 'final_training_data_v3.csv' not found.")
        return

    # ---------------------------------------------------------
    # 1. PREPARE FEATURES
    # ---------------------------------------------------------
    print("   🏷️ Encoding Cinema IDs...")
    le = LabelEncoder()
    df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))
    
    # Save the encoder (Critical for the App later)
    joblib.dump(le, "cinema_encoder.pkl")

    # The "Winning" Feature List
    features = [
        # Content (The Movie)
        'budget', 'runtime', 'popularity', 'vote_average',
        
        # Context (The Environment)
        'day_of_week', 'is_weekend', 'hour', 'is_holiday',
        'competitors_on_screen', 'log_days_since_release',
        
        # Location (The Cinema)
        'cinema_id_encoded',
        
        # Trends (The Secret Sauce)
        'movie_trend_7d',   # Is this movie hot right now?
        'cinema_trend_7d'   # Is this cinema busy this week?
    ]
    
    target = 'sold_tickets'
    
    X = df[features]
    y = df[target]
    
    # ---------------------------------------------------------
    # 2. TRAIN MODEL
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Optimized for 1.8 Million Rows
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=700,        # More trees = smarter model
        learning_rate=0.04,      # Slower learning = higher precision
        max_depth=9,             # Deep trees for complex patterns
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,               # Use all CPU cores
        random_state=42
    )
    
    print(f"   🔥 Training on {len(X_train)} samples... (This may take 5 mins)")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # ---------------------------------------------------------
    # 3. EVALUATE
    # ---------------------------------------------------------
    print("   🔍 Evaluating...")
    predictions = np.maximum(model.predict(X_test), 0)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*50)
    print(f"🏆 ULTIMATE RESULTS (V3 Model)")
    print("="*50)
    print(f"   ✅ Accuracy Score (R²): {r2:.4f}")
    print(f"   📉 Error (MAE):         +/- {mae:.2f} Tickets")
    print(f"   📉 Error (RMSE):        {rmse:.2f}")
    print("="*50)
    
    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    print("\n📊 Feature Importance:")
    print(importance.to_string(index=False))
    
    # Save Final Model
    model.save_model("xgb_cinema_model_final.json")
    print("\n💾 Model Saved: xgb_cinema_model_final.json")
    print("👉 We are ready to build the Frontend!")

if __name__ == "__main__":
    train_final()