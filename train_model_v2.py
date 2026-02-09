import pandas as pd
import xgboost as xgb
import numpy as np
import joblib
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from holiday_utils import get_holiday_weight

def train_model_v2():
    print("🚀 Loading Datasets...")
    # Load the base training data (history + basic metadata)
    df = pd.read_csv("final_training_data_v4.csv")
    
    # Load the new BH features
    bh_features = pd.read_csv("movie_features_with_bh.csv")
    
    print(f"   Base data shape: {df.shape}")
    print(f"   BH features shape: {bh_features.shape}")
    
    # 1. Merge BH Features
    print("🔄 Merging Bollywood Hungama features...")
    # Select only necessary columns from BH to avoid duplication
    bh_subset = bh_features[['original_name', 'bh_opening_day', 'bh_opening_weekend', 'bh_lifetime', 'bh_verdict']]
    
    # Merge on 'original_name'
    df = pd.merge(df, bh_subset, on='original_name', how='left')
    
    # 2. Handle Missing Values (The "No Match" Scenario)
    fill_values = {
        'bh_opening_day': 0.0,
        'bh_opening_weekend': 0.0,
        'bh_lifetime': 0.0,
        'bh_verdict': 'Unknown'
    }
    df.fillna(value=fill_values, inplace=True)
    
    # 3. Encode 'Verdict' (Categorical -> Numerical)
    verdict_map = {
        'All Time Blockbuster': 5,
        'Blockbuster': 4,
        'Super Hit': 3,
        'Hit': 2,
        'Semi Hit': 1,
        'Average': 0,
        'Below Average': -1,
        'Flop': -2,
        'Disaster': -3,
        'Unknown': 0
    }
    df['bh_verdict_score'] = df['bh_verdict'].map(verdict_map).fillna(0)
    
    # 4. Feature Engineering
    print("✨ Engineering Features (Holidays, Encoding)...")
    
    # Holiday Weight
    if 'date_str' not in df.columns:
        df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
    df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)
    
    # Cinema Encoding
    le = LabelEncoder()
    df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))
    # Save the encoder for inference (v2/v5)
    joblib.dump(le, "cinema_encoder_v5.pkl")
    
    # 5. Define Features & Target
    features = [
        # Existing Features (matching train_v4.py logic)
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 
        'holiday_weight', 
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d', 'cinema_trend_7d',
        
        # NEW BH Features
        'bh_opening_day', 'bh_verdict_score'
    ]
    
    target = 'sold_tickets'
    
    print(f"Features used: {features}")
    
    X = df[features]
    y = df[target]
    
    # 6. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 7. Train XGBoost
    print(f"🔥 Training XGBoost Model on {len(X_train)} samples...")
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=1000, 
        learning_rate=0.03, # Kept slightly conservative
        max_depth=10,       # Matching v4 depth
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=100)
    
    # 8. Evaluate
    predictions = np.maximum(model.predict(X_test), 0)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print("\n" + "="*50)
    print(f"🏆 MODEL V2 (with BH) RESULTS")
    print("="*50)
    print(f"   ✅ R² Score: {r2:.4f}")
    print(f"   🎫 MAE:      {mae:.2f}")
    print(f"   📉 RMSE:     {rmse:.2f}")
    print("="*50)
    
    # 9. Save the upgraded model
    # Saving as JSON for app compatibility (based on previous patterns)
    model.save_model("xgb_cinema_model_v5.json")
    
    # Saving as PKL as requested by user
    with open('xgboost_cinepolis_v2.pkl', 'wb') as f:
        pickle.dump(model, f)
        
    print("\n💾 Models saved: xgb_cinema_model_v5.json & xgboost_cinepolis_v2.pkl")
    print("💾 Encoder saved: cinema_encoder_v5.pkl")

if __name__ == "__main__":
    train_model_v2()
