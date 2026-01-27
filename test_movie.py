import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder

def validate_real_world():
    print("🕵️ Loading Data for Blind Test...")
    
    # 1. Load Data & Resources
    try:
        # Load the full dataset
        df = pd.read_csv("final_training_data_v3.csv")
        model = xgb.XGBRegressor()
        model.load_model("xgb_cinema_model_final.json")
        encoder = joblib.load("cinema_encoder.pkl")
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # 2. Re-Encode Cinema IDs (Critical Step)
    # The CSV has raw IDs (e.g., 'CIN01'), model needs numbers (e.g., 5)
    # We rely on the saved encoder to map them correctly.
    try:
        df['cinema_id_encoded'] = encoder.transform(df['cinema_id'].astype(str))
    except:
        # Fallback if new cinemas appear (rare in training data)
        print("⚠️ Warning: Some cinemas in file not in encoder. Handling...")
        df = df[df['cinema_id'].astype(str).isin(encoder.classes_)]
        df['cinema_id_encoded'] = encoder.transform(df['cinema_id'].astype(str))

    # 3. Pick 10 Random "Real World" Examples
    # We sample random rows to get a variety of movies and cinemas
    sample = df.sample(10, random_state=42)
    
    # 4. Define Features (MUST match training exactly)
    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 'is_holiday',
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded', 'movie_trend_7d', 'cinema_trend_7d'
    ]
    
    # 5. Ask Model to Predict
    X_sample = sample[features]
    sample['Predicted_Sales'] = model.predict(X_sample)
    
    # Clean up negatives and round
    sample['Predicted_Sales'] = sample['Predicted_Sales'].apply(lambda x: max(int(x), 0))
    sample['Actual_Sales'] = sample['sold_tickets'].astype(int)
    
    # Calculate Error
    sample['Error'] = sample['Actual_Sales'] - sample['Predicted_Sales']
    sample['Accuracy'] = 100 - (abs(sample['Error']) / (sample['Actual_Sales'] + 1) * 100)
    
    # 6. Display Results
    print("\n" + "="*80)
    print("🎬 REAL WORLD vs. AI PREDICTION BATTLE")
    print("="*80)
    print(f"{'MOVIE NAME':<30} | {'CINEMA':<15} | {'REAL':<5} | {'AI':<5} | {'DIFF':<5} | {'ACCURACY'}")
    print("-" * 85)
    
    for index, row in sample.iterrows():
        # Shorten movie name for display
        movie_name = (row['movie_name'][:28] + '..') if len(row['movie_name']) > 28 else row['movie_name']
        cinema = str(row['cinema_id'])
        real = row['Actual_Sales']
        ai = row['Predicted_Sales']
        diff = row['Error'] # Positive means AI under-predicted, Negative means Over-predicted
        acc = max(row['Accuracy'], 0) # Cap at 0%
        
        # Color coding for terminal (optional, keeping it simple text)
        print(f"{movie_name:<30} | {cinema:<15} | {real:<5} | {ai:<5} | {diff:<5} | {acc:.1f}%")

    print("="*80)
    print("Legend:")
    print("   REAL: How many tickets were actually sold.")
    print("   AI:   How many tickets your XGBoost model predicted.")
    print("   DIFF: (Real - AI). Small numbers are good!")

if __name__ == "__main__":
    validate_real_world()