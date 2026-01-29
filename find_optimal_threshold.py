import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

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

def run_analysis():
    print("Loading resources...")
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    encoder = joblib.load("cinema_encoder_v4.pkl")
    
    # Load Data (using a larger sample for better statistical significance)
    print("Loading data...")
    df = pd.read_csv("final_training_data_v3.csv")
    
    # Filter for valid tickets/sales
    df = df[df['sold_tickets'] > 0]
    
    # Sample 5000 rows (or all if less)
    sample_size = min(5000, len(df))
    df = df.sample(sample_size, random_state=42)
    print(f"Analyzing {len(df)} samples...")

    # Feature Engineering
    if 'top_cast' in df.columns:
        df['star_power'] = df['top_cast'].apply(get_star_power)
    else:
        df['star_power'] = 0

    if 'date_str' not in df.columns:
        df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
    df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)
    
    def safe_encode(cid):
        try: return encoder.transform([str(cid)])[0]
        except: return 0
    df['cinema_id_encoded'] = df['cinema_id'].apply(safe_encode)
    
    features = [
        'budget', 'runtime', 'popularity', 'vote_average',
        'day_of_week', 'is_weekend', 'hour', 
        'holiday_weight', 
        'competitors_on_screen', 'log_days_since_release',
        'cinema_id_encoded',
        'movie_trend_7d', 'cinema_trend_7d',
        'star_power'
    ]
    
    X = df[features]
    y_actual = df['sold_tickets'].values
    
    print("Predicting...")
    preds = model.predict(X)
    preds = np.maximum(preds, 0)
    
    # Analyze thresholds
    thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 50]
    results = []
    
    for t in thresholds:
        # Check if actual is within pred +/- t%
        # i.e. |pred - actual| / actual <= t/100  <-- This is error based on actual
        # OR |pred - actual| / pred <= t/100 <-- This is error based on prediction
        # Usually user cares about error relative to actual.
        
        # Calculate Error %
        # Avoid division by zero (though we filtered > 0 tickets)
        errors = np.abs(preds - y_actual)
        pct_errors = (errors / y_actual) * 100
        
        match_count = np.sum(pct_errors <= t)
        accuracy_rate = (match_count / len(df)) * 100
        
        results.append({
            "Threshold (+/- %)": f"{t}%",
            "Matches": match_count,
            "Accuracy Rate": f"{accuracy_rate:.1f}%"
        })
        
    print("\n" + "="*50)
    print(" ANALYSIS RESULTS: ACCURACY AT DIFFERENT THRESHOLDS")
    print("="*50)
    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))
    print("="*50)
    
    with open("threshold_results.txt", "w") as f:
        f.write(res_df.to_string(index=False))
        f.write("\n\nOBSERVATION:\n")
        f.write("Accuracy Rate = Percentage of predictions that fall within the +/- Threshold of the Actual value.\n")
    print("If you want > 80% confidence, pick the lowest threshold that gives > 80% Accuracy Rate.")

if __name__ == "__main__":
    run_analysis()
