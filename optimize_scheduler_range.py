import pandas as pd
import xgboost as xgb
import joblib
import numpy as np

def run_range_analysis():
    print("Loading resources...")
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json")
    encoder = joblib.load("cinema_encoder_v4.pkl")
    
    print("Loading data...")
    df = pd.read_csv("final_training_data_v3.csv")
    df = df[df['sold_tickets'] > 0].sample(min(5000, len(df)), random_state=42)

    # Feature Engineering (Minimal needed for prediction)
    if 'date_str' not in df.columns:
        df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
    
    # Helper functions inline to save space/time
    def get_holiday_weight(date_str):
        mega = ['2023-08-15', '2023-10-02', '2023-12-25', '2024-01-01', '2024-01-26', '2024-08-15', '2024-10-02', '2024-12-25']
        major = ['2023-10-24', '2023-11-12', '2024-03-25', '2024-04-11', '2024-06-17', '2024-10-12']
        if date_str in mega: return 10.0
        if date_str in major: return 5.0
        return 1.0

    def get_star_power(cast_str):
        if pd.isna(cast_str): return 0
        mega = ['Shah Rukh Khan', 'Salman Khan', 'Aamir Khan', 'Prabhas', 'Rajinikanth', 'Vijay', 'Allu Arjun', 'Ranbir Kapoor', 'Hrithik Roshan', 'Yash', 'Kamal Haasan', 'Mohanlal', 'Mammootty', 'Deepika Padukone', 'Alia Bhatt']
        return sum(100 for x in str(cast_str).split('|') if x.strip() in mega)
    
    df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)
    df['star_power'] = df['top_cast'].apply(get_star_power) if 'top_cast' in df.columns else 0
    
    def safe_encode(cid):
        try: return encoder.transform([str(cid)])[0]
        except: return 0
    df['cinema_id_encoded'] = df['cinema_id'].apply(safe_encode)
    
    features = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 'log_days_since_release', 'cinema_id_encoded', 'movie_trend_7d', 'cinema_trend_7d', 'star_power']
    
    print("Predicting...")
    preds = model.predict(df[features])
    preds = np.maximum(preds, 0)
    
    actual_tickets = df['sold_tickets'].values
    
    # --- ANALYSIS LOOP ---
    # We found that 46% of errors are "Miss Low". 
    # Current Logic: Low = 0.85 * Pred. High = Low + 20.
    
    # We will test combinations to find the best pair.
    test_factors = [0.85, 0.75, 0.65, 0.55] # Lowering the start point
    test_offsets = [20, 25, 30, 35, 40]    # Widening the range
    
    results = []
    
    print("\nAnalyzing Range Coverage (Capacity=300)...")
    
    for factor in test_factors:
        for offset in test_offsets:
            inside = 0
            
            for i in range(len(preds)):
                raw_pred = preds[i]
                act_ticket = actual_tickets[i]
                
                # Dynamic Logic
                pred_low_pct = (raw_pred * factor / 300) * 100
                pred_high_pct = pred_low_pct + offset
                
                actual_pct = (act_ticket / 300) * 100
                
                if pred_low_pct <= actual_pct <= pred_high_pct:
                    inside += 1
                    
            acc = (inside / len(preds)) * 100
            
            results.append({
                "Low Factor": factor,
                "Added Value (+%)": offset,
                "Accuracy (%)": f"{acc:.1f}%"
            })

    print("\n" + "="*50)
    print(" 2D OPTIMIZATION RESULTS")
    print("="*50)
    res_df = pd.DataFrame(results)
    # Sort by Accuracy
    # res_df = res_df.sort_values(by="Accuracy (%)", ascending=False)
    print(res_df.to_string(index=False))
    print("="*50)
    
    with open("range_analysis_results.txt", "w") as f:
        f.write(res_df.to_string(index=False))

if __name__ == "__main__":
    run_range_analysis()
