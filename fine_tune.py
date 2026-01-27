import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
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
# 2. LOAD & PREPARE DATA
# ==========================================
print("🧠 Loading Data...")
df = pd.read_csv("final_training_data_v3.csv")

# --- OPTIMIZATION: SAMPLE THE DATA ---
# 1.8 Million rows is too big for a Grid Search on a laptop.
# We take a random 10% sample (approx 180k rows) to find the best settings.
if len(df) > 200000:
    print(f"⚠️ Dataset is huge ({len(df)} rows). Sampling 150,000 rows for faster tuning...")
    df = df.sample(n=150000, random_state=42)

print("✨ Generating Features...")

# 1. Star Power
if 'top_cast' in df.columns:
    df['star_power'] = df['top_cast'].apply(get_star_power)
else:
    df['star_power'] = 0

# 2. Holiday Weight
if 'date_str' not in df.columns:
    df['date_str'] = pd.to_datetime(df['show_time']).dt.strftime('%Y-%m-%d')
df['holiday_weight'] = df['date_str'].apply(get_holiday_weight)

# 3. Cinema Encoding
le = LabelEncoder()
df['cinema_id_encoded'] = le.fit_transform(df['cinema_id'].astype(str))

# --- CRITICAL: SELECT ONLY NUMERIC FEATURES ---
# We explicitly list the columns we want. If a column is missing, we skip it.
# This PREVENTS the "Invalid columns: object" error.
numeric_features = [
    'budget', 'runtime', 'popularity', 'vote_average',
    'day_of_week', 'is_weekend', 'hour', 
    'holiday_weight', 
    'competitors_on_screen', 'log_days_since_release',
    'cinema_id_encoded',
    'movie_trend_7d', 'cinema_trend_7d',
    'star_power'
]

# Double check that these columns exist
available_features = [col for col in numeric_features if col in df.columns]
X = df[available_features]
y = df['sold_tickets']

print(f"✅ Training on {X.shape[1]} numeric features: {available_features}")

# ==========================================
# 3. GRID SEARCH
# ==========================================
param_grid = {
    'n_estimators': [500, 800],           
    'learning_rate': [0.03, 0.05],       
    'max_depth': [7, 10],                 
    'subsample': [0.8],                  
    'colsample_bytree': [0.8]            
}

print(f"🐢 Starting Grid Search on {len(X)} rows...")
model = xgb.XGBRegressor(objective='reg:squarederror', n_jobs=-1, random_state=42)

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, n_jobs=-1, verbose=2)

try:
    grid_search.fit(X, y)
    print("\n" + "="*50)
    print(f"🏆 BEST PARAMETERS FOUND")
    print("="*50)
    print(grid_search.best_params_)
    print("="*50)
except Exception as e:
    print(f"\n❌ Tuning Failed: {e}")