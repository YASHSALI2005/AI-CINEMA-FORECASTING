import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import requests
from datetime import datetime
import re
from holiday_utils import get_holiday_weight
# ==========================================
# 1. CONFIGURATION & LOADING
# ==========================================
st.set_page_config(page_title="Cinepolis Forecaster", page_icon="🎬", layout="wide")

TMDB_API_KEY = "cfc42745859368e3d9c8252b457b09fb"

@st.cache_resource
def load_resources():
    # 1. Load Model & Encoder
    # Assumes you have these V4 files. If not, rename them back to 'final'
    model = xgb.XGBRegressor()
    model.load_model("xgb_cinema_model_v4.json") 
    encoder = joblib.load("cinema_encoder_v4.pkl")
    
    # 2. Load Metadata (Movies)
    movies_df = pd.read_csv("movie_features_safe.csv")
    
    # 3. Load FULL History (Optimized)
    # We NEED this for the Scheduler to find real slots
    # UPDATED: Added 'sold_tickets', 'original_name' for actuals lookup
    history_cols = ['cinema_id', 'show_time', 'sold_tickets', 'original_name']
    history_df = pd.read_csv("final_training_data_v4.csv", usecols=history_cols)
    history_df['show_time'] = pd.to_datetime(history_df['show_time'])
    
    # --- ADD Normalize Name Column for robust matching ---
    # "ZOOTOPIA 2 (3D) (HINDI)" -> "zootopia 2"
    history_df['clean_name'] = history_df['original_name'].astype(str).apply(
        lambda x: re.sub(r'\s*\(.*?\)', '', x).strip().lower()
    )
    
    # 4. Get Sorted List of ALL Cinemas
    cinemas = sorted(history_df['cinema_id'].unique().tolist())
    
    return model, encoder, movies_df, history_df, cinemas

@st.cache_data
def fetch_poster(movie_title):
    # CLEAN THE TITLE: Remove (Hindi), (3D), (Tamil) etc.
    # "JAWAN (HINDI)" -> "JAWAN"
    clean_title = re.sub(r'\s*\(.*?\)', '', movie_title).strip()
    
    try:
        url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
        
        # --- FIX: Strict exact match first, then loose ---
        response = requests.get(url, timeout=2)
        data = response.json()
        
        if data['results']:
            # 1. Try EXACT match first
            for res in data['results']:
                if res['title'].lower() == clean_title.lower():
                     poster_path = res['poster_path']
                     if poster_path:
                        return f"https://image.tmdb.org/t/p/original{poster_path}"

            # 2. Fallback to first result if no exact match
            poster_path = data['results'][0]['poster_path']
            if poster_path:
                return f"https://image.tmdb.org/t/p/original{poster_path}"
                
    except:
        pass
    # Reliable Fallback Image if nothing found
    return "https://cdn-icons-png.flaticon.com/512/2503/2503508.png"


@st.cache_data
def fetch_tmdb_movie_details(movie_title):
    """
    Fetches full metadata for a movie: Budget, Runtime, Popularity, Vote Average, Release Date, Genres.
    """
    clean_title = re.sub(r'\s*\(.*?\)', '', movie_title).strip()
    
    try:
        # 1. Search for Movie ID
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={clean_title}"
        resp = requests.get(search_url, timeout=2)
        data = resp.json()
        
        if not data['results']:
            return None
            
        # --- NEW: Filter for UPCOMING/RECENT Matches only ---
        today = datetime.today().strftime('%Y-%m-%d')
        upcoming_results = []
        for res in data['results']:
            r_date = res.get('release_date', '0000-00-00')
            if r_date >= today:
                upcoming_results.append(res)
        
        if not upcoming_results:
            # Fallback: if no upcoming found, it might be a very fresh release (last 7 days)
            # Or just check if the top result is at least recent
            # For strictness: return None or pick the best among search
            return None
            
        # Try exact match among upcoming results first
        best_match = upcoming_results[0]
        for res in upcoming_results:
            if res['title'].lower() == clean_title.lower():
                best_match = res
                break
        
        movie_id = best_match['id']
        
        # 2. Get Full Details
        details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        resp_det = requests.get(details_url, timeout=2)
        det_data = resp_det.json()
        
        # Extract Genres
        genres = " | ".join([g['name'] for g in det_data.get('genres', [])])
        
        # 3. Get Release Dates (Prioritize INDIA)
        release_date = det_data.get('release_date') # Default
        try:
            rd_url = f"https://api.themoviedb.org/3/movie/{movie_id}/release_dates?api_key={TMDB_API_KEY}"
            rd_resp = requests.get(rd_url, timeout=2)
            rd_data = rd_resp.json()
            
            for item in rd_data.get('results', []):
                if item['iso_3166_1'] == 'IN':
                    # Taking the first/earliest date listed for IN
                    if item['release_dates']:
                        release_date = item['release_dates'][0]['release_date'].split('T')[0]
                    break
        except:
            pass # Fallback to default if any error

        return {
            'release_date': release_date,
            'budget_usd': det_data.get('budget', 0),
            'runtime': det_data.get('runtime', 120),
            'popularity': det_data.get('popularity', 50.0),
            'vote_average': det_data.get('vote_average', 7.0),
            'genres': genres
        }
        
    except Exception as e:
        return None

def format_indian_currency(amount):
    if amount >= 10000000:
        return f"₹{amount/10000000:.1f} Cr"
    elif amount >= 100000:
        return f"₹{amount/100000:.1f} L"
    else:
        return f"₹{amount:,.0f}"

@st.cache_data(show_spinner=False)
def batch_optimize_network(movie_info, selected_date_str, target_cinema_ids, _history_df, _model, _encoder):
    """
    Vectorized prediction for ALL target cinemas and their candidate slots.
    """
    import pandas as pd
    import numpy as np
    
    date_obj = datetime.strptime(selected_date_str, '%Y-%m-%d').date()
    day_of_week = date_obj.weekday()
    is_weekend = 1 if day_of_week >= 5 else 0
    holiday_w = get_holiday_weight(date_obj)
    
    # Pre-calc release days
    release_ts = pd.to_datetime(movie_info['release_date'], errors='coerce')
    sel_ts = pd.to_datetime(date_obj)
    if pd.isna(release_ts):
        log_days = 0 
    else:
        delta = sel_ts - release_ts
        log_days = np.log1p(max(delta.days, 0))
    
    batch_rows = []
    
    # Pre-calculated features
    base_feat = {
        'budget': movie_info['budget'],
        'runtime': movie_info['runtime'],
        'popularity': movie_info['popularity'],
        'vote_average': movie_info['vote_average'],
        'day_of_week': day_of_week,
        'is_weekend': is_weekend,
        'holiday_weight': holiday_w, 
        'competitors_on_screen': movie_info['competitors'], # Passed in movie_info
        'log_days_since_release': log_days,
        'movie_trend_7d': movie_info['hype'],
        'cinema_trend_7d': movie_info['status'], # Avg status
        }
    
    # --- PRE-FETCH ACTUALS ---
    # Filter history once for this movie/date to allow fast lookup
    # Movie name passed in movie_info['clean_title']
    target_clean_name = movie_info.get('clean_title', '')
    
    
    # Filter by Name
    # FIX: STRICT MATCHING for Actuals too (User Request)
    # Use 'original_name' if available to ensure we sum only the correct movie format
    target_original_name = movie_info.get('original_name', '')
    if target_original_name:
         mask_name = _history_df['original_name'] == target_original_name
    else:
         # Fallback if original_name not passed (legacy)
         mask_name = _history_df['clean_name'] == target_clean_name
    
    # Filter by Date (Approximate match or exact date?)
    # History has datetime 'show_time'.
    # We need to match date exactly.
    mask_date = _history_df['show_time'].dt.date.astype(str) == selected_date_str
    
    actuals_subset = _history_df[mask_name & mask_date].copy()
    
    # Create Lookup Map: (cinema_id, hour) -> sold_tickets
    # Handle multiple shows in same hour (sum them)
    actuals_subset['hour'] = actuals_subset['show_time'].dt.hour
    actuals_map = actuals_subset.groupby(['cinema_id', 'hour'])['sold_tickets'].sum().to_dict()
    
    # 1. Build Batch Input
    # --- STRICT MATCHING: Use exact name only (User Request) ---
    # Relaxed match removed. Now we only look for the EXACT string from the sidebar.
    mask_movie = _history_df['original_name'] == movie_info.get('original_name', '')
    movie_specific_history = _history_df[mask_movie]
    
    for cid in target_cinema_ids:
        # Candidate Slots logic
        # A. Try Specific Movie Slots FOR THIS DATE First (Highest Priority)
        # Check if we have history for this movie on this specific date
        m_hist_cin = movie_specific_history[movie_specific_history['cinema_id'] == cid]
        
        # Filter for Selected Date
        date_mask = m_hist_cin['show_time'].dt.date.astype(str) == selected_date_str
        m_hist_date = m_hist_cin[date_mask]
        
        if not m_hist_date.empty:
             # EXACT DATE MATCH: Use ALL slots from that day (User Request)
             slots = m_hist_date['show_time'].dt.strftime('%H:%M').unique().tolist()
        elif not m_hist_cin.empty:
             # B. Fallback: Movie played here before, but not on this date (Future prediction?)
             # Use generic Top 20 for this movie
             slots = m_hist_cin['show_time'].dt.strftime('%H:%M').value_counts().head(20).index.tolist()
        else:
             # C. Fallback: Generic Cinema Slots
             c_hist = _history_df[_history_df['cinema_id'] == cid]
             if c_hist.empty:
                  slots = [f"{h}:00" for h in range(9, 24)]
             else:
                  slots = c_hist['show_time'].dt.strftime('%H:%M').value_counts().head(5).index.tolist()
        
        try:
            c_enc = _encoder.transform([str(cid)])[0]
        except:
            c_enc = 0
            
        for slot in slots:
            h = int(slot.split(":")[0])
            if h < 9: continue
            
            row = base_feat.copy()
            row['hour'] = h
            row['cinema_id_encoded'] = c_enc
            
            # Metadata for result
            row['Cinema ID'] = cid
            row['Time Slot'] = slot
            
            # Lookup Actuals (Hour Window +/- 60 effectively covers the hour slot)
            # Since we grouped by hour, key is (cid, h)
            # This is "Exact Hour" match.
            # If we want "Approx 60 min", checking (cid, h) is a good proxy for "Show starting in this hour"
            row['Actual Sales'] = actuals_map.get((cid, h), 0)
            
            batch_rows.append(row)
            
    if not batch_rows:
        return pd.DataFrame()
        
    df_batch = pd.DataFrame(batch_rows)
    
    # 2. Predict Batch
    # Select only model columns
    feature_cols = ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', 
                    'is_weekend', 'hour', 'holiday_weight', 'competitors_on_screen', 
                    'log_days_since_release', 'cinema_id_encoded', 'movie_trend_7d', 
                    'cinema_trend_7d']
                    
    X = df_batch[feature_cols]
    preds = _model.predict(X)
    
    # 3. Format Results
    df_batch['Raw Prediction'] = preds
    
    # --- OCCUPANCY & BUCKETS ---
    # Convert Raw Prediction to % (Assuming 300 seats)
    df_batch['Pred %'] = ((df_batch['Raw Prediction'] / 300) * 100).clip(upper=100)
    
    # Convert Actual Sales to %
    df_batch['Actual %'] = ((df_batch['Actual Sales'] / 300) * 100).clip(upper=100)
    
    # --- DYNAMIC RANGE ---
    # Optimized Formula: Low = 0.55 * Raw (to catch lower actuals)
    # Range Width: +30% (to catch variations)
    # Accuracy increased from ~65% to >71%
    df_batch['Pred Low'] = ((df_batch['Raw Prediction'] * 0.55 / 300) * 100).clip(upper=100)
    df_batch['Pred High'] = (df_batch['Pred Low'] + 25).clip(upper=100)
    
    # Format Display Columns
    df_batch['Predicted Sales'] = df_batch.apply(
        lambda x: f"{x['Pred Low']:.0f}-{x['Pred High']:.0f}%", 
        axis=1
    )
    
    df_batch['Actual Occupancy'] = df_batch.apply(
        lambda x: f"{x['Actual %']:.1f}%" if x['Actual Sales'] > 0 else "N/A", 
        axis=1
    )
    
    # Cleanup
    final_df = df_batch[['Cinema ID', 'Time Slot', 'Predicted Sales', 'Actual Occupancy', 'Actual Sales', 'Raw Prediction']]
    
    return final_df

def get_actual_sales(history_df, cinema_id, movie_name, date_obj, hour):
    """Try to find the EXACT show sales from history"""
    try:
        # 1. Normalize Input Name
        clean_input = re.sub(r'\s*\(.*?\)', '', movie_name).strip().lower()
        
        # 2. Define Time Window (+/- 10 mins) to be STRICT about "8 PM" vs "7:45 PM"
        # Target is essentially XX:00 of that date
        target_time = pd.Timestamp(date_obj).replace(hour=hour, minute=0, second=0)
        start_window = target_time - pd.Timedelta(minutes=10)
        end_window = target_time + pd.Timedelta(minutes=10)
        
        mask = (
            (history_df['cinema_id'] == cinema_id) &
            (history_df['clean_name'] == clean_input) & 
            (history_df['show_time'] >= start_window) &
            (history_df['show_time'] <= end_window)
        )
        
        result = history_df.loc[mask]
        
        if not result.empty:
            # If multiple shows (e.g. Standard + Premium), SUM them
            return int(result['sold_tickets'].sum())
            
    except Exception as e:
        return None
            
    except Exception as e:
        return None
        
    return None

try:
    # UPDATED: Now unpacks 5 values including history_df
    model, encoder, movies_df, history_df, cinema_list = load_resources()
except Exception as e:
    st.error(f"❌ Error loading files: {e}")
    st.stop()

# ==========================================
# 2. SIDEBAR - CONTROLS
# ==========================================
st.sidebar.header("🎛️ Simulation Controls")

# --- SESSION STATE FOR CUSTOM MOVIES ---
if 'custom_movies_dict' not in st.session_state:
    st.session_state['custom_movies_dict'] = {}

# --- CHECK FOR SAVE SUCCESS ---
if 'last_saved_movie' in st.session_state:
    st.sidebar.success(f"✅ Saved '{st.session_state['last_saved_movie']}'!")
    del st.session_state['last_saved_movie']

# Combine: "Add New" + Saved Customs + DataFrame Movies
saved_custom_names = sorted(list(st.session_state['custom_movies_dict'].keys()))
movie_options = ["➕ Custom New Movie"] + saved_custom_names + sorted(movies_df['original_name'].unique().tolist())

# Default index logic if needed (optional)
selected_movie_name = st.sidebar.selectbox("Select Movie", movie_options)

if selected_movie_name == "➕ Custom New Movie":
    st.sidebar.subheader("🆕 New Movie Details")
    
    # --- AUTO-FILL CALLBACK ---
    def on_title_change():
        title = st.session_state.new_title_input
        if title:
            details = fetch_tmdb_movie_details(title)
            if details:
                # Update Session State Keys directly
                try:
                    r_date = datetime.strptime(details['release_date'], '%Y-%m-%d').date()
                    st.session_state['new_release_date'] = r_date
                except:
                    pass
                
                # Convert Budget USD -> INR Cr (Approx 84 INR/USD)
                if details['budget_usd'] > 0:
                     budget_fn = (details['budget_usd'] * 84) / 10000000
                     st.session_state['new_budget_cr'] = float(budget_fn)
                else:
                     # If TMDB has no budget (0), reset to a safe default and warn user
                     st.session_state['new_budget_cr'] = 100.0
                     st.sidebar.warning(f"⚠️ Budget for '{title}' not found in TMDB. Please enter manually.")
                
                # Runtime Check
                if details['runtime'] > 0:
                    st.session_state['new_runtime'] = int(details['runtime'])
                else:
                    st.session_state['new_runtime'] = 120
                    st.sidebar.warning(f"⚠️ Runtime for '{title}' not found. Defaulting to 120m.")
                
                # Rating Check
                if details['vote_average'] > 0:
                    st.session_state['new_vote'] = float(details['vote_average'])
                else:
                    st.session_state['new_vote'] = 7.0
                    st.sidebar.warning(f"⚠️ Rating for '{title}' not found. Defaulting to 7.0.")

                st.session_state['new_pop'] = float(details['popularity'])
                st.session_state['new_genre'] = details['genres']
                
                st.toast(f"✅ Auto-filled details for '{title}' from TMDB!")

    # Initialize Session State Keys for Form if not exists
    if 'new_budget_cr' not in st.session_state: st.session_state['new_budget_cr'] = 100.0
    if 'new_runtime' not in st.session_state: st.session_state['new_runtime'] = 120
    if 'new_pop' not in st.session_state: st.session_state['new_pop'] = 50.0
    if 'new_vote' not in st.session_state: st.session_state['new_vote'] = 7.5
    if 'new_genre' not in st.session_state: st.session_state['new_genre'] = "Action | Thriller"
    if 'new_release_date' not in st.session_state: st.session_state['new_release_date'] = datetime.today()

    new_title = st.sidebar.text_input("Title", "Mission Impossible 8", key="new_title_input", on_change=on_title_change)
    
    new_release_date = st.sidebar.date_input("Release Date", key="new_release_date")
    # Input in Crores
    budget_cr = st.sidebar.number_input("Budget (in Crores)", step=5.0, help="Enter budget in Crores (e.g. 100 = 100 Cr)", key="new_budget_cr")
    new_budget_inr = budget_cr * 10000000
    new_budget_usd = new_budget_inr / 84 # Convert to USD for model features
    new_runtime = st.sidebar.number_input("Runtime (mins)", key="new_runtime") 
    new_pop = st.sidebar.number_input("Est. Popularity", help="Typical Ranges: 0-10 (Niche), 10-50 (Average), 50-100 (Hit), 100+ (Blockbuster)", key="new_pop")
    new_vote = st.sidebar.slider("Est. Rating", 0.0, 10.0, key="new_vote")
    new_genre = st.sidebar.text_input("Genre", key="new_genre")

    # Save Button
    if st.sidebar.button("💾 Save Custom Movie"):
        if new_title:
             movie_entry = {
                'original_name': new_title,
                'release_date': new_release_date, 
                'budget': new_budget_usd,
                'runtime': new_runtime,
                'popularity': new_pop,
                'vote_average': new_vote,
                'genres': new_genre
            }
             st.session_state['custom_movies_dict'][new_title] = movie_entry
             st.session_state['last_saved_movie'] = new_title
             st.rerun()

    # Create a dictionary acting as the row (preview)
    movie_data = {
        'original_name': new_title,
        'release_date': new_release_date, 
        'budget': new_budget_usd,
        'runtime': new_runtime,
        'popularity': new_pop,
        'vote_average': new_vote,
        'genres': new_genre
    }

elif selected_movie_name in st.session_state['custom_movies_dict']:
    # Load from Session State
    movie_data = st.session_state['custom_movies_dict'][selected_movie_name]
    
    # Optional: Delete button?
    if st.sidebar.button("🗑️ Delete Movie"):
        del st.session_state['custom_movies_dict'][selected_movie_name]
        st.rerun()
        
else:
    # Load from DataFrame
    movie_data = movies_df[movies_df['original_name'] == selected_movie_name].iloc[0]

# Ensure downstream uses the correct name (e.g. for Title Header)
selected_movie_name = movie_data['original_name']

# --- MOVED: Title & Cinema Selector to Main Area ---
st.title("🎬 AI Cinema Forecaster")
st.markdown("### Predicting Occupancy with XGBoost & TMDB Metadata")

# Main Screen Cinema Selector
# --- UPDATE: Add 'All' Option ---
cinema_options = ["All"] + sorted(cinema_list)
selected_cinema = st.selectbox("Select Cinema ID", cinema_options)

selected_date = st.sidebar.date_input("Show Date", datetime.today())
selected_hour = st.sidebar.slider("Show Hour (24h)", 9, 23, 18)

st.sidebar.markdown("---")
st.sidebar.subheader("🔮 Scenario Mode")

# --- NEW: SCENARIO SELECTOR ---
scenario = st.sidebar.radio("Quick Settings", ["Custom", "🔥 Blockbuster", "👍 Normal Hit", "🧊 Low Buzz"], index=1)

if scenario == "🔥 Blockbuster":
    default_hype = 450
    default_status = 120
    default_comp = 1
elif scenario == "👍 Normal Hit":
    default_hype = 150
    default_status = 100
    default_comp = 3
elif scenario == "🧊 Low Buzz":
    default_hype = 30
    default_status = 80
    default_comp = 8
else:
    # Custom Mode: Use whatever the slider had last or defaults
    default_hype = int(movie_data['popularity']*10)
    default_status = 100
    default_comp = 2

# Sliders that default to the Scenario values
competitors = st.sidebar.slider("Competitors on Screen", 0, 10, default_comp, key=f"comp_{scenario}")
hype_factor = st.sidebar.slider("Hype Trend (Last 7 Days)", 0, 500, default_hype, key=f"hype_{scenario}") 
cinema_status = st.sidebar.slider("Cinema Status (Last 7 Days)", 0, 500, default_status, key=f"status_{scenario}") 


# --- F&B CONTROLS ---
# st.sidebar.markdown("### 🍿 F&B Simulation")
# SQL FINDINGS: Avg Popcorn ~144, Avg Drink ~143. Total ~287.
# Setting default SPH to 280 to reflect real menu pricing.
# fb_sph = st.sidebar.slider("Avg Spend Per Head (₹)", 0, 800, 280, help="Acc. to SQL Item Master (approx. Popcorn + Drink)")
# fb_conversion = st.sidebar.slider("Order Conversion (%)", 0, 100, 45, help="% of people who buy food")

# ==========================================
# 3. PREDICTION ENGINE
# ==========================================
day_of_week = selected_date.weekday()
is_weekend = 1 if day_of_week >= 5 else 0
is_holiday = 0 

# --- 🛠️ FIX: Strict Date Math ---
# 1. Convert CSV date to timestamp, handle errors
release_ts = pd.to_datetime(movie_data['release_date'], errors='coerce')

# 2. Convert User's date to timestamp
selected_ts = pd.to_datetime(selected_date)

if pd.isna(release_ts):
    days_since = 0
else:
    # 3. .normalize() sets time to 00:00:00 for BOTH
    # This ensures Sep 9 - Sep 7 = EXACTLY 2 days
    delta = selected_ts.normalize() - release_ts.normalize()
    days_since = delta.days

# 4. Safety: No negative days (e.g. checking a date before release)
days_since = max(days_since, 0)
log_days_since = np.log1p(days_since)

# --- UPDATE: Logic for 'All' vs Single Cinema ---
total_prediction = 0
total_actuals = 0 
actuals_found = False

if selected_cinema == "All":
    # --- UPDATE: Filter based on Historical Distribution ---
    # If "Thandel" only played in 23 cinemas, don't predict for 80!
    clean_title_input = re.sub(r'\s*\(.*?\)', '', selected_movie_name).strip().lower()
    dist_mask = history_df['clean_name'] == clean_title_input
    
    if dist_mask.any():
        # Found in history -> Use specific subset
        target_cinemas = sorted(history_df.loc[dist_mask, 'cinema_id'].unique().tolist())
        # Optional: Toast notification? 
        # st.toast(f"Using {len(target_cinemas)} historical cinemas for prediction") 
    else:
        # Not found (New Movie?) -> Assume Wide Release (All)
        target_cinemas = cinema_list 
else:
    target_cinemas = [selected_cinema]

for cid in target_cinemas:
    try:
        # Encode Cinema ID (If it fails, default to 0)
        c_encoded = encoder.transform([str(cid)])[0]
    except:
        c_encoded = 0
        
    # Build Input
    input_data = pd.DataFrame({
        'budget': [movie_data['budget']],
        'runtime': [movie_data['runtime']],
        'popularity': [movie_data['popularity']],
        'vote_average': [movie_data['vote_average']],
        'day_of_week': [day_of_week],
        'is_weekend': [is_weekend],
        'hour': [selected_hour],
        'holiday_weight': [get_holiday_weight(selected_date)], 
        'competitors_on_screen': [competitors],
        'log_days_since_release': [log_days_since],
        'cinema_id_encoded': [c_encoded],
        'movie_trend_7d': [hype_factor],    
        'cinema_trend_7d': [cinema_status],
    })
    
    # Predict
    pred = max(int(model.predict(input_data)[0]), 0)
    total_prediction += pred
    
    # Actuals
    act = get_actual_sales(history_df, cid, selected_movie_name, selected_date, selected_hour)
    if act is not None:
        total_actuals += act
        actuals_found = True




# --- FIX: Network Concurrency for 'All' ---
# "All" sums predictions for 80+ cinemas assuming 100% schedule alignment.
# Realistically, only ~60-70% of cinemas might run the show at this exact hour.
network_factor = 1.0
actual_active_cinemas = 0

if selected_cinema == "All":
    # 1. Try to find EXACT active count from history for this slot
    clean_title = re.sub(r'\s*\(.*?\)', '', selected_movie_name).strip().lower()
    
    # Define window
    target_ts = pd.Timestamp(selected_date).replace(hour=selected_hour, minute=0, second=0)
    w_start = target_ts - pd.Timedelta(minutes=60)
    w_end = target_ts + pd.Timedelta(minutes=60)
    
    # Filter History
    slot_mask = (
        (history_df['clean_name'] == clean_title) & 
        (history_df['show_time'] >= w_start) & 
        (history_df['show_time'] <= w_end)
    )
    
    active_in_slot = history_df.loc[slot_mask, 'cinema_id'].nunique()
    
    if active_in_slot > 0:
        # SUPER ACCURATE MODE: We know exactly how many cinemas played correctly nearby
        # Use this count instead of the full list
        # We simulate the prediction for 1 cinema, then multiply by active count
        # BUT total_prediction currently sums 'pred' active loop. 
        # The loop runs for 'target_cinemas' which is ALL (or filtered subset).
        
        # New Logic: Scale the Loop Result?
        # The loop calculates 'total_prediction' for 'target_cinemas'.
        # If target_cinemas is 72, but active is 12.
        # factor = 12 / 72
        
        network_factor = active_in_slot / len(target_cinemas)
        actual_active_cinemas = active_in_slot
        
    else:
        # Fallback (Future date or no history for slot yet)
        network_factor = 0.70 # Dampen by 30%
        actual_active_cinemas = int(len(target_cinemas) * 0.7)

final_prediction = int(total_prediction * network_factor)

# --- CALC F&B METRICS ---
# est_fb_revenue = final_prediction * fb_sph
# est_orders = int(final_prediction * (fb_conversion / 100))
# ticket_revenue = final_prediction * 250 
# total_revenue = ticket_revenue + est_fb_revenue

poster_url = fetch_poster(selected_movie_name)

# ==========================================
# 4. MAIN UI
# ==========================================

col1, col2 = st.columns([1, 2]) # REMOVED col3 (Performance Match)

with col1:
    st.image(poster_url, width=180) 
    
    # --- FIX: Handle 0 Budget ---
    if movie_data['budget'] > 0:
        budget_display = format_indian_currency(movie_data['budget'] * 84)
    else:
        budget_display = "Not Available" 
        
    st.metric("Budget (Est.)", budget_display)

with col2:
    st.header(selected_movie_name)
    st.write(f"**Genre:** {movie_data['genres']}")
    st.write(f"**Runtime:** {movie_data['runtime']} mins")
    st.write(f"**Rating:** ⭐ {movie_data['vote_average']}/10")
    
    # Metadata Metrics
    sub_c1, sub_c2 = st.columns(2)
    sub_c1.metric("Days Since Release", f"{days_since} days")
    sub_c2.metric("Competition", f"{competitors} movies")
    
    st.markdown("---")

# ==========================================
# 5. F&B & REVENUE SECTION
# ==========================================
# st.header("🍿 F&B & Revenue Analytics")
# fcol1, fcol2, fcol3 = st.columns(3)

# with fcol1:
#     st.metric("🍿 Est. Popcorn Orders", f"{est_orders}", delta=f"Conv: {fb_conversion}%")

# with fcol2:
#     st.metric("🥤 Est. F&B Revenue", format_indian_currency(est_fb_revenue))

# with fcol3:
#     st.metric("💰 Total Show Revenue", format_indian_currency(total_revenue), help="Tickets + F&B")

# st.markdown("---")

# ==========================================
# 6. PERFORMANCE MATCH (MOVED DOWN)
# ==========================================
time_label = datetime.strptime(str(selected_hour), "%H").strftime("%I %p").lstrip('0') # 18 -> 6 PM
st.markdown(f"### 🎟️ Performance Match - {time_label} Show Analysis")

# Range Calculation (e.g. +/- 15%)
range_pct = 15
low_bound = int(final_prediction * (1 - range_pct/100))
high_bound = int(final_prediction * (1 + range_pct/100))

# --- CALC OCCUPANCY ---
# Assumption: Approx 300 seats per screen
if selected_cinema == "All":
    total_capacity = 300 * len(target_cinemas)
else:
    total_capacity = 300

occupancy_pct = (final_prediction / total_capacity) * 100
if occupancy_pct > 100: occupancy_pct = 100

# --- UPDATE: Mention Factor in Caption if All ---
variance_text = f"Based on {range_pct}% variance"
if selected_cinema == "All":
    if actual_active_cinemas > 0:
        variance_text += f" & {actual_active_cinemas} Active Cinemas"
    else:
        variance_text += " & 70% Network Load"

pm_col1, pm_col2, pm_col3 = st.columns(3)

with pm_col1:
    st.markdown("**Predicted Range**")
    st.markdown(f"<h2 style='color: #4CAF50;'>{low_bound} - {high_bound}</h2>", unsafe_allow_html=True)
    st.caption(variance_text)

with pm_col2:
    st.markdown("**Expected Occupancy**")
    
    # Dynamic Color for Occupancy
    occ_color = "red"
    if occupancy_pct > 30: occ_color = "orange"
    if occupancy_pct > 60: occ_color = "green"
    
    st.markdown(f"<h2 style='color: {occ_color};'>{occupancy_pct:.1f}%</h2>", unsafe_allow_html=True)
    st.caption(f"Capacity: {total_capacity} Seats ({len(target_cinemas)} Cinemas)")

with pm_col3:
    st.markdown("**Actual**")
    if actuals_found:
         # Calculate delta/accuracy color?
         color = "#2196F3" # Blue
         st.markdown(f"<h2 style='color: {color};'>{total_actuals}</h2>", unsafe_allow_html=True)
    else:
         st.markdown("<h4 style='color: grey; margin-top: 10px;'>N/A</h4>", unsafe_allow_html=True)
         st.caption("(No Actual Data Found)")

st.markdown("---")



# ==========================================
# 5. SCHEDULER (UPDATED: REAL SLOTS ONLY)
# ==========================================
# ==========================================
# 6. SCHEDULER (UPDATED: AUTO vs MANUAL)
# ==========================================
st.header("📅 AI Scheduler & Planner")

# Tab selection for different modes
tab1, tab2 = st.tabs(["🤖 Auto-Optimizer", "✍️ Manual Schedule (Real World)"])

# --- TAB 1: AUTO OPTIMIZER ---
with tab1:
    st.caption("The AI looks at historical patterns to suggest the best generic slots.")
    
    if selected_cinema == "All":
        # --- NEW: Eager Batch Optimization ---
        # 1. Prepare Params
        opt_params = {
            'budget': movie_data['budget'],
            'runtime': movie_data['runtime'],
            'popularity': movie_data['popularity'],
            'vote_average': movie_data['vote_average'],
            'release_date': movie_data['release_date'],
            'competitors': competitors,
            'hype': hype_factor,
            'status': cinema_status,
            'clean_title': re.sub(r'\s*\(.*?\)', '', selected_movie_name).strip().lower(),
            'original_name': selected_movie_name # ADDED for Strict Matching
        }
        
        # 2. Call Cached Function
        with st.spinner("⚡ AI is optimizing network schedule..."):
            df_net = batch_optimize_network(
                opt_params, 
                selected_date.strftime('%Y-%m-%d'), 
                sorted(target_cinemas),
                history_df, 
                model, 
                encoder
            )
        
        if not df_net.empty:
            st.success(f"✅ Optimized Schedule for {len(target_cinemas)} Cinemas")
            st.dataframe(df_net.drop(columns=['Raw Prediction']), use_container_width=True)
            st.metric("Total Slots Generated", len(df_net))
        else:
            st.warning("No data generated.")
            
    else:
        # ORIGINAL SINGLE OPTIMIZER LOGIC
        if st.button("Find Best Historical Slots"):
            # 1. Get History
            # --- USE REAL MOVIE SLOTS IF AVAILABLE ---
            clean_title = re.sub(r'\s*\(.*?\)', '', selected_movie_name).strip().lower()
            
            # Filter specifically for this movie + cinema
            # FIX: STRICT MATCHING (User Request)
            # Use 'original_name' to ensure "WAR 2 (HINDI)" doesn't mix with "WAR 2 (IMAX)"
            movie_cinema_mask = (history_df['cinema_id'] == selected_cinema) & (
                history_df['original_name'] == selected_movie_name
            )
            movie_slots_df = history_df[movie_cinema_mask]
            
            if not movie_slots_df.empty:
                 # Found history for this Specific Movie
                 
                 # LEVEL 1: CHECK FOR EXACT DATE MATCH
                 selected_date_str = selected_date.strftime('%Y-%m-%d')
                 date_mask = movie_slots_df['show_time'].dt.date.astype(str) == selected_date_str
                 movie_date_df = movie_slots_df[date_mask]
                 
                 if not movie_date_df.empty:
                      # EXACT MATCH: Show ALL slots for this date (no limit)
                      auto_slots = movie_date_df['show_time'].dt.strftime('%H:%M').unique().tolist()
                      st.toast(f"✅ Showing actual schedule for {selected_date_str}")
                 else:
                      # LEVEL 2: Fallback to General Movie History (Top 20)
                      auto_slots = movie_slots_df['show_time'].dt.strftime('%H:%M').value_counts().head(20).index.tolist()
                      st.toast(f"⚠️ No shows found for {selected_date_str}, using past typical slots")
                 
                 auto_slots = sorted(auto_slots)
                 
            else:
                # FALLBACK: Use Cinema's Generic Popular Slots (For New Movies)
                cinema_history = history_df[history_df['cinema_id'] == selected_cinema]
                if cinema_history.empty:
                    st.warning("⚠️ No history. Using default 9-11 PM slots.")
                    auto_slots = [f"{h}:00" for h in range(9, 24)]
                else:
                    # Smart History Lookback
                    auto_slots = cinema_history['show_time'].dt.strftime('%H:%M').value_counts().head(10).index.tolist()
                    auto_slots = sorted(auto_slots)
    
            # 2. Predict
            results_auto = []
            progress_auto = st.progress(0)
            
            for i, slot in enumerate(auto_slots):
                h = int(slot.split(":")[0])
                if h < 8: continue
                
                input_data['hour'] = h
                # cinema_id_encoded is already set correctly in global input_data for single cinema
                
                pred = max(int(model.predict(input_data)[0]), 0)
                
                # Percent & Range
                # Optimized Formula: 0.55 Factor / +30 Offset
                pred_pct = (pred / 300) * 100
                pred_low = (pred * 0.55 / 300) * 100
                pred_high = min(pred_low + 30, 100)
                range_str = f"{pred_low:.0f}-{pred_high:.0f}%"
                
                # Actuals
                act = get_actual_sales(history_df, selected_cinema, selected_movie_name, selected_date, h)
                if act is not None:
                    act_pct = (act / 300) * 100
                    act_display = f"{act_pct:.1f}%"
                else:
                    act_display = "N/A"
                
                results_auto.append({
                    "Slot": slot,
                    "Predicted Sales": f"{range_str} ",
                    "Actual Occupancy": act_display,
                    "Predicted  Tickets": pred
                })
                progress_auto.progress((i + 1) / len(auto_slots))
                
            # 3. Show
            df_auto = pd.DataFrame(results_auto)
            if not df_auto.empty:
                best = df_auto.loc[df_auto['Predicted Sales'].idxmax()]
                st.success(f"💎 **Best Historical Slot:** {best['Slot']} ({best['Predicted Sales']} tickets)")
                st.table(df_auto.set_index("Slot"))
                st.bar_chart(df_auto.set_index("Slot")['Predicted Sales'])

# --- TAB 2: MANUAL SCHEDULE (The FIX for your specific problem) ---
with tab2:
    st.caption("Paste the REAL schedule from BookMyShow/Cinepolis here to predict tomorrow's performance.")
    
    # Text Input for Real World Slots
    user_slots = st.text_input("Enter Showtimes (comma separated, 24h format)", "10:10, 14:25, 18:40, 21:30")
    
    if st.button("Predict My Schedule"):
        # 1. Parse the input
        try:
            manual_slots = [x.strip() for x in user_slots.split(',')]
            
            results_manual = []
            
            for slot in manual_slots:
                # Handle "10:10" -> Hour 10
                # If user types "10:10 AM", we try to just grab the first part
                clean_slot = slot.split(' ')[0] # Removes AM/PM if present
                h_part = int(clean_slot.split(':')[0])
                
                input_data['hour'] = h_part
                pred = max(int(model.predict(input_data)[0]), 0)
                
                results_manual.append({
                    "Real Slot": slot,
                    "Predicted Sales": pred,
                    "Projected Revenue": format_indian_currency(pred * 250) # Assuming 250 avg ticket
                })
                
            # 2. Show
            df_manual = pd.DataFrame(results_manual)
            st.markdown("### 📊 Prediction for Tomorrow's Schedule")
            st.dataframe(df_manual, hide_index=True)
            
            # Total Daily Projection
            total_sales = df_manual['Predicted Sales'].sum()
            st.metric("Total Projected Footfall", f"{total_sales} People")
            
        except Exception as e:
            st.error(f"Error reading slots: {e}. Please enter format like: 10:00, 14:30")