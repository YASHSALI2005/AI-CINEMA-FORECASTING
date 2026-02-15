import pandas as pd
import json
import re
import os
from thefuzz import fuzz, process

# --- CONFIG ---
CSV_FILE = "final_training_data_from_dump.csv"
JSON_FILE = "bh_box_office_data.json"

def clean_currency(value):
    if not value or value == 'N/A' or value == '-':
        return 0.0
    
    s = str(value).lower()
    # Remove common symbols first
    s = s.replace('cr.', '').replace('cr', '').replace('₹', '').replace(',', '').strip()
    
    try:
        return float(s)
    except:
        # Fallback: extract first valid number pattern
        match = re.search(r'(\d+\.?\d*)', s)
        if match:
            try: return float(match.group(1))
            except: pass
        return 0.0

def get_verdict_score(verdict):
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
    return verdict_map.get(verdict, 0)

def main():
    print(f"🔄 Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return

    print(f"🔄 Loading {JSON_FILE}...")
    if not os.path.exists(JSON_FILE):
        print(f"❌ JSON file not found: {JSON_FILE}")
        return
    
    with open(JSON_FILE, 'r', encoding='utf-8') as f:
        bh_data = json.load(f)

    # Create Lookup Dictionary
    print("🔍 Building BH Lookup Dictionary...")
    bh_lookup = {}
    for entry in bh_data:
        name = entry.get('original_name')
        if not name: continue
        
        summary = entry.get('summary', {})
        bh_lookup[name.lower().strip()] = {
            'bh_opening_day': clean_currency(summary.get('opening_day')),
            'bh_verdict_score': get_verdict_score(summary.get('verdict', 'Unknown'))
        }

    bh_names = list(bh_lookup.keys())

    # Ensure columns exist
    if 'bh_opening_day' not in df.columns:
        df['bh_opening_day'] = 0.0
    if 'bh_verdict_score' not in df.columns:
        df['bh_verdict_score'] = 0

    print("🚀 Updating CSV Rows...")
    matched_count = 0
    unique_movies = df['original_name'].dropna().unique()
    print(f"   Note: Found {len(unique_movies)} unique movies in CSV.")
    
    # Pre-calculate matches for unique names
    name_map = {} 
    
    def clean_csv_name(name):
        # Remove (Hindi), (English), (3D), etc.
        name = str(name).lower()
        # Remove parentheses content
        name = re.sub(r'\(.*?\)', '', name)
        # Remove extra spaces
        name = re.sub(r'\s+', ' ', name).strip()
        # Remove common suffixes like '3d', '2d' if redundant
        name = name.replace(' 3d', '').replace(' 2d', '')
        return name

    for original_name in unique_movies:
        # Strategy 1: Exact Match (Raw)
        clean_raw = str(original_name).lower().strip()
        if clean_raw in bh_lookup:
            name_map[original_name] = bh_lookup[clean_raw]
            matched_count += 1
            continue
        
        # Strategy 2: Cleaned Name Match
        cleaned_name = clean_csv_name(original_name)
        if cleaned_name in bh_lookup:
             name_map[original_name] = bh_lookup[cleaned_name]
             matched_count += 1
             continue
             
        # Strategy 3: Fuzzy Match on Cleaned Name
        if len(cleaned_name) > 3: # Avoid fuzzing very short names
            best_match, score = process.extractOne(cleaned_name, bh_names, scorer=fuzz.token_set_ratio)
            if score >= 90:
                name_map[original_name] = bh_lookup[best_match]
                matched_count += 1
    
    print(f"   Matched {matched_count} unique movies with BH data.")
    print("   Samples of matched movies:")
    for m in list(name_map.keys())[:10]:
        print(f"    - {m} -> {name_map[m]}")

    # Apply mappings
    # map() is faster than iterating rows
    
    # Create mappers
    od_map = {name: data['bh_opening_day'] for name, data in name_map.items()}
    vs_map = {name: data['bh_verdict_score'] for name, data in name_map.items()}
    
    # Update columns
    # We use map and fallback to existing value (which is 0 initially)
    # But wait, if we initialized to 0, fillna(0) is fine.
    
    print("⏳ Applying updates to dataframe...")
    df['bh_opening_day'] = df['original_name'].map(od_map).fillna(0)
    df['bh_verdict_score'] = df['original_name'].map(vs_map).fillna(0)
    
    # Save
    print(f"💾 Saving to {CSV_FILE}...")
    df.to_csv(CSV_FILE, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()
