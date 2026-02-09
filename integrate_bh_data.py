import pandas as pd
import json
import re
import os

def clean_currency(value):
    if not value or value == 'N/A' or value == '-':
        return 0.0
    # Remove Currency symbols and commas
    clean = re.sub(r'[^\d.]', '', value)
    try:
        return float(clean)
    except:
        return 0.0

from thefuzz import fuzz, process

def integrate_bh_data(json_file='bh_box_office_data.json', csv_file='movie_features_safe.csv'):
    print(f"🔄 Integrating data from {json_file} into {csv_file}...")
    
    if not os.path.exists(json_file):
        print(f"❌ JSON file not found: {json_file}")
        return
    
    with open(json_file, 'r', encoding='utf-8') as f:
        bh_data = json.load(f)
    
    # Create a BH Lookup Dictionary
    bh_lookup = {}
    for entry in bh_data:
        name = entry.get('original_name')
        if not name: continue
        
        summary = entry.get('summary', {})
        bh_lookup[name.lower().strip()] = {
            'bh_name': name,
            'bh_opening_day': clean_currency(summary.get('opening_day')),
            'bh_opening_weekend': clean_currency(summary.get('opening_weekend')),
            'bh_lifetime': clean_currency(summary.get('lifetime')),
            'bh_verdict': summary.get('verdict', 'Unknown')
        }
    
    bh_names = list(bh_lookup.keys())
    
    # Load main features
    df = pd.read_csv(csv_file)
    
    # Matching Logic
    new_cols = ['bh_name', 'bh_opening_day', 'bh_opening_weekend', 'bh_lifetime', 'bh_verdict']
    for col in new_cols:
        df[col] = None

    matched_count = 0
    for idx, row in df.iterrows():
        original_name = str(row['original_name']).lower().strip()
        
        # 1. Exact Match
        if original_name in bh_lookup:
            for col in new_cols:
                df.at[idx, col] = bh_lookup[original_name][col]
            matched_count += 1
            continue
            
        # 2. Fuzzy Match (Threshold 90)
        best_match, score = process.extractOne(original_name, bh_names, scorer=fuzz.token_set_ratio)
        if score >= 90:
            for col in new_cols:
                df.at[idx, col] = bh_lookup[best_match][col]
            matched_count += 1
    
    # Save the updated features
    output_file = 'movie_features_with_bh.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ Integration complete. Saved to {output_file}")
    print(f"📊 Matched {matched_count} movies out of {len(df)}")

if __name__ == "__main__":
    integrate_bh_data()
