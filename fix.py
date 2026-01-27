import pandas as pd

# 1. Load the file
csv_file = "movie_features_safe.csv"
try:
    df = pd.read_csv(csv_file)
except FileNotFoundError:
    print("❌ Error: movie_features_safe.csv not found.")
    exit()

# 2. Define the correct date
correct_date = "2024-12-05"
target_keyword = "PUSHPA (THE RULE PART -02)"

# 3. Find all rows that match the keyword
# FIX: Added 'regex=False' so the ( ) brackets don't crash the script
mask = df['original_name'].str.contains(target_keyword, case=False, na=False, regex=False)
count = mask.sum()

if count > 0:
    print(f"✅ Found {count} versions of Pushpa 2!")
    
    # Update ALL of them
    df.loc[mask, 'release_date'] = correct_date
    
    # 4. Save back to CSV
    df.to_csv(csv_file, index=False)
    print(f"🎉 Success! Updated {count} rows to Release Date: {correct_date}")
    
else:
    print(f"❌ Error: Still could not find any movie matching '{target_keyword}'")