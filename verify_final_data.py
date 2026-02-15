import pandas as pd

def check():
    print("Loading CSV...")
    df = pd.read_csv('final_training_data_from_dump.csv')
    print(f"Total Rows: {len(df)}")
    
    non_zero = (df['bh_opening_day'] > 0).sum()
    print(f"Rows with bh_opening_day > 0: {non_zero}")
    
    unique_with_data = df[df['bh_opening_day'] > 0]['original_name'].nunique()
    print(f"Unique movies with data: {unique_with_data}")
    
    print("\nTop 10 movies by bh_opening_day:")
    print(df.groupby('original_name')['bh_opening_day'].max().sort_values(ascending=False).head(15))
    
    # Check specific 2023 hits
    hits = ["DUNKI", "ANIMAL", "JAWAN", "GADAR 2", "PATHAAN"]
    print("\nChecking specific hits:")
    for h in hits:
        # fuzzy match check
        mask = df['original_name'].str.contains(h, case=False, na=False)
        if mask.any():
            val = df[mask]['bh_opening_day'].max()
            print(f"  {h}: {val}")
        else:
            print(f"  {h}: Not found")

if __name__ == "__main__":
    check()
