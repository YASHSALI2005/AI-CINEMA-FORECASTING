import pandas as pd

print("📊 Counting Cinemas...")

# Load your V3 dataset
try:
    df = pd.read_csv("final_training_data_v3.csv")
    
    # Get unique Cinema IDs
    unique_cinemas = df['cinema_id'].unique()
    count = len(unique_cinemas)
    
    print(f"\n✅ Total Unique Cinemas in Database: {count}")
    print(f"🆔 List of Cinema IDs: {sorted(unique_cinemas)}")

except FileNotFoundError:
    print("❌ Error: 'final_training_data_v3.csv' not found. Make sure you are in the right folder.")