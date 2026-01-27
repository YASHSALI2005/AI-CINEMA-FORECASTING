import pandas as pd
import re

print("Loading data...")
df = pd.read_csv("final_training_data_v3.csv", usecols=['original_name', 'cinema_id'])

# Normalize
df['clean_name'] = df['original_name'].astype(str).dropna().apply(
    lambda x: re.sub(r'\s*\(.*?\)', '', x).strip().lower()
)

target = "thandel"
print(f"Checking for '{target}'...")

subset = df[df['clean_name'] == target]
unique_cinemas = subset['cinema_id'].nunique()
print(f"Cinemas showing 'Thandel': {unique_cinemas}")

# Compare to Total Cinemas
total_cinemas = df['cinema_id'].nunique()
print(f"Total Cinemas in Network: {total_cinemas}")

if unique_cinemas < total_cinemas:
    print("INSIGHT: The movie is NOT playing in all cinemas. Prediction must be filtered.")
