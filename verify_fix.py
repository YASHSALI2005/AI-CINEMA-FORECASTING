import pandas as pd
import re

print("Loading history...")
history_cols = ['original_name', 'sold_tickets', 'cinema_id']
df = pd.read_csv("final_training_data_v3.csv", usecols=history_cols)

# Apply the same normalization as app.py
print("Normalizing names...")
df['clean_name'] = df['original_name'].astype(str).apply(
    lambda x: re.sub(r'\s*\(.*?\)', '', x).strip().lower()
)

target_raw = "ZOOTOPIA 2 (3D) (HINDI)"
target_clean = re.sub(r'\s*\(.*?\)', '', target_raw).strip().lower()
print(f"Target Clean Name: '{target_clean}'")

# Filter
subset = df[df['clean_name'] == target_clean]
total_tickets = subset['sold_tickets'].sum()
unique_cinemas = subset['cinema_id'].nunique()

print(f"Total Tickets for '{target_clean}' (All Variants): {total_tickets}")
print(f"Unique Cinemas: {unique_cinemas}")

if unique_cinemas > 70:
    print("SUCCESS: Found ~80 cinemas as expected.")
else:
    print("WARNING: Still finding low cinema count.")
