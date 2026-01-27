import csv
import random

filename = "final_training_data_v3.csv"

print(f"Selecting a random 2024 record from {filename}...")

rows_2024 = []
with open(filename, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if "2024-01-15" in row['show_time']: # Mid-Jan 2024 for a stable schedule
            rows_2024.append(row)
            if len(rows_2024) > 100: break

if rows_2024:
    selected = random.choice(rows_2024)
    print("\n--- 🎲 Random Selected Record ---")
    for k, v in selected.items():
        print(f"{k}: {v}")
else:
    print("No 2024 records found.")
