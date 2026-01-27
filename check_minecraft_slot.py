import pandas as pd
import re

print("Loading history...")
df = pd.read_csv("final_training_data_v3.csv", usecols=['original_name', 'show_time', 'cinema_id', 'sold_tickets'])
df['show_time'] = pd.to_datetime(df['show_time'])

# Normalize
df['clean_name'] = df['original_name'].astype(str).apply(
    lambda x: re.sub(r'\s*\(.*?\)', '', x).strip().lower()
)

target_name = "a minecraft movie"
target_date = "2025-04-07"
target_hour = 22

print(f"Checking '{target_name}' on {target_date} around {target_hour}:00...")

# Filter Name
subset = df[df['clean_name'] == target_name]

# Filter Date
subset = subset[subset['show_time'].dt.date.astype(str) == target_date]

# Filter Time Window (+/- 60 mins)
target_ts = pd.Timestamp(f"{target_date} {target_hour}:00:00")
start = target_ts - pd.Timedelta(minutes=60)
end = target_ts + pd.Timedelta(minutes=60)

time_subset = subset[(subset['show_time'] >= start) & (subset['show_time'] <= end)]

print("\n--- RESULTS ---")
unique_cinemas = time_subset['cinema_id'].nunique()
total_tickets = time_subset['sold_tickets'].sum()

print(f"Cinemas Playing at this time: {unique_cinemas}")
print(f"Total Tickets Sold: {total_tickets}")
print(f"Avg Tickets per Cinema: {total_tickets / unique_cinemas if unique_cinemas > 0 else 0}")

# Also check TOTAL cinemas that played it that day at ANY time
day_cinemas = subset['cinema_id'].nunique()
print(f"\nTotal Cinemas playing it ON THAT DAY (Any time): {day_cinemas}")
