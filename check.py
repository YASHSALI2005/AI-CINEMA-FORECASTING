import pandas as pd

# Load your V3 data
df = pd.read_csv("final_training_data_v3.csv")

# Filter for JAWAN
jawan_data = df[df['movie_name'].str.contains("AVATAR : FIRE AND ASH (3D) (HINDI)", case=False)]

# Sort by show time to find Day 2
jawan_data['show_time'] = pd.to_datetime(jawan_data['show_time'])
jawan_data = jawan_data.sort_values('show_time')

# Show the first few rows (Day 1 and Day 2)
print(jawan_data[['show_time', 'movie_name', 'sold_tickets', 'movie_trend_7d', 'cinema_trend_7d']].head(10))