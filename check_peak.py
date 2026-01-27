import pandas as pd

# Load history
print("Loading history...")
df = pd.read_csv("final_training_data_v3.csv", usecols=['show_time', 'sold_tickets'])
df['show_time'] = pd.to_datetime(df['show_time'])

# Group by Hour and Sum
print("Grouping by hour...")
# Floor to hour
df['hour_slot'] = df['show_time'].dt.floor('h')

hourly_sales = df.groupby('hour_slot')['sold_tickets'].sum().sort_values(ascending=False)

print("Top 10 Busiest Hours in History (Network Wide):")
print(hourly_sales.head(10))

avg_peak = hourly_sales.head(50).mean()
print(f"\nAverage Peak Hour Sales: {avg_peak}")
