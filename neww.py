import pandas as pd; df = pd.read_csv('final_training_data_v3.csv');
df['show_time'] = pd.to_datetime(df['show_time']); 
mask = (df['cinema_id']==890) & (df['original_name'].str.contains('MINECRAFT', case=False)) & (df['show_time'].dt.date == pd.to_datetime('2025-04-06').date()); 
print(df[mask][['original_name', 'show_time', 'sold_tickets', 'popularity']].head())