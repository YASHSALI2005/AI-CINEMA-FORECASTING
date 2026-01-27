import pandas as pd

# Load history
print("Loading history...")
history_cols = ['original_name', 'sold_tickets']
df = pd.read_csv("final_training_data_v3.csv", usecols=history_cols)
print("History loaded.")

# Check for "Zootopia"
print("\nUnique names containing 'Zootopia':")
matches = df[df['original_name'].astype(str).str.contains("Zootopia", case=False, na=False)]['original_name'].unique()
for m in matches:
    print(f"- '{m}'")

# Check for "ZOOTOPIA 2 (3D) (HINDI)" specifically
target = "ZOOTOPIA 2 (3D) (HINDI)"
print(f"\nExact match for '{target}':")
count = df[df['original_name'] == target].shape[0]
print(f"Row count: {count}")

# Check sold tickets for variations
for m in matches:
    subset = df[df['original_name'] == m]
    total = subset['sold_tickets'].sum()
    cinemas = subset['cinemas_id'].nunique() if 'cinemas_id' in subset.columns else 'N/A' 
    # Wait, column is cinema_id
    cinemas = df[df['original_name'] == m]['cinema_id'].nunique() if 'cinema_id' in df.columns else 'Unknown'
    if cinemas == 'Unknown':
         # Re-read with cinema_id
         df_c = pd.read_csv("final_training_data_v3.csv", usecols=['original_name', 'cinema_id'])
         cinemas = df_c[df_c['original_name'] == m]['cinema_id'].nunique()
         
    print(f"Movie: '{m}' | Tickets: {total} | Cinemas: {cinemas}")
