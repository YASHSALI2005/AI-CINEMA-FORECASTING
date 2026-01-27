import pandas as pd

# Load the file used for training
df = pd.read_csv("final_training_data_v3.csv")

# Show the Top 5 Cinemas by volume (The one with most rows is likely yours)
print("📊 Top Cinemas in Training Data:")
print(df['cinema_id'].value_counts().head(5))

print("\n-----------------------------------")
print("ℹ️ Try changing 'cinema_id = 1049' in your test script")
print("   to the #1 ID shown above (e.g., 1, 2, or 5).")
print("-----------------------------------")