import pandas as pd
import numpy as np

# Load the data
file_path = r'd:\Forecasting\data.csv.xlsx'
print(f"Loading data from: {file_path}")
df = pd.read_excel(file_path)

# 1. Infer Capacity for each Screen
# Assumption: The maximum 'Seats Available' observed for a screen is its Total Capacity.
# This assumes at least one show was completely empty or mostly empty.
screen_capacities = df.groupby('Screen_bytNum')['Session_intSeatsAvail'].max().to_dict()

print("\nInferred Screen Capacities:")
for screen, cap in screen_capacities.items():
    print(f"Screen {screen}: {cap} seats")

# 2. Calculate Sold, Revenue, and Occupancy
def calculate_metrics(row):
    screen = row['Screen_bytNum']
    capacity = screen_capacities.get(screen, 0)
    available = row['Session_intSeatsAvail']
    
    # Sanity check: Sold cannot be negative. 
    # If available > inferred capacity, our capacity inference is wrong (update it?) 
    # or data is weird. For now, max(0, ...)
    sold = max(0, capacity - available)
    
    occupancy = (sold / capacity) * 100 if capacity > 0 else 0
    revenue = sold * 100 # Fixed 100rs price
    
    return pd.Series([capacity, sold, occupancy, revenue], index=['Capacity', 'Sold', 'Occupancy%', 'Revenue'])

df[['Capacity', 'Sold', 'Occupancy%', 'Revenue']] = df.apply(calculate_metrics, axis=1)

# 3. Aggregation by Movie and Screen Category (Audi 1 vs Others)
df['Screen_Category'] = df['Screen_bytNum'].apply(lambda x: 'Audi 1' if x == 1 else 'Other Audis')

# We want to compare performance.
summary = df.groupby(['Film_strTitle', 'Screen_Category']).agg({
    'Sold': 'sum',
    'Capacity': 'sum',
    'Revenue': 'sum',
    'Session_lngSessionId': 'count' # Count of shows
}).rename(columns={'Session_lngSessionId': 'Shows'})

summary['Occupancy%'] = (summary['Sold'] / summary['Capacity']) * 100

print("\nSummary Validation:")
print(summary.head())

# 4. Pivot for Comparison
pivot_df = summary.unstack(level='Screen_Category')

# Flatten columns
pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]

# Ensure cols exist (handle cases where a movie only played in one category)
expected_cols = [
    'Revenue_Audi 1', 'Revenue_Other Audis', 
    'Occupancy%_Audi 1', 'Occupancy%_Other Audis',
    'Shows_Audi 1', 'Shows_Other Audis'
]
for col in expected_cols:
    if col not in pivot_df.columns:
        pivot_df[col] = 0

# 5. Calculate "Profit" / Gain metrics
# "profit if i placed that movie in audi 1 than other likewise"
# Interpretation: 
# Difference in Total Revenue is straightforward: Revenue_Audi 1 - Revenue_Other Audis.
# But this depends on number of shows.
# A better metric: Average Revenue Per Show.
pivot_df['Avg_Rev_Per_Show_Audi_1'] = pivot_df['Revenue_Audi 1'] / pivot_df['Shows_Audi 1'].replace(0, np.nan)
pivot_df['Avg_Rev_Per_Show_Other'] = pivot_df['Revenue_Other Audis'] / pivot_df['Shows_Other Audis'].replace(0, np.nan)

# "Value Gain" -> Difference in Avg Revenue per show (Base Price 100rs used in Revenue calc)
pivot_df['Value_Gain_Per_Show'] = pivot_df['Avg_Rev_Per_Show_Audi_1'] - pivot_df['Avg_Rev_Per_Show_Other']

# "% Gain"
pivot_df['Pct_Gain'] = (pivot_df['Value_Gain_Per_Show'] / pivot_df['Avg_Rev_Per_Show_Other']) * 100

# Cleanup for calculation (but keep NaNs for logic if needed, here we fill 0 for safety in math)
pivot_df = pivot_df.fillna(0) 

# 6. Textual Performance Analysis (Similar to schedule.py)
def get_performance_text(row):
    val_gain = row['Value_Gain_Per_Show']
    pct_gain = row['Pct_Gain']
    avg_other = row['Avg_Rev_Per_Show_Other']
    
    # Case: No shows in Other Audis (Value Gain is essentially the full Audi 1 revenue)
    if avg_other <= 0:
        if row['Avg_Rev_Per_Show_Audi_1'] > 0:
            return f"Exclusive to Audi 1 (Avg {row['Avg_Rev_Per_Show_Audi_1']:.0f} Rs/show)"
        else:
            return "No data/revenue in either category."

    # Logic from schedule.py adapted for past performance analysis
    explanation = ""
    if val_gain > 0:
        if pct_gain > 75:
            explanation = f"Drastically outperformed Other Audis"
        elif pct_gain > 30:
            explanation = f"Significantly higher footfall than Other Audis"
        else:
            explanation = f"Edged out Other Audis"
        
        explanation += f" by +{val_gain:.0f} Rs/show (+{pct_gain:.1f}%)"
        
    elif val_gain < 0:
        # Negative gain means Audi 1 performed worse, or Other Audis performed better
        explanation = f"Underperformed vs Other Audis by {val_gain:.0f} Rs/show ({pct_gain:.1f}%)"
    else:
        explanation = "Performance matched Other Audis exactly."
        
    return explanation

pivot_df['Performance_Analysis'] = pivot_df.apply(get_performance_text, axis=1)

# Rounding
pivot_df = pivot_df.round(2)

# Save
output_path = r'd:\Forecasting\Cinema_Analysis_100rs.xlsx'
pivot_df.to_excel(output_path)
print(f"\nAnalysis saved to: {output_path}")
