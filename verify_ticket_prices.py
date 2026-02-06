
import pandas as pd
import os

def check_prices():
    path = "final_training_data_v3.csv"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print("Loading data...")
    # Load necessary columns
    df = pd.read_csv(path, usecols=["movie_name", "cinema_id", "show_time", "sold_tickets", "revenue_x"])
    
    # Calculate Price Per Ticket
    # Handle division by zero just in case
    df = df[df["sold_tickets"] > 0]
    df["price_per_ticket"] = df["revenue_x"] / df["sold_tickets"]
    
    # Find duplicates
    keys = ["cinema_id", "show_time", "movie_name"]
    duplicates = df[df.duplicated(subset=keys, keep=False)].copy()
    
    if duplicates.empty:
        print("No duplicates found.")
        return

    # Sort to see groups together
    duplicates = duplicates.sort_values(by=keys)
    
    # Pick top 5 groups to show
    print("\n--- Evidence of Different Ticket Prices (Ticket Categories) ---")
    
    # helper to print a group
    def print_group(group):
        first_row = group.iloc[0]
        print(f"\nMovie: {first_row['movie_name']}")
        print(f"Cinema: {first_row['cinema_id']} | Time: {first_row['show_time']}")
        print(f"{'Sold':<10} {'Revenue':<15} {'Price/Ticket':<15}")
        print("-" * 40)
        for _, row in group.iterrows():
            print(f"{row['sold_tickets']:<10} {row['revenue_x']:<15} {row['price_per_ticket']:.2f}")

    # Group by the keys and iterate
    grouped = duplicates.groupby(keys)
    
    count = 0
    for name, group in grouped:
        # Only show groups where there are DIFFERENT prices
        unique_prices = group["price_per_ticket"].round(0).nunique()
        if unique_prices > 1:
            print_group(group)
            count += 1
            if count >= 5: # Show 5 examples
                break
                
    if count == 0:
        print("All duplicates had the same price (strange?)")

if __name__ == "__main__":
    check_prices()
