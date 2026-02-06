import pandas as pd
import os

CSV_FILE = "final_training_data_v4.csv"
OUTPUT_FILE = "final_training_data_10k_v3.xlsx"
ROWS_TO_READ = 10000

def convert():
    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        return

    print(f"Reading first {ROWS_TO_READ} rows from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, nrows=ROWS_TO_READ)
        print(f"Read {len(df)} rows. Processing ticket categories...")

        # 1. Calculate Price Per Ticket
        df["price_per_ticket"] = df.apply(lambda row: row["revenue_x"] / row["sold_tickets"] if row["sold_tickets"] > 0 else 0, axis=1)
        df["price_per_ticket"] = df["price_per_ticket"].round(2)

        # 2. Define Category Logic based on relative rank in the same show
        def assign_categories(group):
            # Get unique prices for this show, sorted
            unique_prices = sorted(group["price_per_ticket"].unique())
            
            # Map price -> category
            price_map = {}
            n = len(unique_prices)
            
            if n == 1:
                price_map[unique_prices[0]] = "Standard"
            elif n == 2:
                price_map[unique_prices[0]] = "Standard"
                price_map[unique_prices[1]] = "Premium"
            elif n == 3:
                price_map[unique_prices[0]] = "Standard"
                price_map[unique_prices[1]] = "Executive"
                price_map[unique_prices[2]] = "Premium"
            else:
                # 4 or more: spread them out
                for i, p in enumerate(unique_prices):
                    if i == 0:
                        price_map[p] = "Standard"
                    elif i == n - 1:
                        price_map[p] = "Premium"
                    else:
                        price_map[p] = "Executive"
            
            # Apply map
            group["Likely Category"] = group["price_per_ticket"].map(price_map)
            return group

        # 3. Apply group-wise (might take a moment for 10k rows)
        # Grouping by cinema, time, movie to isolate a single "Show"
        print("Grouping and assigning categories...")
        # handling NaN if any
        df["movie_name"] = df["movie_name"].fillna("")
        df["cinema_id"] = df["cinema_id"].fillna(0)
        df["show_time"] = df["show_time"].fillna("")
        
        df = df.groupby(["cinema_id", "show_time", "movie_name"], group_keys=False).apply(assign_categories)
        
        # Rename for user
        df["Price Per Ticket"] = df["price_per_ticket"]
        
        # Drop temp/duplicate cols if needed, but keep originals for context
        # Just select column order for nice input
        cols = ["movie_name", "cinema_id", "show_time", "sold_tickets", "revenue_x", "Price Per Ticket", "Likely Category"]
        # Add others if present
        remaining = [c for c in df.columns if c not in cols and c != "price_per_ticket"]
        df = df[cols + remaining]

        print(f"Saving to {OUTPUT_FILE}...")
        
        # Save to Excel
        df.to_excel(OUTPUT_FILE, index=False)
        print(f"Successfully saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    convert()
