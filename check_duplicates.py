
import pandas as pd
import os

def check_duplicates():
    path = "final_training_data_v3.csv"
    if not os.path.exists(path):
        print(f"File {path} not found.")
        return

    print("Loading data...")
    # Load necessary columns
    df = pd.read_csv(path, usecols=["movie_name", "cinema_id", "show_time", "sold_tickets", "revenue_x"])
    
    print(f"Total rows: {len(df)}")
    
    # Check for duplicates on keys
    keys = ["cinema_id", "show_time", "movie_name"]
    duplicates = df[df.duplicated(subset=keys, keep=False)]
    
    if duplicates.empty:
        print("No duplicates found based on keys:", keys)
    else:
        print(f"Found {len(duplicates)} duplicate rows.")
        print(duplicates.sort_values(by=keys).head(20))
        
        # Check if revenue is different
        diff_revenue = duplicates.groupby(keys)["revenue_x"].nunique()
        diff_revenue = diff_revenue[diff_revenue > 1]
        
        if not diff_revenue.empty:
            print("\nFound duplicates with DIFFERENT revenue:")
            print(diff_revenue.head())
            
            # Show example
            example_keys = diff_revenue.index[0]
            print("\nExample of different revenue:")
            mask = (
                (df["cinema_id"] == example_keys[0]) & 
                (df["show_time"] == example_keys[1]) & 
                (df["movie_name"] == example_keys[2])
            )
            print(df[mask])
        else:
            print("\nDuplicates have same revenue.")

if __name__ == "__main__":
    check_duplicates()
