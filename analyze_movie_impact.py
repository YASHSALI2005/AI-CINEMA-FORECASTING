import pandas as pd
import numpy as np
import os
import re

# =============================================================================
#  CONFIG
# =============================================================================
DATA_PATH = "final_training_data_v4.csv"
CAPACITY_PER_SCREEN = 300
OUTPUT_FILE = "movie_impact_analysis.txt"

def load_data():
    if not os.path.isfile(DATA_PATH):
        print(f"[X] {DATA_PATH} not found.")
        return None
    
    print(f"[*] Loading {DATA_PATH}...")
    # Load necessary columns to save memory
    usecols = ["movie_name", "cinema_id", "show_time", "sold_tickets"]
    # Check if 'original_name' exists in the first few rows
    preview = pd.read_csv(DATA_PATH, nrows=5)
    if "original_name" in preview.columns:
        usecols.append("original_name")
        
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    
    # Preprocessing
    if "original_name" not in df.columns:
        df["original_name"] = df["movie_name"]  # Fallback
        
    df["show_time"] = pd.to_datetime(df["show_time"], errors='coerce')
    df.dropna(subset=["show_time", "sold_tickets"], inplace=True)
    
    df["date"] = df["show_time"].dt.date
    df["hour"] = df["show_time"].dt.hour
    
    # Normalize movie names
    df["clean_name"] = df["original_name"].astype(str).str.lower().str.strip()
    # Remove strings like (Hindi), (3D), etc for better grouping if needed, 
    # but for specific movie analysis, we might want to keep them or just use careful filtering.
    # For now, let's keep it simple.
    
    return df

def get_top_movies(df, n=5):
    """Get top N movies by total ticket sales."""
    grouped = df.groupby("clean_name")["sold_tickets"].sum().sort_values(ascending=False)
    return grouped.head(n).index.tolist()

def analyze_movie(df, movie_name_cleaned):
    print(f"\n--- Analyzing Movie: {movie_name_cleaned} ---")
    
    # Filter for the movie
    movie_df = df[df["clean_name"] == movie_name_cleaned].copy()
    
    if movie_df.empty:
        print("No data found for this movie.")
        return

    # --- 1. Cinema ID Impact (Covariance/Correlation) ---
    print("\n[1] Cinema ID Correlation Analysis")
    
    # Pivot: Index=(Date, Hour), Columns=Cinema_ID, Values=Sold_Tickets
    # We round hour to nearest major slot or just use raw hour? 
    # existing compare_cinemas uses broad slots. Let's stick to raw hours for correlation to see exact match.
    # Actually, let's group by Date-Hour.
    
    pivot_df = movie_df.pivot_table(
        index=["date", "hour"], 
        columns="cinema_id", 
        values="sold_tickets", 
        aggfunc="sum"
    )
    
    # Filter cinemas with enough data points (e.g., at least 10 shows)
    min_shows = 10
    mask = pivot_df.count() >= min_shows
    pivot_filtered = pivot_df.loc[:, mask]
    
    n_cinemas = pivot_filtered.shape[1]
    print(f"Cinemas with >{min_shows} shows: {n_cinemas}")
    
    if n_cinemas > 1:
        corr_matrix = pivot_filtered.corr()
        avg_corr = corr_matrix.mean().mean()
        print(f"Average Correlation between Cinemas: {avg_corr:.4f}")
        
        # High avg correlation -> Cinemas move together (Movie is strong everywhere or weak everywhere)
        # Low avg correlation -> Local factors matter more
        
        # Let's show a snippet of the corr matrix
        print("\nCorrelation Matrix Snippet (Top 5 cinemas):")
        print(corr_matrix.iloc[:5, :5].round(2))
    else:
        print("Not enough cinemas for correlation analysis.")

    # --- 2. Time Slot Impact ---
    print("\n[2] Time Slot Analysis")
    # Bin hours
    bins = [0, 12, 16, 20, 24]
    labels = ["Morning (0-12)", "Afternoon (12-16)", "Evening (16-20)", "Night (20-24)"]
    movie_df["time_slot"] = pd.cut(movie_df["hour"], bins=bins, labels=labels, right=False)
    
    slot_stats = movie_df.groupby("time_slot", observed=True)["sold_tickets"].agg(["mean", "count", "std"])
    print(slot_stats.round(2))

    # --- 3. Variance Analysis (Location Rule Necessity) ---
    print("\n[3] Variance Analysis (Location Rule Necessity)")
    # For each time slot, how much does occupancy vary ACROSS cinemas?
    
    # Group by Slot, then calculate stats across Cinemas
    # We first aggregate by Slot+Cinema (Average occupancy for that cinema in that slot)
    cinema_slot_avg = movie_df.groupby(["time_slot", "cinema_id"], observed=True)["sold_tickets"].mean().reset_index()
    
    # Now, for each slot, calculate the Coefficient of Variation (CV) of these averages
    # CV = (Std Dev / Mean) * 100
    
    print(f"{'Time Slot':<20} | {'Mean Sales':<10} | {'Std Dev':<10} | {'CV (%)':<10} | {'Interpretation'}")
    print("-" * 80)
    
    for slot in labels:
        slot_data = cinema_slot_avg[cinema_slot_avg["time_slot"] == slot]
        if slot_data.empty:
            continue
            
        mean_val = slot_data["sold_tickets"].mean()
        std_val = slot_data["sold_tickets"].std()
        
        if mean_val == 0:
            cv = 0
        else:
            cv = (std_val / mean_val) * 100
            
        interpretation = "Consistent"
        if cv > 30: interpretation = "Moderate Var"
        if cv > 60: interpretation = "High Var (Loc Rule Needed)"
        
        print(f"{slot:<20} | {mean_val:>10.2f} | {std_val:>10.2f} | {cv:>10.1f} | {interpretation}")

def run():
    df = load_data()
    if df is None: return

    top_movies = get_top_movies(df, n=3)
    print(f"Top 3 Movies by volume: {top_movies}")
    
    for movie in top_movies:
        analyze_movie(df, movie)

if __name__ == "__main__":
    run()
