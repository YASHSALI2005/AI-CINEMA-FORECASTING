import pandas as pd
import numpy as np
from fpdf import FPDF
import datetime

# --- CONFIGURATION ---
DATE_RANGES = [
    (pd.to_datetime("2026-01-25"), pd.to_datetime("2026-02-05"), "Period 1: Jan 25 - Feb 5"),
    (pd.to_datetime("2026-02-06"), pd.to_datetime("2026-02-15"), "Period 2: Feb 6 - Feb 15")
]
CSV_FILE = "final_training_data_v4.csv"
OUTPUT_DIR = "."

def get_top_3_movies():
    """Identifies top 3 movies by show count in the combined date range."""
    # We'll read the file and filter for the full range
    start_full = DATE_RANGES[0][0]
    end_full = DATE_RANGES[1][1]
    
    movie_counts = {}
    
    print(f"Scanning for top movies between {start_full.date()} and {end_full.date()}...")
    
    try:
        # Optimization: Read only necessary columns first to find movies
        # 'show_time', 'original_name'
        # Check actual columns from your file. Assuming 'show_time', 'original_name' exist.
        for chunk in pd.read_csv(CSV_FILE, chunksize=100000, usecols=['show_time', 'original_name']):
            chunk['show_time'] = pd.to_datetime(chunk['show_time'], errors='coerce')
            mask = (chunk['show_time'] >= start_full) & (chunk['show_time'] <= end_full)
            relevant = chunk[mask]
            
            if not relevant.empty:
                counts = relevant['original_name'].value_counts()
                for name, count in counts.items():
                    movie_counts[name] = movie_counts.get(name, 0) + count
                    
    except ValueError as e:
        # Fallback if usecols fails (column names might differ slightly?)
        print(f"Warning: {e}. Reading all columns.")
        for chunk in pd.read_csv(CSV_FILE, chunksize=100000):
            chunk['show_time'] = pd.to_datetime(chunk['show_time'], errors='coerce')
            mask = (chunk['show_time'] >= start_full) & (chunk['show_time'] <= end_full)
            relevant = chunk[mask]
            if not relevant.empty:
                counts = relevant['original_name'].value_counts()
                for name, count in counts.items():
                    movie_counts[name] = movie_counts.get(name, 0) + count

    # Sort
    sorted_movies = sorted(movie_counts.items(), key=lambda x: x[1], reverse=True)
    top_3 = [m[0] for m in sorted_movies[:3]]
    print(f"Top 3 Movies Found: {top_3}")
    return top_3


def generate_report(movie_name, df_movie):
    """Generates a PDF report for a single movie."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Title
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt=f"Performance Report: {movie_name}", ln=True, align='C')
    pdf.ln(10)
    
    # Process each period
    for start_dt, end_dt, label in DATE_RANGES:
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, txt=label, ln=True, align='L')
        pdf.ln(5)
        
        # Filter Data
        mask = (df_movie['show_time'] >= start_dt) & (df_movie['show_time'] <= end_dt)
        period_data = df_movie[mask].copy()
        
        if period_data.empty:
            pdf.set_font("Arial", 'I', 12)
            pdf.cell(200, 10, txt="No shows found in this period.", ln=True)
            pdf.ln(5)
            continue
            
        # --- CALCULATIONS ---
        # Logic from app.py:
        # Pred Low = ((Raw * 0.55 / 300) * 100).clip(upper=100)
        # Pred High = (Pred Low + 25).clip(upper=100)
        # Actual % = ((Sold / 300) * 100).clip(upper=100)
        
        # Columns needed: 'sold_tickets', 'Raw Prediction' (Do we have Raw Prediction in CSV? likely not)
        # The user said "predicted sales in range %". 
        # If 'Raw Prediction' is NOT in CSV, we assume we need to generating it? 
        # OR maybe there is a 'predicted_sales' column?
        # Let's assume for now the CSV has what we need or we mock/calculate it if missing?
        # WAIT. The user prompt says "table of showtime,predicted sales in range %,actual sales in %".
        # 'final_training_data_v4.csv' usually has historical data. Does it have predictions?
        # If not, I'd need the MODEL to predict. BUT loading model is heavy.
        # Let's check CSV columns in next step headers. 
        # For now, I will assume columns exist or use placeholders if not found.
        # IF 'predicted_tickets' or similar exists, use it.
        # IF NOT, I might need to run the model? 
        # Actually, `final_training_data_v4.csv` implies it's training data (Actuals).
        # Does it contain predictions? Probably not.
        # However, `app.py` generates predictions on the fly.
        # Re-generating predictions for thousands of rows might be slow but necessary.
        # OR maybe the user just wants the Actuals and I should simulate predictions?
        # The user asked: "consisting table of showtime,predicted sales in range %,actual sales in %"
        # I will check if 'predicted_sales' column exists. If not, I will use a simple heuristic matching `app.py` logic 
        # OR (Critical) Load the model and predict.
        # Given "Analysis" mode task, maybe simulated/heuristic or model inference is expected.
        # I'll stick to check columns first. If missing, I'll assume I need to standard `Pred %` logic?
        # But `Raw Prediction` requires Model.
        # Let's assume for now we might need to load model. 
        # Actually, for the report to be meaningful, it probably needs real predictions.
        # I will attempt to load the model IF simple columns are missing.
        
        # For this script, I'll check columns dynamically.
        
        # Table Header
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(60, 10, "Showtime", 1)
        pdf.cell(65, 10, "Pred Sales Range %", 1)
        pdf.cell(65, 10, "Actual Sales %", 1)
        pdf.ln()
        
        pdf.set_font("Arial", size=10)
        
        # Sort by time
        period_data = period_data.sort_values('show_time')
        
        for _, row in period_data.iterrows():
            showtime_str = row['show_time'].strftime('%Y-%m-%d %H:%M')
            
            # Actuals
            sold = row.get('sold_tickets', 0)
            actual_pct = min((sold / 300) * 100, 100)
            
            # Prediction (Mock logic if missing, or use column)
            # Check if 'ticket_price' or 'capacity' is there?
            # Using app.py logic: Pred Low = Actual * some_variance? No, that's cheating.
            # Use 'sold_tickets' as proxy for "perfect prediction" if we lack model?
            # No, user wants "Predicted vs Actual".
            # Let's assumes we use the `app.py` logic `Raw Prediction` which comes from model.
            # Loading model for this script:
            # `model.load_model("xgb_cinema_model_v4.json")`
            # I'll add model loading to this script to be safe.
            
            # Placeholder for now until we confirm we can run model.
            # I'll output values.
            
            # Temporary logic: 
            # If we don't have predictions, I'll mark as "N/A" or use a dummy.
            # But the user likely wants real values.
            # I will try to Calculate using the columns available.
            
            pred_range_str = "N/A" # Fill in later
            
            # --- FILLER LOGIC TO BE REPLACED BY MODEL INFERENCE ---
            # Ideally we'd run the model here.
            # For now, I'll assume we can't run model 1000s times quickly without setup.
            # Use 'sold_tickets' +/- random? No.
            # I will leave this as TODO and update script after checking CSV columns.
            
            pdf.cell(60, 10, showtime_str, 1)
            pdf.cell(65, 10, pred_range_str, 1)
            pdf.cell(65, 10, f"{actual_pct:.1f}%", 1)
            pdf.ln()
            
        pdf.ln(10)
        
    filename = f"Report_{movie_name.replace(':', '').replace(' ', '_')}.pdf"
    pdf.output(filename)
    print(f"Generated: {filename}")

def main():
    # 1. Identify Movies
    top_movies = get_top_3_movies()
    if not top_movies:
        print("No movies found in range.")
        return

    # 2. Extract Data & Generate Reports
    # We need to read the full rows for these movies now
    # Using 'chunksize' again to manage memory
    print("Extracting detailed data for top movies...")
    
    # Store dataframes
    movie_dfs = {m: [] for m in top_movies}
    
    # Data columns needed: show_time, original_name, sold_tickets, ... features for model?
    # To run model we need: ['budget', 'runtime', 'popularity', 'vote_average', 'day_of_week', ...]
    # Does CSV have these? 'movie_features_safe.csv' has metadata. CSV has history.
    # Joining them is needed for prediction.
    # This is getting complex. 
    # LET'S CHECK IF CSV HAS PREDICTIONS. 
    
    for chunk in pd.read_csv(CSV_FILE, chunksize=100000):
        chunk['show_time'] = pd.to_datetime(chunk['show_time'], errors='coerce')
        mask_movies = chunk['original_name'].isin(top_movies)
        
        # Date Filter (Combined Range)
        start_full = DATE_RANGES[0][0]
        end_full = DATE_RANGES[1][1]
        mask_date = (chunk['show_time'] >= start_full) & (chunk['show_time'] <= end_full)
        
        filtered = chunk[mask_movies & mask_date]
        
        if not filtered.empty:
            for m in top_movies:
                movie_dfs[m].append(filtered[filtered['original_name'] == m])
    
    # Generate PDFs
    for m in top_movies:
        if movie_dfs[m]:
            full_df = pd.concat(movie_dfs[m])
            generate_report(m, full_df)
        else:
            print(f"No data for {m}")

if __name__ == "__main__":
    main()
