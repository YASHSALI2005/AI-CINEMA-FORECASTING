import pandas as pd
import sys

def check_sales():
    print("Loading sales history from final_training_data_v3.csv...")
    try:
        # Load necessary columns: Now including cinema_id and show_time
        df = pd.read_csv("final_training_data_v3.csv", usecols=['original_name', 'sold_tickets', 'cinema_id', 'show_time'])
    except FileNotFoundError:
        print("❌ Error: 'final_training_data_v3.csv' not found in current directory.")
        return

    while True:
        user_input = input("\n🔎 Enter movie name to check (or 'exit'): ").strip()
        
        if user_input.lower() in ['exit', 'quit', 'q']:
            print("Goodbye!")
            break
            
        if not user_input:
            continue
            
        # Filter by movie name (case-insensitive)
        # We escape the input to avoid regex errors with parentheses like (3D)
        matches = df[df['original_name'].astype(str).str.contains(pd.Series([user_input]).str.replace(r'([()\[\]{}?*+])', r'\\\1', regex=True)[0], case=False, na=False)]
        
        if matches.empty:
            print(f"⚠️  No movies found matching '{user_input}'")
        else:
            # Get unique movie names found
            unique_movies = matches['original_name'].unique()
            
            for movie in unique_movies:
                print(f"\n🎬 MOVIE: {movie}")
                print("=" * 60)
                
                # Filter for this specific movie and sort by date
                movie_data = matches[matches['original_name'] == movie].sort_values('show_time')
                
                # Filter only shows with sales > 0 to see "sold" tickets? 
                # User asked "how many ticket sold", usually implies seeing the non-zero ones or all.
                # Let's show all valid entries but highlight high sales.
                
                if movie_data.empty:
                    print("   No show records found.")
                    continue
                
                total_for_movie = movie_data['sold_tickets'].sum()
                num_shows = len(movie_data)
                
                print(f"   TOTAL TICKETS SOLD: {total_for_movie}")
                print(f"   TOTAL SHOWS: {num_shows}")
                
                # Save to CSV to avoid terminal scrolling issues
                filename = "sales_report.csv"
                # Select and rename columns for clarity
                export_df = movie_data[['show_time', 'cinema_id', 'sold_tickets']].copy()
                export_df.columns = ['Show Time', 'Cinema ID', 'Tickets Sold']
                export_df.to_csv(filename, index=False)
                
                print(f"\n   ✅ Detailed report with {num_shows} rows saved to: {filename}")
                print("   (Open this file to see all Cinema IDs and Time Slots)")
                print("   " + "-" * 45)
                
                # Optional: still show top 5 rows
                print("   Latest 5 Shows:")
                print(export_df.tail(5).to_string(index=False))
                print("")

if __name__ == "__main__":
    check_sales()
