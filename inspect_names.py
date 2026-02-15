import pandas as pd
import json

CSV_FILE = "final_training_data_from_dump.csv"
JSON_FILE = "bh_box_office_data.json"

def main():
    print(f"Loading {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, usecols=['original_name'])
        unique_csv_names = df['original_name'].dropna().unique()
        print(f"Found {len(unique_csv_names)} unique names in CSV.")
        print("\n--- SAMPLE CSV NAMES (First 20) ---")
        for name in unique_csv_names[:20]:
            print(f"  '{name}'")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    print(f"\nLoading {JSON_FILE}...")
    try:
        with open(JSON_FILE, 'r', encoding='utf-8') as f:
            bh_data = json.load(f)
        
        bh_names = [entry.get('original_name') for entry in bh_data if entry.get('original_name')]
        print(f"Found {len(bh_names)} unique names in BH JSON.")
        print("\n--- SAMPLE BH NAMES (First 20) ---")
        for name in bh_names[:20]:
            print(f"  '{name}'")
            
        print("\n--- CHECKING SPECIFIC MISMATCHES ---")
        # Check specific movies details
        targets = ["jawan"]
        for t in targets:
            print(f"\nScanning for '{t}'...")
            found = False
            for entry in bh_data:
                name = entry.get('original_name', '').lower()
                if t in name:
                    print(f"--- MATCH FOUND: {name} ---")
                    print("Summary:", json.dumps(entry.get('summary'), indent=4))
                    days = entry.get('daily', [])
                    print(f"Daily entries: {len(days)}")
                    if days:
                         print("First Day:", days[0])
                    found = True
                    break
            if not found:
                print(f"  Not found in JSON.")

    except Exception as e:
        print(f"Error reading JSON: {e}")

if __name__ == "__main__":
    main()
