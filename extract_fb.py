import re
import pandas as pd

file_path = 'cinepolis_schedule_23_01_2026.sql'

def parse_sql():
    print("Reading file...")
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    print("File read. Searching for m_item inserts...")
    
    # Matches: INSERT INTO `m_item` VALUES (...);
    # Dump files often group inserts: INSERT INTO `t` VALUES (1, 'a'), (2, 'b');
    # We need to handle that.
    
    # 1. Find the chunk of text containing m_item inserts
    # Look for start of info
    start_pattern = r"INSERT INTO `m_item` VALUES"
    start_indices = [m.start() for m in re.finditer(start_pattern, content)]
    
    if not start_indices:
        print("No INSERT INTO `m_item` found.")
        return

    parsed_data = []
    
    print(f"Found {len(start_indices)} INSERT blocks. Parsing...")
    
    for start in start_indices:
        # Find the end of this statement (;)
        end = content.find(";", start)
        block = content[start:end]
        
        # Remove prefix
        values_part = block.replace("INSERT INTO `m_item` VALUES ", "")
        
        # Split by "), (" to separate rows
        # This is a bit dirty but effective for dumps
        # (1, '...'), (2, '...')
        rows = values_part.split("),(")
        
        for row in rows:
            # Clean brackets
            row = row.replace("(", "").replace(")", "")
            
            # Split by comma
            # Handle quoted commas? SQL dump usuall escapes ' as \'. 
            # Simple split might fail on 'Popcorn, Salted'
            # Let's try to regex split or just simple split and see
            
            parts = row.split(",")
            # Clean whitespace and quotes
            parts = [p.strip().strip("'") for p in parts]
            
            # Based on screenshot:
            # 0: Id
            # 1: Status
            # 2: Description (Name)
            # 3: ShortDesc
            # 4: ClassCode
            # 5: Price
            
            if len(parts) >= 6:
                try:
                    # Index 2 might contain commas, shifting the price index.
                    # Price is usually numeric. Let's look for the first numeric after index 4?
                    # Or just assume simple names for now.
                    name = parts[2]
                    price = parts[5]
                    
                    # Verify price is numeric-ish
                    if price.replace('.','',1).isdigit():
                         parsed_data.append({"name": name, "price": price})
                except:
                    pass

    df = pd.DataFrame(parsed_data)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Filter
    popcorn = df[df['name'].str.contains('POPCORN|CORN|TUB', case=False, na=False)]
    coke = df[df['name'].str.contains('COKE|PEPSI|COLA|SPRITE|DRINK|FANT', case=False, na=False)]
    
    print(f"\nExtracted {len(df)} total items.")
    
    print("\n--- POPCORN EXAMPLES ---")
    print(popcorn[['name', 'price']].head(20))
    if not popcorn.empty:
        print(f"Avg Price: {popcorn['price'].mean()}")
    
    print("\n--- BEVERAGE EXAMPLES ---")
    print(coke[['name', 'price']].head(20))
    if not coke.empty:
        print(f"Avg Price: {coke['price'].mean()}")

if __name__ == "__main__":
    parse_sql()
