import pandas as pd
from sqlalchemy import create_engine, inspect, text

# ==========================================
# CONFIGURATION
# ==========================================
DB_USER = 'root'
DB_PASSWORD = 'Yashsali@*2005'  # <--- PUT YOUR PASSWORD HERE
DB_HOST = 'localhost'
DB_NAME = 'movie_analytics'

def inspect_data():
    print("⏳ Connecting to database...")
    
    # Create connection string
    # We use pymysql or mysql-connector. Ensure you install the library below.
    connection_str = f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    
    try:
        engine = create_engine(connection_str)
        inspector = inspect(engine)
        
        # 1. Get all table names
        tables = inspector.get_table_names()
        print(f"✅ Connected! Found {len(tables)} tables: {tables}")
        print("="*60)

        # 2. Loop through tables to find the "Gold" (Columns)
        for table in tables:
            print(f"\n📂 TABLE: {table.upper()}")
            
            # Get columns and types
            columns = inspector.get_columns(table)
            col_names = [col['name'] for col in columns]
            print(f"   Columns: {col_names}")
            
            # Get Row Count (How much data do we have?)
            with engine.connect() as conn:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                print(f"   Total Rows: {count}")
                
                # Preview Data (First 3 rows)
                print(f"   Sample Data:")
                df = pd.read_sql(f"SELECT * FROM {table} LIMIT 3", conn)
                print(df.to_string(index=False))
                
            print("-" * 60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("Tip: Did you update the password in the script?")

if __name__ == "__main__":
    inspect_data()