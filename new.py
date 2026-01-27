import pandas as pd
from sqlalchemy import create_engine
from urllib.parse import quote_plus  # <--- This fixes the password issue

# DATABASE SETUP
db_user = 'root'
db_password ='Yashsali@*2005'  # <--- Type your password here safely
db_host = 'localhost'
db_name = 'movie_analytics'

def check_descriptions():
    # Safely encode the password to handle special characters like '*' or '@'
    encoded_password = quote_plus(db_password)
    
    # Create the engine with the encoded password
    connection_str = f"mysql+mysqlconnector://{db_user}:{encoded_password}@{db_host}/{db_name}"
    engine = create_engine(connection_str)
    
    print("🕵️ Checking Movie Descriptions for 'Hidden DNA'...")
    
    # Fetch 10 random movies with their descriptions
    query = "SELECT movie_title, movie_description FROM m_movie LIMIT 10"
    
    try:
        df = pd.read_sql(query, engine)
        
        if df.empty:
            print("⚠️ The table is empty!")
        else:
            for index, row in df.iterrows():
                print(f"\n🎬 Movie: {row['movie_title']}")
                print(f"📝 Desc:  {str(row['movie_description'])[:150]}...") 
                print("-" * 50)
            
    except Exception as e:
        print("\n❌ Connection Error. Double check your password!")
        print(f"Details: {e}")

if __name__ == "__main__":
    check_descriptions()