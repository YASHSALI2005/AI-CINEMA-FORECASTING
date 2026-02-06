
def check_session_schema_and_data():
    sql_file = "backup_20260205.sql"
    print(f"Checking m_session schema and sample data...")
    
    try:
        with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
            in_table = False
            for line in f:
                if "CREATE TABLE `m_session`" in line:
                    in_table = True
                    print(line.strip())
                    continue
                
                if in_table:
                    print(line.strip())
                    if ";" in line:
                        in_table = False
                        break
                        
        print("\n--- SAMPLE INSERT ---")
        with open(sql_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if "INSERT INTO `m_session`" in line:
                    print(line[:1000])
                    break
                        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_session_schema_and_data()
