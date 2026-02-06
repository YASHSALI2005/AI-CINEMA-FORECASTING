
def inspect():
    with open("backup_20260205.sql", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "INSERT INTO `m_cinema`" in line:
                print(line[:500]) # Print first 500 chars of each insert line
                # Also looks for 863 inside
                if "863" in line:
                    idx = line.find("863")
                    print(f"CONTEXT: {line[max(0, idx-50):idx+100]}")

if __name__ == "__main__":
    inspect()
