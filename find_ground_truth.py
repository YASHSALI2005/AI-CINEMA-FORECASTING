import csv

filename = "final_training_data_v3.csv"
target_cinema = "890"
target_date = "2023-12-02"
target_movie = "ANIMAL"

print(f"Searching {filename} for {target_movie} at Cinema {target_cinema} on {target_date}...")

matches = []
with open(filename, 'r', encoding='utf-8') as f:
    next(f) # skip header
    for line in f:
        if f",{target_cinema}," in line and target_date in line and target_movie in line:
            matches.append(line.strip())

if matches:
    print(f"Found {len(matches)} matches:")
    for m in matches:
        print(m)
else:
    print("No exact matches found.")
    
    # Check if date exists at all
    print(f"Checking if {target_date} exists in file...")
    found_date = False
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            if target_date in line:
                found_date = True
                break
    if found_date:
        print(f"Date {target_date} EXISTS in file.")
    else:
        print(f"Date {target_date} does NOT exist in file.")
