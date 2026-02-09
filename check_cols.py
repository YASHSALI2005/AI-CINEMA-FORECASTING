import pandas as pd
df = pd.read_csv("final_training_data_v4.csv", nrows=5)
for col in df.columns:
    print(col)
