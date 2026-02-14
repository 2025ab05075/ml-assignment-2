import pandas as pd

df = pd.read_csv("data/bank-full.csv", sep=";")

# take random 500 rows
test_df = df.sample(500, random_state=42)

test_df.to_csv("data/test.csv", index=False, sep=";")

print("✅ test.csv created in data folder")
