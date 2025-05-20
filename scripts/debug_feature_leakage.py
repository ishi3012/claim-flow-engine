import pandas as pd

df = pd.read_csv("claimflowengine/data/processed_claims.csv")
df = df.drop(columns=["denied"])
y = pd.read_csv("claimflowengine/data/processed_claims.csv")["denied"]

# Check for 100% predictive rows
matches = (
    (df.drop_duplicates().assign(denied=y)).groupby(list(df.columns)).denied.nunique()
)
perfect_leaks = matches[matches == 1]

print(
    f"🔍 Found {len(perfect_leaks)} unique feature"
    + " patterns with only one class label"
)
