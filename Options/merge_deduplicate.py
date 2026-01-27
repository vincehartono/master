"""
Merge two CSV files and remove duplicates - return DataFrame
"""

import pandas as pd

# Read both files
file1 = "Designated_Bene_Individual_XXX742_Transactions_20260126-233740.csv"
file2 = "Designated_Bene_Individual_XXX742_Transactions_20260127-161226.csv"

print(f"Loading {file1}...")
df1 = pd.read_csv(file1)
print(f"  Rows: {len(df1)}")

print(f"Loading {file2}...")
df2 = pd.read_csv(file2)
print(f"  Rows: {len(df2)}")

# Combine both DataFrames
print(f"\nCombining both files...")
df = pd.concat([df1, df2], ignore_index=True)
print(f"  Total rows: {len(df)}")

# Remove duplicates - keep first occurrence
print(f"\nRemoving duplicates...")
df = df.drop_duplicates(keep='first')
print(f"  Unique rows: {len(df)}")

print(f"\n{'='*60}")
print(f"MERGED DATAFRAME")
print(f"{'='*60}")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\n{'='*60}")
