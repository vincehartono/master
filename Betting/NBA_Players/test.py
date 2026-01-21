import pandas as pd

# Load the CSV file into a DataFrame (without assuming the first row is the header)
df = pd.read_csv('odds_today.csv', header=None)

# Step 1: Select rows 1st, 3rd, 9th, 11th, etc.
selected_rows = df.iloc[[i for i in range(len(df)) if i % 6 == 0 or i % 6 == 2]].reset_index(drop=True)

# Step 2: Split into "odd" and "even" rows
odd_rows = selected_rows.iloc[::2].reset_index(drop=True)  # rows like 1st, 9th, 17th...
even_rows = selected_rows.iloc[1::2].reset_index(drop=True)  # rows like 3rd, 11th, 19th...

# Step 3: Clean "Over(" and ")" from even rows (removes "Over(" and ")")
for col in even_rows.columns:
    even_rows[col] = even_rows[col].astype(str)  # Ensure string type
    try:
        even_rows[col] = (
            even_rows[col]
            .str.replace(r"Over\(", "", regex=True)  # Correctly escape '('
            .str.replace(r"\)", "", regex=True)  # Correctly escape ')'
        )
    except Exception as e:
        print(f"Error cleaning column '{col}': {e}")
        print("Problematic data:", even_rows[col].head())

# Step 4: Move cleaned even rows to new columns
combined = odd_rows.copy()
for col in even_rows.columns:
    combined[f"{col}_even"] = even_rows[col]

# Step 5: Assign column names at the end
combined.columns = ['player_name', 'over/under']

# Display the resulting DataFrame
print("After cleaning and combining:")
print(combined.head())

# Step 6: Save the final combined DataFrame to CSV
combined.to_csv('combined_odds.csv', index=False)

print("CSV file 'combined_odds.csv' has been saved!")
