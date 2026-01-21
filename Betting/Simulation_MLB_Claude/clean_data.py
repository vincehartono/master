import pandas as pd

# File path
file_path = r"Betting/Simulation_MLB_ML_not_working/ML/mlb_atbat_data_2023_to_now.parquet"

# Desired columns
cols = [
    "player_name", "batter", "pitcher", "events", "des", "p_throws",
    "home_team", "away_team", "balls", "strikes", "at_bat_number",
    "pitch_number", "pitch_name", "home_score", "away_score", "bat_score"
]

# Load DataFrame
df = pd.read_parquet(file_path)

# Filter to desired columns (that exist)
existing_cols = [col for col in cols if col in df.columns]
df = df[existing_cols]

# Fill missing pitch_name
if "pitch_name" in df.columns:
    df["pitch_name"] = df["pitch_name"].fillna("Unknown")

# Check for any remaining missing data
missing_data = df.isnull().sum()
missing_data = missing_data[missing_data > 0]

if not missing_data.empty:
    print("⚠️ Columns with remaining missing data:")
    print(missing_data)
else:
    print("✅ No missing data in selected columns.")

# Preview data
print("\n🔍 Preview:")
print(df.head())

# Save cleaned data
output_path = r"Betting/Simulation_MLB_ML_not_working/ML/mlb_atbat_data_filtered.parquet"
df.to_parquet(output_path, index=False)
print(f"\n💾 Filtered data saved to: {output_path}")
