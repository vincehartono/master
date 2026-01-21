from pybaseball import statcast, cache
from datetime import datetime, timedelta
import pandas as pd
import os

# Enable caching
cache.enable()

# Output file
output_path = r"C:\Users\Vince\master\Betting\Simulation_MLB\ML\mlb_atbat_data_2023_to_now.parquet"

# Start and end dates
if os.path.exists(output_path):
    existing_df = pd.read_parquet(output_path)
    last_date = pd.to_datetime(existing_df['game_date']).max()
    start_date = last_date + timedelta(days=1)
    print(f"Appending new data from {start_date.date()} onward...")
else:
    existing_df = pd.DataFrame()
    start_date = datetime(2023, 1, 1)
    print(f"No existing data. Starting from {start_date.date()}...")

end_date = datetime.today()

# Monthly batching
current = start_date
all_new_data = []

while current < end_date:
    next_month = (current.replace(day=1) + timedelta(days=32)).replace(day=1)
    batch_start = current.strftime('%Y-%m-%d')
    batch_end = min(next_month - timedelta(days=1), end_date).strftime('%Y-%m-%d')

    print(f"📅 Fetching {batch_start} to {batch_end}...")

    try:
        df = statcast(start_dt=batch_start, end_dt=batch_end)
        df = df[df['events'].notna()]
        atbats = df.drop_duplicates(subset=['game_pk', 'at_bat_number'])

        print(f"  ✅ {len(atbats)} at-bats found.")
        all_new_data.append(atbats)

    except Exception as e:
        print(f"  ❌ Failed for {batch_start} to {batch_end}: {e}")

    current = next_month

# Combine and save
if all_new_data:
    combined_new = pd.concat(all_new_data, ignore_index=True)
    full_df = pd.concat([existing_df, combined_new], ignore_index=True).drop_duplicates(subset=['game_pk', 'at_bat_number'])
    full_df.to_parquet(output_path, index=False)
    print(f"\n✅ Updated Parquet with {len(combined_new)} new at-bats. Total: {len(full_df)}")
else:
    print("⚠️ No new valid data to add.")
