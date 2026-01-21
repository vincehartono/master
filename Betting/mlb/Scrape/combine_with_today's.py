import pandas as pd

# === Load Data ===
picks_df = pd.read_csv(r"C:\Users\Vince\master\Betting\mlb\Scrape\combined_picks_with_matchups.csv")
schedule_df = pd.read_parquet(r"C:\Users\Vince\master\Betting\Simulation_MLB\todays_schedule.parquet")

# === Normalize Team Names ===
schedule_df['Home Team'] = schedule_df['Home'].str.strip()
schedule_df['Away Team'] = schedule_df['Away'].str.strip()

# === Count Expert Picks ===
all_picks = pd.Series(picks_df.values.ravel()).dropna().str.strip()
pick_counts = all_picks.value_counts().rename_axis('team').reset_index(name='pick_count')

# === Merge Pick Counts with Schedule ===
schedule_df = schedule_df.merge(pick_counts, left_on='Home Team', right_on='team', how='left')
schedule_df = schedule_df.rename(columns={'pick_count': 'home_pick_count'}).drop(columns=['team'])

schedule_df = schedule_df.merge(pick_counts, left_on='Away Team', right_on='team', how='left')
schedule_df = schedule_df.rename(columns={'pick_count': 'away_pick_count'}).drop(columns=['team'])

schedule_df['home_pick_count'] = schedule_df['home_pick_count'].fillna(0).astype(int)
schedule_df['away_pick_count'] = schedule_df['away_pick_count'].fillna(0).astype(int)

# === Determine Most Favored Team ===
schedule_df['pick_margin'] = schedule_df['home_pick_count'] - schedule_df['away_pick_count']
schedule_df['most_favored_team'] = schedule_df.apply(
    lambda row: row['Home Team'] if row['pick_margin'] > 0
    else (row['Away Team'] if row['pick_margin'] < 0 else 'Even'),
    axis=1
)

# === Print Top 3–4 Most Favorable Matchups ===
top_matchups = schedule_df.reindex(schedule_df['pick_margin'].abs().sort_values(ascending=False).index).head(4)

print("Top 3–4 most favorable matchups based on expert picks:")
for _, row in top_matchups.iterrows():
    print(
        f"{row['Home Team']} ({row['home_pick_count']} picks) vs "
        f"{row['Away Team']} ({row['away_pick_count']} picks) | "
        f"Favor: {row['most_favored_team']} (Margin: {abs(row['pick_margin'])} votes)"
    )

# === Print Games With 0 Picks on Either Side ===
zero_pick_games = schedule_df[
    (schedule_df['home_pick_count'] == 0) | (schedule_df['away_pick_count'] == 0)
]

print("\nGames with 0 picks on either side:")
for _, row in zero_pick_games.iterrows():
    print(
        f"{row['Home Team']} ({row['home_pick_count']} picks) vs "
        f"{row['Away Team']} ({row['away_pick_count']} picks)"
    )

# === Save Combined Output ===
output_path = r"C:\Users\Vince\master\Betting\mlb\Scrape\combined_schedule_with_picks.csv"
schedule_df.to_csv(output_path, index=False)
print(f"\n✅ Saved combined schedule with picks to: {output_path}")
