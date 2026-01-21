import pandas as pd
import numpy as np

# Load the data
file_path = "player_scores.csv"
per_path = "player_PER_scores.csv"
df = pd.read_csv(file_path)
df_per = pd.read_csv(per_path)

# Combine player.firstname and player.lastname if needed
df['player_name'] = df['player.firstname'] + ' ' + df['player.lastname']
df.drop(['player.firstname', 'player.lastname'], axis=1, inplace=True)

# Drop any unnecessary columns like 'comment' (if present)
df.drop(columns=['comment'], errors='ignore', inplace=True)

# Ensure 'game_date' is datetime
df['game_date'] = pd.to_datetime(df['game_date'])

# Drop rows with missing values in 'min' column
df.dropna(subset=['min'], inplace=True)

# Combine `df` and `df_per` on `player_name`
df_combined = pd.merge(df, df_per, on=['player_name', 'min'], how='inner')

# Group by player_name and calculate mean and std for 'min' and 'uPER' columns
player_stats = df_combined.groupby('player_name').agg({
    'min': ['mean', 'std', 'count'],
    'uPER': ['mean', 'std']
}).reset_index()

# Flatten the MultiIndex columns for better access
player_stats.columns = ['player_name', 'mean_min', 'std_min', 'count_min', 'mean_uPER', 'std_uPER']

# Filter out players with only one entry (std=0)
player_stats = player_stats[player_stats['count_min'] > 1]

# Initialize a list to hold the simulated results
simulated_data = []

# Loop through each player to simulate the 'min' and 'uPER' values
for index, row in player_stats.iterrows():
    player_name = row['player_name']
    mean_min = row['mean_min']
    std_min = row['std_min']
    mean_uPER = row['mean_uPER']
    std_uPER = row['std_uPER']

    # Get the team code for the player
    team_code = df[df['player_name'] == player_name]['team.code'].iloc[0]

    # Simulate 'min'
    if std_min == 0:
        simulated_min = [mean_min] * 1000
    else:
        simulated_min = np.random.normal(loc=mean_min, scale=std_min, size=1000).tolist()

    # Simulate 'uPER'
    if pd.isna(std_uPER) or std_uPER == 0:
        simulated_uPER = [mean_uPER] * 1000
    else:
        simulated_uPER = np.random.normal(loc=mean_uPER, scale=std_uPER, size=1000).tolist()

    simulated_data.append([player_name, team_code, simulated_min, simulated_uPER])

# Create a DataFrame for the simulated results
simulated_df = pd.DataFrame(simulated_data, columns=['player_name', 'team.code', 'simulated_min', 'simulated_uPER'])

# Calculate normalized_PER (min-max scaling on mean_uPER across all players)
uPER_values = np.concatenate(simulated_df['simulated_uPER'].values)
min_uPER, max_uPER = uPER_values.min(), uPER_values.max()
simulated_df['normalized_PER'] = simulated_df['simulated_uPER'].apply(
    lambda x: [(val - min_uPER) / (max_uPER - min_uPER) for val in x]
)

# Save the simulated results to a CSV file
simulated_df.to_csv('simulated_player_metrics.csv', index=False)

# Show the first few rows of the result for verification
print(simulated_df.head())
