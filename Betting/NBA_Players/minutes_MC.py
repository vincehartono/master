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

# Group by player_name and calculate the mean and std for the 'min' column
player_stats = df.groupby('player_name')['min'].agg(['mean', 'std', 'count']).reset_index()

# Filter out players with only one entry (this results in std=0, no variation)
player_stats = player_stats[player_stats['count'] > 1]

# Initialize a list to hold the simulated results
simulated_data = []

# Loop through each player to simulate the 'min' values
for index, row in player_stats.iterrows():
    player_name = row['player_name']
    mean_min = row['mean']
    std_min = row['std']

    # Get the team code for the player (assuming one team per player)
    team_code = df[df['player_name'] == player_name]['team.code'].iloc[0]

    # Handle players with only one entry (std = 0)
    if std_min == 0:
        # If std is 0, simulate the mean repeated 1000 times (no variability)
        simulated_min = [mean_min] * 1000
    else:
        # Simulate 'min' 1000 times based on mean and std
        simulated_min = np.random.normal(loc=mean_min, scale=std_min, size=1000).tolist()

    # Append the results
    simulated_data.append([player_name, team_code, simulated_min])

# Create a DataFrame for the simulated results
simulated_df = pd.DataFrame(simulated_data, columns=['player_name', 'team.code', 'simulated_min'])

# Save the simulated results to a CSV file
simulated_df.to_csv('simulated_player_minutes.csv', index=False)

# Show the first few rows of the result for verification
print(simulated_df.head())
