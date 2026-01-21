import pandas as pd

# Load the data
file_path = "player_scores.csv"
df = pd.read_csv(file_path)

# Combine player.firstname and player.lastname, then drop the original columns
df['player_name'] = df['player.firstname'] + ' ' + df['player.lastname']
df.drop(['player.firstname', 'player.lastname'], axis=1, inplace=True)

# Drop the 'comment' column
df.drop('comment', axis=1, inplace=True)

# Function to get the opposing team code
def find_opponent(row, df):
    same_game = df[df['game.id'] == row['game.id']]  # Find all rows with the same game.id
    opp_team = same_game[same_game['team.code'] != row['team.code']]['team.code']
    return opp_team.iloc[0] if not opp_team.empty else None

# Apply the function to create the 'opponent' column
df['opponent'] = df.apply(lambda row: find_opponent(row, df), axis=1)

# Function to project minutes based on player’s position and team
def project_minutes(player_name, team_code, position, opponent, df):
    # Filter out the player's historical data from the same team
    player_data = df[(df['player_name'] == player_name) & (df['team.code'] == team_code)]

    # Further filter by position to make the projection specific to their role
    position_data = player_data[player_data['pos'] == position]

    # Compute the player's average minutes per game
    avg_minutes_played = position_data['min'].mean()

    # If there's no data available, assume an average value of 20 minutes as a fallback
    if pd.isna(avg_minutes_played):
        avg_minutes_played = 20

    # Return the projected minutes
    return avg_minutes_played

# Apply minutes projection to each player in the dataframe
df['projected_minutes'] = df.apply(lambda row: project_minutes(row['player_name'], row['team.code'], row['pos'], row['opponent'], df), axis=1)

# Select only the relevant columns: player_name and projected_minutes
players_and_minutes = df[['player_name', 'projected_minutes']]

# Combine same players by averaging their projected minutes
average_players_and_minutes = players_and_minutes.groupby('player_name').agg({'projected_minutes': 'mean'}).reset_index()

# Display the combined results
print(average_players_and_minutes.head())

# Optional: Save the result to a new CSV file
average_players_and_minutes.to_csv("average_players_and_projected_minutes.csv", index=False)
