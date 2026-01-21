import pandas as pd
import pickle
import ast  # Safely evaluate string representations of lists
import numpy as np  # For statistical calculations
import time  # For measuring execution time

# Start the timer
start_time = time.time()

# Load the models
with open('model_assists.pkl', 'rb') as f:
    model_assists = pickle.load(f)

with open('model_points.pkl', 'rb') as f:
    model_points = pickle.load(f)

with open('model_totReb.pkl', 'rb') as f:
    model_totReb = pickle.load(f)

# Load the encoder (must be the same as the one used for training)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the input data
df = pd.read_csv('unique_players_today.csv')

# Step 3: Initialize lists to hold summarized statistics
summary_assists = []
summary_points = []
summary_rebounds = []

# Step 4: Loop through each player and make predictions for each simulation
for index, row in df.iterrows():
    # Convert string representations of lists to actual Python lists
    simulated_min = ast.literal_eval(row['simulated_min'])
    simulated_uPER = ast.literal_eval(row['simulated_uPER'])
    normalized_PER = ast.literal_eval(row['normalized_PER'])

    # Extract the categorical variables (player_name, team.code, location)
    categorical_data = [row['player_name'], row['team.code'], row['location']]
    categorical_encoded = encoder.transform([categorical_data])

    # Loop over the 1000 simulations and predict the stats for each
    assists_for_simulations = []
    points_for_simulations = []
    rebounds_for_simulations = []

    for i in range(1000):
        # Create the feature array for the current simulation
        features = [
            simulated_min[i],
            simulated_uPER[i],
            normalized_PER[i]
        ]

        # Combine the encoded categorical features with the numerical features
        features_full = list(categorical_encoded[0]) + features  # Concatenate categorical and numerical

        # Reshape input to match model expectation (model expects a 2D array for a single sample)
        features_reshaped = [features_full]  # List of one list to represent a single sample with all features

        # Get predictions for the current simulation
        assists_prediction = model_assists.predict(features_reshaped)
        points_prediction = model_points.predict(features_reshaped)
        rebounds_prediction = model_totReb.predict(features_reshaped)

        # Store the results for the current simulation
        assists_for_simulations.append(assists_prediction[0])  # model returns array, get the first element
        points_for_simulations.append(points_prediction[0])
        rebounds_for_simulations.append(rebounds_prediction[0])

    # Summarize the statistics
    assists_stats = {
        'mean': np.mean(assists_for_simulations),
        'median': np.median(assists_for_simulations),
        'std': np.std(assists_for_simulations),
        '95th_percentile': np.percentile(assists_for_simulations, 95)
    }

    points_stats = {
        'mean': np.mean(points_for_simulations),
        'median': np.median(points_for_simulations),
        'std': np.std(points_for_simulations),
        '95th_percentile': np.percentile(points_for_simulations, 95)
    }

    rebounds_stats = {
        'mean': np.mean(rebounds_for_simulations),
        'median': np.median(rebounds_for_simulations),
        'std': np.std(rebounds_for_simulations),
        '95th_percentile': np.percentile(rebounds_for_simulations, 95)
    }

    # Append the summarized stats to the respective lists
    summary_assists.append(assists_stats)
    summary_points.append(points_stats)
    summary_rebounds.append(rebounds_stats)

# Step 5: Flatten the summary into separate columns for assists, points, and rebounds
result_df = df.copy()

# For assists
result_df['predicted_assists_mean'] = [assists_stats['mean'] for assists_stats in summary_assists]
result_df['predicted_assists_median'] = [assists_stats['median'] for assists_stats in summary_assists]
result_df['predicted_assists_std'] = [assists_stats['std'] for assists_stats in summary_assists]
result_df['predicted_assists_95th_percentile'] = [assists_stats['95th_percentile'] for assists_stats in summary_assists]

# For points
result_df['predicted_points_mean'] = [points_stats['mean'] for points_stats in summary_points]
result_df['predicted_points_median'] = [points_stats['median'] for points_stats in summary_points]
result_df['predicted_points_std'] = [points_stats['std'] for points_stats in summary_points]
result_df['predicted_points_95th_percentile'] = [points_stats['95th_percentile'] for points_stats in summary_points]

# For rebounds
result_df['predicted_rebounds_mean'] = [rebounds_stats['mean'] for rebounds_stats in summary_rebounds]
result_df['predicted_rebounds_median'] = [rebounds_stats['median'] for rebounds_stats in summary_rebounds]
result_df['predicted_rebounds_std'] = [rebounds_stats['std'] for rebounds_stats in summary_rebounds]
result_df['predicted_rebounds_95th_percentile'] = [rebounds_stats['95th_percentile'] for rebounds_stats in summary_rebounds]

# Step 6: Save the results to a new CSV file with the requested columns
result_df[['player_name', 'team.code', 'location',
           'predicted_assists_mean', 'predicted_assists_median', 'predicted_assists_std', 'predicted_assists_95th_percentile',
           'predicted_points_mean', 'predicted_points_median', 'predicted_points_std', 'predicted_points_95th_percentile',
           'predicted_rebounds_mean', 'predicted_rebounds_median', 'predicted_rebounds_std', 'predicted_rebounds_95th_percentile']].to_csv('predicted_results_summary.csv', index=False)

# Stop the timer and print the total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Predictions saved to 'predicted_results_summary.csv'")
print(f"Execution time: {execution_time:.2f} seconds")
