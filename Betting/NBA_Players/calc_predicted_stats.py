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

# with open('model_points.pkl', 'rb') as f:
#     model_points = pickle.load(f)

# with open('model_totReb.pkl', 'rb') as f:
#     model_totReb = pickle.load(f)

# Load the encoder (must be the same as the one used for training)
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# Load the input data
df = pd.read_csv('merged_odds_and_players.csv')

# Step 3: Initialize lists to hold summarized statistics
summary_assists = []
summary_points = []
summary_rebounds = []

# Function to handle the model selection
def select_models_to_run():
    print("Select the prediction model(s) to run:")
    print("1. Assists")
    print("2. Points")
    print("3. Total Rebounds")
    print("4. All")

    choice = "1"
    #choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

    return choice

# Step 4: Loop through each player and make predictions for each simulation
def run_predictions(choice):
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

            # Conditional prediction based on user choice
            if choice == "1" or choice == "4":
                assists_prediction = model_assists.predict(features_reshaped)
                assists_for_simulations.append(assists_prediction[0])
            if choice == "2" or choice == "4":
                points_prediction = model_points.predict(features_reshaped)
                points_for_simulations.append(points_prediction[0])
            if choice == "3" or choice == "4":
                rebounds_prediction = model_totReb.predict(features_reshaped)
                rebounds_for_simulations.append(rebounds_prediction[0])

        # Calculate the 5th and 95th percentiles for the selected models
        if choice == "1" or choice == "4":
            assists_5th_percentile = np.percentile(assists_for_simulations, 30)
            assists_95th_percentile = np.percentile(assists_for_simulations, 95)
            summary_assists.append((assists_5th_percentile, assists_95th_percentile))
        if choice == "2" or choice == "4":
            points_5th_percentile = np.percentile(points_for_simulations, 30)
            points_95th_percentile = np.percentile(points_for_simulations, 95)
            summary_points.append((points_5th_percentile, points_95th_percentile))
        if choice == "3" or choice == "4":
            rebounds_5th_percentile = np.percentile(rebounds_for_simulations, 30)
            rebounds_95th_percentile = np.percentile(rebounds_for_simulations, 95)
            summary_rebounds.append((rebounds_5th_percentile, rebounds_95th_percentile))

# Get user choice for which models to run
choice = select_models_to_run()

# Run the selected predictions
run_predictions(choice)

# Step 5: Flatten the summary into separate columns for assists, points, and rebounds
result_df = df.copy()

# For assists (if needed)
if choice == "1" or choice == "4":
    result_df['assists_30th_percentile'] = [x[0] for x in summary_assists]
    result_df['assists_95th_percentile'] = [x[1] for x in summary_assists]

# For points (if needed)
if choice == "2" or choice == "4":
    result_df['points_30th_percentile'] = [x[0] for x in summary_points]
    result_df['points_95th_percentile'] = [x[1] for x in summary_points]

# For rebounds (if needed)
if choice == "3" or choice == "4":
    result_df['rebounds_30th_percentile'] = [x[0] for x in summary_rebounds]
    result_df['rebounds_95th_percentile'] = [x[1] for x in summary_rebounds]

# Step 6: Save the results to a new CSV file with the requested columns
result_columns = ['player_name', 'team.code', 'location', 'over/under']

# Add the results for the selected models
if choice == "1" or choice == "4":
    result_columns.extend(['assists_30th_percentile', 'assists_95th_percentile'])
if choice == "2" or choice == "4":
    result_columns.extend(['points_30th_percentile', 'points_95th_percentile'])
if choice == "3" or choice == "4":
    result_columns.extend(['rebounds_30th_percentile', 'rebounds_95th_percentile'])

# Step 7: Calculate 'Take over' and 'Take under' columns
if choice == "1" or choice == "4":
    result_df['Take over'] = (result_df['assists_30th_percentile'] / result_df['over/under']) - 1
    result_df['Take under'] = (result_df['over/under'] / result_df['assists_95th_percentile']) - 1
    result_columns.extend(['Take over', 'Take under'])

# For points (if needed)
if choice == "2" or choice == "4":
    result_df['Take over'] = (result_df['points_30th_percentile'] / result_df['over/under']) - 1
    result_df['Take under'] = (result_df['over/under'] / result_df['points_95th_percentile']) - 1
    result_columns.extend(['Take over', 'Take under'])

# For rebounds (if needed)
if choice == "3" or choice == "4":
    result_df['Take over'] = (result_df['rebounds_30th_percentile'] / result_df['over/under']) - 1
    result_df['Take under'] = (result_df['over/under'] / result_df['rebounds_95th_percentile']) - 1
    result_columns.extend(['Take over', 'Take under'])    

result_df[result_columns].to_csv('predicted_results_percentiles.csv', index=False)

# Stop the timer and print the total execution time
end_time = time.time()
execution_time = end_time - start_time
print(f"Predictions saved to 'predicted_results_percentiles.csv'")
print(f"Execution time: {execution_time:.2f} seconds")