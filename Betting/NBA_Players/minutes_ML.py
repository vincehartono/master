import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib

# Load the data (assuming it's already cleaned up as per your previous code)
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

# Encode categorical variables with LabelEncoder
label_encoder = LabelEncoder()
df['pos_encoded'] = label_encoder.fit_transform(df['pos'])  # Positional encoding
df['team_encoded'] = label_encoder.fit_transform(df['team.code'])  # Team encoding
df['opponent_encoded'] = label_encoder.fit_transform(df['opponent'])  # Opponent encoding

# Ensure the 'game_date' column is in datetime format
df['game_date'] = pd.to_datetime(df['game_date'])

# Feature extraction from 'game_date'
df['game_year'] = df['game_date'].dt.year
df['game_month'] = df['game_date'].dt.month
df['game_day'] = df['game_date'].dt.day

# Model 1: Predict 'min' using Random Forest
X_min = df[['game_year', 'game_month', 'game_day', 'pos_encoded', 'team_encoded', 'opponent_encoded']]
y_min = df['min']  # Target: Minutes played

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_min, y_min, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor
model_min = RandomForestRegressor(random_state=42)

# Set up parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'max_features': ['auto', 'sqrt']
}

# Perform GridSearchCV for the best model
grid_search_min = GridSearchCV(estimator=model_min, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_min.fit(X_train, y_train)

# Use the best estimator to predict
best_model_min = grid_search_min.best_estimator_
y_pred_min = best_model_min.predict(X_test)

# Evaluate the minutes model
mse_min = mean_squared_error(y_test, y_pred_min)
print(f"\nMean Squared Error for 'min' Prediction: {mse_min:.2f}")

# Save the trained minutes model
joblib.dump(best_model_min, 'random_forest_min_model.pkl')

# Predict the 'min' for all rows
df['predicted_min'] = best_model_min.predict(X_min)

# ---- Rating System ----
def create_ratings(df):
    scaler = MinMaxScaler(feature_range=(0, 100))

    # Points Rating
    df['points_rating'] = scaler.fit_transform(df[['points']])

    # Rebounds Rating
    df['rebounds_rating'] = scaler.fit_transform(df[['totReb']])

    # Assists Rating
    df['assists_rating'] = scaler.fit_transform(df[['assists']])

    # Overall Player Rating (average of individual ratings)
    df['overall_rating'] = df[['points_rating', 'rebounds_rating', 'assists_rating']].mean(axis=1)

    return df

# Generate ratings and add them to the dataframe
df = create_ratings(df)

# Save the combined dataframe with predictions and ratings
df.to_csv("player_scores_with_predictions_and_ratings.csv", index=False)

# Display sample results
print("\nSample Data with Predicted Minutes and Ratings:")
print(df[['player_name', 'game_date', 'points', 'totReb', 'assists', 'predicted_min',
         'points_rating', 'rebounds_rating', 'assists_rating', 'overall_rating']].head())
