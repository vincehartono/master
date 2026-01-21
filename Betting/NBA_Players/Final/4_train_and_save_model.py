import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import pickle

# Load the dataset
data = pd.read_csv("player_PER_scores.csv")

# Preprocessing
# Select input and target variables
X = data[['player_name', 'team.code', 'min', 'uPER', 'normalized_PER', 'location']]
y_points = data['points']
y_totReb = data['totReb']
y_assists = data['assists']

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X[['player_name', 'team.code', 'location']])
X_numerical = X[['min', 'uPER', 'normalized_PER']].values
X_processed = pd.concat(
    [pd.DataFrame(X_encoded), pd.DataFrame(X_numerical)], axis=1)

# Save the fitted encoder for future use
with open("encoder.pkl", "wb") as f:
    pickle.dump(encoder, f)
    print("Encoder saved as 'encoder.pkl'.")

# Split datasets for each target
X_train_pts, X_test_pts, y_train_pts, y_test_pts = train_test_split(
    X_processed, y_points, test_size=0.2, random_state=42)
X_train_reb, X_test_reb, y_train_reb, y_test_reb = train_test_split(
    X_processed, y_totReb, test_size=0.2, random_state=42)
X_train_ast, X_test_ast, y_train_ast, y_test_ast = train_test_split(
    X_processed, y_assists, test_size=0.2, random_state=42)


# Function to find the best model using GridSearchCV
def train_and_tune_model(X_train, y_train, X_test, y_test, file_name):
    # Define the parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Best model
    best_model = grid_search.best_estimator_

    # Evaluate the model
    predictions = best_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"{file_name} Best Parameters: {grid_search.best_params_}")
    print(f"{file_name} MSE: {mse}")

    # Save the best model
    with open(file_name, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {file_name}")


# Train and tune models for each target
train_and_tune_model(X_train_pts, y_train_pts, X_test_pts, y_test_pts, "model_points.pkl")
train_and_tune_model(X_train_reb, y_train_reb, X_test_reb, y_test_reb, "model_totReb.pkl")
train_and_tune_model(X_train_ast, y_train_ast, X_test_ast, y_test_ast, "model_assists.pkl")

print("All models have been tuned, trained, and saved successfully.")
