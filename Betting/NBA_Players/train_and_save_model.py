import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
import pickle
import pandas as pd

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
    X_processed, y_points, test_size=0.5, random_state=42)
X_train_reb, X_test_reb, y_train_reb, y_test_reb = train_test_split(
    X_processed, y_totReb, test_size=0.5, random_state=42)
X_train_ast, X_test_ast, y_train_ast, y_test_ast = train_test_split(
    X_processed, y_assists, test_size=0.5, random_state=42)


# Function to find the best model using GridSearchCV with randomized random_state
def train_and_tune_model(X_train, y_train, X_test, y_test, file_name):
    # Define the parameter grid
    param_grid = {
        'bootstrap': [True],  # Keeping it fixed since False may often lead to worse results
        'max_depth': [10, 50],  # Reduced depth options
        'max_features': ['sqrt', 'log2', None],  # Keeping it simple, 'sqrt' might add unnecessary complexity in some cases
        'min_samples_leaf': [2],  # Testing with one option
        'min_samples_split': [2, 5],  # Fewer choices to test
        'n_estimators': [100, 500, 1000]  # Reduced range
    }

    # Randomize the random state for each model training
    random_state = np.random.randint(0, 10000)  # Random state between 0 and 9999

    print(f"Training with random_state={random_state}")

    # Initialize the Random Forest Regressor with randomized random_state
    rf = RandomForestRegressor(random_state=random_state)

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
    print(f"{file_name} MSE with random_state={random_state}: {mse}")

    # Save the best model
    with open(file_name, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Model saved to {file_name}")


# Add the toggle for model selection
def select_model_to_train():
    print("Select the model(s) you want to train:")
    print("1. Points")
    print("2. Total Rebounds")
    print("3. Assists")
    print("4. All")

    # choice = input("Enter your choice (1, 2, 3, or 4): ").strip()
    choice = "3"

    if choice == "1":
        print("\nTraining model for Points...")
        train_and_tune_model(X_train_pts, y_train_pts, X_test_pts, y_test_pts, "model_points.pkl")
    elif choice == "2":
        print("\nTraining model for Total Rebounds...")
        train_and_tune_model(X_train_reb, y_train_reb, X_test_reb, y_test_reb, "model_totReb.pkl")
    elif choice == "3":
        print("\nTraining model for Assists...")
        train_and_tune_model(X_train_ast, y_train_ast, X_test_ast, y_test_ast, "model_assists.pkl")
    elif choice == "4":
        print("\nTraining all models...")
        #train_and_tune_model(X_train_pts, y_train_pts, X_test_pts, y_test_pts, "model_points.pkl")
        train_and_tune_model(X_train_reb, y_train_reb, X_test_reb, y_test_reb, "model_totReb.pkl")
        train_and_tune_model(X_train_ast, y_train_ast, X_test_ast, y_test_ast, "model_assists.pkl")
    else:
        print("Invalid choice. Please try again.")
        select_model_to_train()


# Run the model selection function
select_model_to_train()

print("All models have been tuned, trained, and saved successfully.")