import pandas as pd
import datetime as dt
import logging
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from get_data import fetch_and_save_nba_data  # Make sure your get_data module includes the fetch function

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load and preprocess data
def load_and_preprocess_data(file_path, target_date):
    logging.info("Loading and preprocessing data.")
    df = pd.read_csv(file_path)
    df['date.start'] = pd.to_datetime(df['date.start'])

    df['day'] = df['date.start'].dt.day
    df['month'] = df['date.start'].dt.month
    df['year'] = df['date.start'].dt.year

    df = pd.get_dummies(df, columns=['arena.city', 'teams.visitors.code', 'teams.home.code'])
    df_today = df[(df['year'] == target_date.year) &
                  (df['month'] == target_date.month) &
                  (df['day'] == target_date.day)].copy()

    # Check if 'Unnamed: 0' exists before dropping
    columns_to_drop = ['date.start']
    if 'Unnamed: 0' in df.columns:
        columns_to_drop.append('Unnamed: 0')
    df = df.drop(columns=columns_to_drop)
    df_today = df_today.drop(columns=columns_to_drop)

    logging.info("Data loaded and preprocessed successfully.")
    return df, df_today


# Function to prepare training data
def prepare_training_data(df):
    logging.info("Preparing training data.")
    df_train = df[df['status.long'] == 'Finished'].copy()
    df_train['total.points'] = df_train['scores.visitors.points'] + df_train['scores.home.points']
    df_train = df_train.drop(columns=['scores.visitors.points', 'scores.home.points', 'status.long'])

    logging.info("Training data prepared.")
    return df_train


# Function to create Decision Tree model
def create_decision_tree_model():
    logging.info("Creating Decision Tree model.")
    model = DecisionTreeRegressor(random_state=42)
    logging.info("Decision Tree model created.")
    return model


# Function to train Decision Tree model
def train_decision_tree_model(x_train, y_train, df_today):
    logging.info("Training Decision Tree model.")
    model = create_decision_tree_model()
    model.fit(x_train, y_train)

    # Predict on today's data
    df_result = pd.DataFrame()
    df_result['prediction'] = model.predict(df_today)
    logging.info("Model training completed.")
    return df_result


# Function to calculate and store results
def calculate_and_store_results(df_result, teams_visitor, teams_home, output_path):
    logging.info("Calculating and storing results.")
    df_result['std'] = df_result['prediction'].std()
    df_result['min'] = df_result['prediction'].min()
    df_result['max'] = df_result['prediction'].max()

    # Reset index before concatenating to ensure alignment
    teams_visitor = teams_visitor.reset_index(drop=True)
    teams_home = teams_home.reset_index(drop=True)
    df_result = df_result.reset_index(drop=True)

    df_result = pd.concat([teams_visitor, teams_home, df_result[['prediction', 'std', 'min', 'max']]], axis=1)
    df_result.to_csv(output_path, index=False)

    logging.info(f"Results saved to {output_path}")


# Main function to fetch data, preprocess, train model, and save results
def main():
    logging.info("Starting the main process.")
    api_key = "a22eff4319msh7f19aefa5ca6675p1153b3jsn08075099bd47"
    host = "api-nba-v1.p.rapidapi.com"
    season = "2024"
    file_path = r"C:\Users\Vince\Downloads\Python\Betting\Games\data.csv"
    output_path = r"C:\Users\Vince\Downloads\Python\Betting\Games\df_result.csv"
    target_date = dt.date.today() + dt.timedelta(days=1)

    # Fetch and save NBA data
    fetch_and_save_nba_data(api_key, host, season, file_path)

    # Load and preprocess data
    df, df_today = load_and_preprocess_data(file_path, target_date)
    df_train = prepare_training_data(df)

    features = df_train.drop(columns='total.points')
    label = df_train['total.points']

    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, random_state=42)

    # Ensure df_today has the same columns as features for training and prediction
    df_today_filtered = df_today[features.columns.intersection(df_today.columns)].copy()
    print(f"Filtered df_today shape: {df_today_filtered.shape}")

    # Convert to float
    x_train = x_train.values.astype(float)
    df_today_filtered = df_today_filtered.values.astype(float)

    df_result = train_decision_tree_model(x_train, y_train, df_today_filtered)

    # Combine one-hot encoded columns back to original columns for output
    teams_visitors = df_today[[col for col in df_today.columns if col.startswith('teams.visitors.code_')]].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
    teams_home = df_today[[col for col in df_today.columns if col.startswith('teams.home.code_')]].idxmax(axis=1).apply(lambda x: x.split('_')[-1])

    # Calculate and store results
    calculate_and_store_results(df_result, teams_visitors, teams_home, output_path)

    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
