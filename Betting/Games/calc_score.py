import pandas as pd
import datetime as dt
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
    df_train = df_train.drop(columns=['status.long'])

    logging.info("Training data prepared.")
    return df_train


# Function to train random forest model
def train_random_forest(x_train, y_train, df_today, num_iterations=100):
    logging.info("Training Random Forest model.")
    df_result = pd.DataFrame()
    for i in range(1, num_iterations + 1):
        rf = RandomForestClassifier(random_state=i)
        rf.fit(x_train, y_train)
        df_result[i] = rf.predict(df_today)

    logging.info("Model training completed.")
    return df_result


# Function to calculate and store results
def calculate_and_store_results(df_result_visitors, df_result_home, teams_visitor, teams_home, output_path):
    logging.info("Calculating and storing results.")
    df_result_visitors['average_visitors'] = df_result_visitors.mean(axis=1)
    df_result_visitors['std_visitors'] = df_result_visitors.std(axis=1)
    df_result_visitors['min_visitors'] = df_result_visitors['average_visitors'] - df_result_visitors['std_visitors']
    df_result_visitors['max_visitors'] = df_result_visitors['average_visitors'] + df_result_visitors['std_visitors']

    df_result_home['average_home'] = df_result_home.mean(axis=1)
    df_result_home['std_home'] = df_result_home.std(axis=1)
    df_result_home['min_home'] = df_result_home['average_home'] - df_result_home['std_home']
    df_result_home['max_home'] = df_result_home['average_home'] + df_result_home['std_home']

    # Reset index before concatenating to ensure alignment
    teams_visitor = teams_visitor.reset_index(drop=True)
    teams_home = teams_home.reset_index(drop=True)
    df_result_visitors = df_result_visitors.reset_index(drop=True)
    df_result_home = df_result_home.reset_index(drop=True)

    # Combine visitor and home results
    df_result = pd.concat([teams_visitor, teams_home, df_result_visitors[['average_visitors', 'std_visitors', 'min_visitors', 'max_visitors']], df_result_home[['average_home', 'std_home', 'min_home', 'max_home']]], axis=1)

    # Calculate point spread
    df_result['point_spread'] = df_result['average_home'] - df_result['average_visitors']

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
    target_date = dt.date.today() + dt.timedelta(days=0)

    # Fetch and save NBA data
    fetch_and_save_nba_data(api_key, host, season, file_path)

    # Load and preprocess data
    df, df_today = load_and_preprocess_data(file_path, target_date)
    df_train = prepare_training_data(df)

    features = df_train.drop(columns=['scores.visitors.points', 'scores.home.points'])
    visitor_points = df_train['scores.visitors.points']
    home_points = df_train['scores.home.points']

    x_train, x_test, y_train_visitors, y_test_visitors = train_test_split(features, visitor_points, test_size=0.1, random_state=42)
    x_train, x_test, y_train_home, y_test_home = train_test_split(features, home_points, test_size=0.1, random_state=42)

    # Ensure df_today has the same columns as features for training and prediction
    df_today_filtered = df_today[features.columns.tolist()]

    df_result_visitors = train_random_forest(x_train, y_train_visitors, df_today_filtered)
    df_result_home = train_random_forest(x_train, y_train_home, df_today_filtered)

    # Combine one-hot encoded columns back to original columns for output
    teams_visitors = df_today[[col for col in df_today.columns if col.startswith('teams.visitors.code_')]].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
    teams_home = df_today[[col for col in df_today.columns if col.startswith('teams.home.code_')]].idxmax(axis=1).apply(lambda x: x.split('_')[-1])

    # Calculate and store results
    calculate_and_store_results(df_result_visitors, df_result_home, teams_visitors, teams_home, output_path)

    logging.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()
