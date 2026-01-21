import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import random
import logging

logging.basicConfig(filename='result_log.log',
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# Define the date
_date = dt.date.today()

# Load the data
df = pd.read_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\data.csv")
df['game_date'] = pd.to_datetime(df['game_date'])

# Calculate total points
df['total.points'] = df['away_score'] + df['home_score']

# Extract day, month, and year
df['day'] = df['game_date'].dt.day
df['month'] = df['game_date'].dt.month
df['year'] = df['game_date'].dt.year

# Drop unnecessary columns
df = df.drop(columns=['game_date'])

# Filter for the specific date
df_result = df[(df['year'] == _date.year) & (df['month'] == _date.month) & (df['day'] == _date.day)]
df_result = df_result[['away_name', 'home_name', 'away_score', 'home_score', 'venue_id']]

# Convert categorical variables to dummies
df = pd.get_dummies(df, columns=['away_name', 'home_name', 'home_probable_pitcher', 'away_probable_pitcher', 'venue_id'])

# Filter today's games
df_today = df[(df['year'] == _date.year) & (df['month'] == _date.month) & (df['day'] == _date.day)]

# Prepare training data
df_train = df[df['status'] == 'Final']
df_train = df_train.drop(columns=['away_score', 'home_score', 'status', 'Unnamed: 0', 'day', 'month', 'year'])
df_today = df_today.drop(columns=['away_score', 'home_score', 'status', 'Unnamed: 0', 'day', 'month', 'year', 'total.points'])

# Define features and labels
features = df_train.drop(columns='total.points')
label = df_train['total.points']

df_rf = pd.DataFrame(index=df_result.index)
df_xgb = pd.DataFrame(index=df_result.index)

logger.debug("test")

for i in range(1,101):
    a = random.randint(1, 1000)
    # Split the data
    x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, random_state=a)

    # Random Forest Parameter Tuning
    rf = RandomForestClassifier(random_state=a)
    param_grid_rf = {
        'n_estimators': [100, 200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth': [4, 6, 8, 10],
        'criterion': ['gini', 'entropy']
    }

    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, verbose=2)
    grid_search_rf.fit(x_train, y_train)

    # Best parameters for Random Forest
    best_rf_model = grid_search_rf.best_estimator_
    df_rf[i] = best_rf_model.predict(df_today)

    # XGBoost Parameter Tuning
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=a)
    param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=3, n_jobs=-1, verbose=2)
    grid_search_xgb.fit(x_train, y_train)

    # Best parameters for XGBoost
    best_xgb_model = grid_search_xgb.best_estimator_
    df_xgb[i] = best_xgb_model.predict(df_today)

df_rf['rf_avg'] = df_rf.mean(axis=1)
df_xgb['xgb_avg'] = df_xgb.mean(axis=1)

df_rf.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_rf.csv", index=False)
df_xgb.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_xgb.csv", index=False)

df_result['rf_avg'] = df_rf['rf_avg']
df_result['xgb_avg'] = df_xgb['xgb_avg']

# x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1, random_state=19)

# # Logistic Regression Parameter Tuning
# lr = LogisticRegression(max_iter=1000)
# param_grid_lr = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.01, 0.1, 1, 10, 100],
#     'solver': ['liblinear', 'saga']
# }
#
# grid_search_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=3, n_jobs=-1, verbose=2)
# grid_search_lr.fit(x_train, y_train)
#
# # Best parameters for Logistic Regression
# best_lr_model = grid_search_lr.best_estimator_
# df_result['logistic_regression'] = best_lr_model.predict(df_today)
#
# # SVM Parameter Tuning
# svr = SVR()
# param_grid_svm = {
#     'kernel': ['linear', 'rbf'],
#     'C': [0.1, 1, 10, 100],
#     'gamma': ['scale', 'auto', 0.01, 0.1, 1]
# }
#
# grid_search_svm = GridSearchCV(estimator=svr, param_grid=param_grid_svm, cv=3, n_jobs=-1, verbose=2)
# grid_search_svm.fit(x_train, y_train)
#
# # Best parameters for SVM
# best_svm_model = grid_search_svm.best_estimator_
# df_result['svm'] = best_svm_model.predict(df_today)

# Select final columns to save
df_result = df_result[['away_name', 'home_name', 'rf_avg', 'xgb_avg']]

# Save the final result
df_result.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_result.csv", index=False)