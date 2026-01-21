import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import random
import logging
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

logging.basicConfig(filename='result_log.log',
                    format='%(asctime)s %(message)s',
                    filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# _date = dt.date.today() + dt.timedelta(days=1)
_date = dt.date.today() + dt.timedelta(days=0)
df = pd.read_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\data.csv")
df['game_date'] = pd.to_datetime(df['game_date'])

# df[['away_score', 'home_score']] = df[['away_score', 'home_score']].apply(pd.to_numeric)
df['total.points'] = df['away_score'] + df['home_score']

df['day'] = df['game_date'].dt.day
df['month'] = df['game_date'].dt.month
df['year'] = df['game_date'].dt.year

df = df.drop(columns=['game_date'])

df_result = df

df_result = df_result[df_result['year'] == _date.year]
df_result = df_result[df_result['month'] == _date.month]
df_result = df_result[df_result['day'] == _date.day]

df_result = df_result[['away_name', 'home_name', 'away_score', 'home_score', 'venue_id']]

# df_result.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_result.csv")

df = pd.get_dummies(df, columns=['away_name', 'home_name', 'home_probable_pitcher', 'away_probable_pitcher', 'venue_id'])

df_today = df[df['year'] == _date.year]
df_today = df_today[df_today['month'] == _date.month]
df_today = df_today[df_today['day'] == _date.day]

# df.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df.csv")

df_train = df[df['status'] == 'Final']

df_train = df_train.drop(columns=['away_score', 'home_score', 'status', 'Unnamed: 0', 'day', 'month', 'year'])
df_today = df_today.drop(columns=['away_score', 'home_score', 'status', 'Unnamed: 0', 'day', 'month', 'year', 'total.points'])

df_train.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_train.csv")
df_today.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_today.csv")

features = df_train.drop(columns='total.points')
label = df_train['total.points']

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.1)

# # Define the parameter grid for Random Forest
# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth': [4, 5, 6, 7, 8],
#     'criterion': ['gini', 'entropy']
# }
#
# # Instantiate the Random Forest classifier
# rf = RandomForestClassifier(random_state=42)

# # Perform grid search to find the best parameters
# CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
# CV_rfc.fit(x_train, y_train)
#
# # Print the best parameters found by grid search
# print(CV_rfc.best_params_)

for i in range(1,101):
    a = random.randint(1,1000)
    # rf = RandomForestClassifier(random_state=a, **CV_rfc.best_params_)
    rf = RandomForestClassifier(random_state=a)
    rf.fit(x_train, y_train)
    df_result[i] = rf.predict(df_today)

    print(i)

# # Define the parameter grid for Logistic Regression
# param_grid_lr = {
#     'penalty': ['l1', 'l2'],
#     'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
#     'solver': ['liblinear', 'saga']
# }
#
# # Instantiate the Logistic Regression classifier
# lr = LogisticRegression(max_iter=1000)

# # Perform grid search to find the best parameters for Logistic Regression
# CV_lr = GridSearchCV(estimator=lr, param_grid=param_grid_lr, cv=5)
# CV_lr.fit(x_train, y_train)
#
# # Print the best parameters found by grid search for Logistic Regression
# print("Best parameters for Logistic Regression: ", CV_lr.best_params_)

# Train Logistic Regression
# best_lr = LogisticRegression(max_iter=1000, **CV_lr.best_params_)
# best_lr.fit(x_train, y_train)
lr = LogisticRegression(max_iter=1000)
lr.fit(x_train, y_train)
df_result['logistic_regression'] = lr.predict(df_today)

df_result['average'] = df_result.iloc[:, 5:15].mean(axis=1)
df_result['std'] = df_result.iloc[:, 5:15].std(axis=1)
df_result['min'] = df_result['average'] - df_result['std']
df_result['max'] = df_result['average'] + df_result['std']

df_result = df_result[['away_name', 'home_name', 'average', 'std', 'min', 'max', 'logistic_regression']]

df_result.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df_result.csv")


# rf = RandomForestClassifier(random_state=42)
#
# param_grid = {
#     'n_estimators': [200, 500],
#     'max_features': ['auto', 'sqrt', 'log2'],
#     'max_depth' : [4,5,6,7,8],
#     'criterion' :['gini', 'entropy']
# }
#
# CV_rfc = GridSearchCV(estimator=rf, param_grid=param_grid, cv= 5)
# CV_rfc.fit(x_train, y_train)
#
# print(CV_rfc.best_params_)
#
# rfc1=RandomForestClassifier(random_state=42, max_features='log2', n_estimators= 200, max_depth=6, criterion='gini')
# rfc1.fit(x_train, y_train)
#
# pred=rfc1.predict(x_test)
#
# print("Accuracy for Random Forest on CV data: ",accuracy_score(y_test,pred))