import pandas as pd
import datetime as dt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import numpy as np

_date = dt.date.today() + dt.timedelta(days=1)
# _date = dt.date.today() + dt.timedelta(days=0)
df = pd.read_csv(r"C:\Users\Vince\Downloads\Python\Betting\Games\data.csv")
df['date.start'] = pd.to_datetime(df['date.start'])

df['day'] = df['date.start'].dt.day
df['month'] = df['date.start'].dt.month
df['year'] = df['date.start'].dt.year

df_result = df[['year', 'month', 'day', 'teams.visitors.code', 'teams.home.code']]

df_result = df_result[df_result['year'] == _date.year]
df_result = df_result[df_result['month'] == _date.month]
df_result = df_result[df_result['day'] == _date.day]

df = pd.get_dummies(df, columns=['arena.city', 'teams.visitors.code', 'teams.home.code'])
df_today = df[df['year'] == _date.year]
df_today = df_today[df_today['month'] == _date.month]
df_today = df_today[df_today['day'] == _date.day]

df = df.drop(columns=['date.start'])

df_train = df[df['status.long'] == 'Finished']
df_train['total.points'] = df_train['scores.visitors.points'] + df_train['scores.home.points']
df_train = df_train.drop(columns=['Unnamed: 0', 'scores.visitors.points', 'scores.home.points', 'status.long'])

df_today = df_today.drop(columns=['Unnamed: 0', 'scores.visitors.points', 'scores.home.points', 'status.long', 'date.start'])

features = df_train.drop(columns='total.points')
label = df_train['total.points']

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size = 0.2)

model = LogisticRegression(random_state=42)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred))

# print(df_today)

df_result = model.predict(df_today)
print(df_result)

# df_result.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\Games\df_result_logistic.csv")

# from pprint import pprint
# # Look at parameters used by our current forest
# print('Parameters currently in use:\n')
# pprint(model.get_params())

# create complex Logistic Regression with max_iter=131
# log_model = LogisticRegression(max_iter=131, verbose=2, random_state=42)
# log_model.fit(x_train, y_train)
# y_pred_log = log_model.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_log))

# Create the parameter grid based on the results of grid search
# Penalty type
# penalty = ['l1', 'l2', 'elasticnet', 'none']
# # Solver type
# solver = ['lbfgs', 'liblinear']
# # Maximum number of iterations
# max_iter = [int(x) for x in np.linspace(start = 80, stop = 120, num = 5)]
# # Multi class
# multi_class = ['auto', 'ovr']
# # Verbosity
# verbose = [0, 1, 2]
# # l1 ratio
# l1_ratio = [0, 0.8, 0.9, 1]
# # C
# C = [0.5, 0.75, 1.0, 1.25, 1.5]
#
# # Create the param grid
# param_grid = {'penalty': penalty, 'solver': solver, 'max_iter':max_iter,
#     'multi_class':multi_class, 'verbose':verbose, 'l1_ratio':l1_ratio,
#     'C':C
# }
# pprint(param_grid)
#
# # Instantiate the grid search model with 2-fold cross-validation
# log_grid_search = GridSearchCV(estimator = LogisticRegression(random_state=42), param_grid = param_grid, cv = 2, n_jobs = -1)
#
# # Fit the grid search to the data
# log_grid_search.fit(x_train, y_train)
# best_log_grid = log_grid_search.best_estimator_
# best_log_grid.fit(x_train, y_train)
# y_pred_best_log = best_log_grid.predict(x_test)
# print(metrics.accuracy_score(y_test, y_pred_best_log))