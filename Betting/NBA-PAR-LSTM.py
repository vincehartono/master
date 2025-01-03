import requests

from bs4 import BeautifulSoup

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, LSTM

from tensorflow.keras.losses import Huber

 

def fetch_data(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    points = []
    assists = []
    rebounds = []
    three_point_made = []

    for row in soup.select('#pgl_basic tbody tr'):
        if row.get('class') and 'thead' in row['class']:
            continue

        pts = row.select_one('td[data-stat="pts"]')
        ast = row.select_one('td[data-stat="ast"]')
        trb = row.select_one('td[data-stat="trb"]')
        fg3 = row.select_one('td[data-stat="fg3"]')

        if pts and ast and trb and fg3:
            points.append(int(pts.text))
            assists.append(int(ast.text))
            rebounds.append(int(trb.text))
            three_point_made.append(int(fg3.text))

    return {
        'points': points,
        'assists': assists,
        'rebounds': rebounds,
        '3P Made': three_point_made
    }


 

def prepare_data(sequence, n_steps):

    X, y = [], []

    for i in range(len(sequence)-n_steps):

        X.append(sequence[i:i+n_steps])

        y.append(sequence[i+n_steps])

    X = np.array(X)

    y = np.array(y)

    return X.reshape((X.shape[0], X.shape[1], 1)), y

 

def train_and_predict_next_number(X, y):

    model = Sequential()

    model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], 1)))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss=Huber(delta=1.0))

    model.fit(X, y, epochs=100, verbose=0)

 

    x_input = np.array(y[-3:]).reshape((1, X.shape[1], 1))

    next_number_prediction = model.predict(x_input, verbose=0)[0][0]

    return max(0, next_number_prediction)

 

url = 'https://www.basketball-reference.com/players/j/jamesle01/gamelog/2023'

fetched_data = fetch_data(url)

 

n_steps = 3

sequence_sets = {
    'Lebron James': [
        {'data': fetched_data['points'], 'label': 'points'},
        {'data': fetched_data['assists'], 'label': 'assists'},
        {'data': fetched_data['rebounds'], 'label': 'rebounds'},
        {'data': fetched_data['3P Made'], 'label': '3P Made'}
    ],
    # Add more sets of input sequences with custom unique names as needed
}


print("The predicted next numbers in the sequence sets are:")

for set_key, sequences in sequence_sets.items():

    print(f"\n{set_key}:")

    for seq in sequences:

        X, y = prepare_data(seq['data'], n_steps)

        next_number_prediction = train_and_predict_next_number(X, y)

        print(f"  {seq['label'].capitalize()}: {next_number_prediction:.2f}")

 