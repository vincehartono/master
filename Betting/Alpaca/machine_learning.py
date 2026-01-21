import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import alpaca_trade_api as tradeapi

def predict_tomorrow_close_lstm(stock_data):
    # Prepare the data
    data = stock_data['close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create the dataset
    X_train = []
    y_train = []
    for i in range(60, len(scaled_data)):  # Use the past 60 days to predict the next day
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Create and fit the LSTM model
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32)

    # Predict the closing price for the next day
    last_60_days = scaled_data[-60:]  # Get the last 60 days data for prediction
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))

    predicted_scaled = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(predicted_scaled)

    return predicted_price[0, 0]


def predict_tomorrow_close_xgboost(stock_data):
    # Prepare the data
    data = stock_data['close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Create the dataset
    X_train = []
    y_train = []
    for i in range(60, len(scaled_data)):  # Use the past 60 days to predict the next day
        X_train.append(scaled_data[i - 60:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape the data for XGBoost input
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1]))

    # Create and fit the XGBoost model
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Predict the closing price for the next day
    last_60_days = scaled_data[-60:]  # Get the last 60 days data for prediction
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0]))

    predicted_scaled = model.predict(last_60_days)
    predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

    return predicted_price[0, 0]


def place_order(api, symbol, qty, side, order_type='market', time_in_force='gtc'):
    api.submit_order(
        symbol=symbol,
        qty=qty,
        side=side,
        type=order_type,
        time_in_force=time_in_force
    )