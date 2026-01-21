import krakenex
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

# Initialize Kraken API
api = krakenex.API()
api.load_key(r'Kraken\kraken.key')  # Use your API credentials saved in this file.

# Define CSV file path for logging trades
csv_file_path = "trade_log.csv"

# Create CSV file with headers if it does not exist
def create_csv_file():
    if not os.path.exists(csv_file_path):
        # Create the file with headers
        df = pd.DataFrame(columns=["timestamp", "action", "entry_price", "stop_loss", "take_profit"])
        df.to_csv(csv_file_path, index=False)

# Log trade details to CSV
def log_trade_to_csv(action, entry_price, stop_loss, take_profit):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Current timestamp
    new_trade = {
        "timestamp": timestamp,
        "action": action,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "take_profit": take_profit
    }
    df = pd.DataFrame([new_trade])
    
    # Append to CSV file
    df.to_csv(csv_file_path, mode='a', header=False, index=False)

# Fetch market data
def fetch_ohlc(pair='SHIBUSD', interval=15):
    print(f"Fetching OHLC data for {pair}...")
    res = api.query_public('OHLC', {'pair': pair, 'interval': interval})
    if 'error' in res and res['error']:
        raise Exception("Error fetching OHLC data:", res['error'])
    data = res['result'][pair]
    ohlc = {
        'open': np.array([float(item[1]) for item in data]),
        'high': np.array([float(item[2]) for item in data]),
        'low': np.array([float(item[3]) for item in data]),
        'close': np.array([float(item[4]) for item in data]),
    }
    print(f"Successfully fetched OHLC data for {pair}.")
    return ohlc

# Identify Engulfing patterns
def engulfing_pattern_strategy(pair='SHIBUSD'):
    print(f"Identifying engulfing pattern for {pair}...")
    ohlc = fetch_ohlc(pair)
    
    # Convert to DataFrame
    df = pd.DataFrame(ohlc)
    
    # Calculate Bullish and Bearish Engulfing patterns
    df['bullish_engulfing'] = ((df['open'] < df['close'].shift(1)) & (df['open'].shift(1) < df['close']))
    df['bearish_engulfing'] = ((df['open'] > df['close'].shift(1)) & (df['open'].shift(1) > df['close']))

    # Check if the latest pattern is a Bullish or Bearish Engulfing
    if df['bullish_engulfing'].iloc[-1]:
        print("Bullish Engulfing detected, preparing to BUY.")
        return "BUY"
    elif df['bearish_engulfing'].iloc[-1]:
        print("Bearish Engulfing detected, preparing to SELL.")
        return "SELL"
    
    print("No Engulfing pattern detected, holding.")
    return "HOLD"

# Get the current balance of a specific pair (e.g., SHIB, USD) for holdings
def check_holdings(pair):
    # Get the account balance from Kraken API
    balance_res = api.query_private('Balance')
    
    if 'error' in balance_res and balance_res['error']:
        raise Exception("Error fetching balance:", balance_res['error'])
    
    # Get balances for pair (example: 'SHIB' or 'USD')
    pair_balance = balance_res['result'].get(pair, 0)
    print(f"Current holdings for {pair}: {pair_balance}")
    return float(pair_balance)

# Load the last trade details from CSV
def load_last_trade_details(pair):
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        last_trade = df[df['action'].isin(['BUY', 'SELL'])].iloc[-1] if not df.empty else None
        if last_trade is not None and last_trade['action'] in ['BUY', 'SELL']:
            return {
                "entry_price": last_trade['entry_price'],
                "stop_loss": last_trade['stop_loss'],
                "take_profit": last_trade['take_profit'],
                "action": last_trade['action']
            }
    return None

# Execute trades with dynamic risk management (stop-loss & take-profit)
def execute_trade_with_risk_management(pair, action, volume):
    print(f"Executing {action} trade for {pair} with volume {volume}...")

    price_res = api.query_public('Ticker', {'pair': pair})
    entry_price = float(price_res['result'][pair]['c'][0])  # Current price

    # Dynamically calculate stop-loss and take-profit
    stop_loss, take_profit = calculate_stop_loss_take_profit(entry_price)

    if action == "BUY":
        order = api.query_private('AddOrder', {
            'pair': pair,
            'type': 'buy',
            'ordertype': 'market',
            'volume': volume
        })
        if 'error' in order and order['error']:
            print(f"Error placing buy order: {order['error']}")
            return None

        # Log trade to CSV
        log_trade_to_csv(action, entry_price, stop_loss, take_profit)
        print(f"Buy order executed at {entry_price}. Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
        return {"type": "BUY", "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit}

    elif action == "SELL":
        order = api.query_private('AddOrder', {
            'pair': pair,
            'type': 'sell',
            'ordertype': 'market',
            'volume': volume
        })
        if 'error' in order and order['error']:
            print(f"Error placing sell order: {order['error']}")
            return None

        # Log trade to CSV
        log_trade_to_csv(action, entry_price, stop_loss, take_profit)
        print(f"Sell order executed at {entry_price}. Stop-Loss: {stop_loss}, Take-Profit: {take_profit}")
        return {"type": "SELL", "entry_price": entry_price, "stop_loss": stop_loss, "take_profit": take_profit}

    print("No action taken.")
    return None

# Function to calculate stop-loss and take-profit dynamically based on the entry price
def calculate_stop_loss_take_profit(entry_price, stop_loss_pct=0.20, take_profit_pct=0.10):
    """
    Calculate stop-loss and take-profit prices based on entry price and percentage thresholds.
    :param entry_price: The price at which the trade was executed (buy/sell).
    :param stop_loss_pct: The percentage below entry price for stop-loss.
    :param take_profit_pct: The percentage above entry price for take-profit.
    :return: tuple (stop_loss, take_profit)
    """
    stop_loss = entry_price * (1 - stop_loss_pct)
    take_profit = entry_price * (1 + take_profit_pct)
    return stop_loss, take_profit


# Monitor and manage stop-loss and take-profit
def monitor_trade(pair, trade_details, volume):
    print("Monitoring active trade...")
    current_price = float(api.query_public('Ticker', {'pair': pair})['result'][pair]['c'][0])

    if trade_details['type'] == "BUY":
        if current_price <= trade_details['stop_loss']:
            print(f"STOP-LOSS triggered at {current_price}")
            execute_trade_with_risk_management(pair, "SELL", volume)
            return True
        elif current_price >= trade_details['take_profit']:
            print(f"TAKE-PROFIT triggered at {current_price}")
            execute_trade_with_risk_management(pair, "SELL", volume)
            return True

    elif trade_details['type'] == "SELL":
        if current_price >= trade_details['stop_loss']:
            print(f"STOP-LOSS triggered at {current_price}")
            execute_trade_with_risk_management(pair, "BUY", volume)
            return True
        elif current_price <= trade_details['take_profit']:
            print(f"TAKE-PROFIT triggered at {current_price}")
            execute_trade_with_risk_management(pair, "BUY", volume)
            return True
    
    return False

# Main bot loop
def trading_bot():
    pair = 'SHIBUSD'
    volume = 1000000  # Adjust volume
    active_trade = None

    # Create or check CSV file for logging trades
    create_csv_file()
    
    while True:
        try:
            # Check if already holding positions
            current_holdings = check_holdings(pair)
            if current_holdings > 0:
                print(f"Holding {current_holdings} units of {pair}. Monitoring current position.")

                # Load the last trade details from CSV
                last_trade_details = load_last_trade_details(pair)
                if last_trade_details:
                    print(f"Last trade entry price: {last_trade_details['entry_price']}")
                    # Apply stop-loss and take-profit rules
                    trade_closed = monitor_trade(pair, last_trade_details, volume)
                    if trade_closed:
                        active_trade = None
                else:
                    print(f"No recorded trade found for {pair}.")
            else:
                if not active_trade:
                    print("No active trade. Identifying next move...")
                    action = engulfing_pattern_strategy(pair)
                    active_trade = execute_trade_with_risk_management(pair, action, volume)

            if active_trade:
                trade_closed = monitor_trade(pair, active_trade, volume)  # Pass volume here
                if trade_closed:
                    active_trade = None

        except Exception as e:
            print(f"Error: {e}")

        print("Wait for 15 minutes ...")
        time.sleep(9000)  # 15-minute intervals

if __name__ == "__main__":
    trading_bot()
