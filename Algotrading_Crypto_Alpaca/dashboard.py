"""
Live Trading Dashboard
Displays real-time strategy, price, indicators, and P&L
"""

import os
import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, jsonify
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

load_dotenv()

# Initialize Flask
app = Flask(__name__)

# API Credentials
API_KEY = os.getenv("APCA_API_KEY_ID_PAPER")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY_PAPER")
BASE_URL = "https://paper-api.alpaca.markets"
os.environ["APCA_API_BASE_URL"] = BASE_URL

# Clients
trading_client = TradingClient(API_KEY, SECRET_KEY)
data_client = CryptoHistoricalDataClient()

# Current strategy (loaded from config or backtest results)
current_strategy = {
    "name": "SMA Crossover",
    "symbol": "BTC/USD",
    "timeframe": "1Min",
    "combo_score": 0.0,
}

def load_strategy_from_results():
    """Load current strategy from backtest results"""
    try:
        backtest_file = Path("backtest_results.json")
        if backtest_file.exists():
            with open(backtest_file, "r") as f:
                results = json.load(f)
            if results:
                # Find best by combo score
                best = max(results, key=lambda x: x.get('combo_score', 0))
                current_strategy["name"] = best.get('strategy_name', 'SMA Crossover')
                current_strategy["symbol"] = best.get('symbol', 'BTC/USD')
                current_strategy["timeframe"] = best.get('timeframe', '1Min')
                current_strategy["combo_score"] = best.get('combo_score', 0)
                current_strategy["win_rate"] = best.get('win_rate', 0) * 100
                current_strategy["profit_factor"] = best.get('profit_factor', 0)
                current_strategy["sharpe"] = best.get('sharpe_ratio', 0)
    except Exception as e:
        print(f"[WARNING] Could not load strategy: {e}")

def get_price_data():
    """Fetch current price and indicators"""
    try:
        symbol = current_strategy["symbol"]
        timeframe_str = current_strategy["timeframe"]
        
        # Parse timeframe
        amount = int(''.join(c for c in timeframe_str if c.isdigit())) or 1
        tf = TimeFrame(amount=amount, unit=TimeFrameUnit.Minute)
        
        # Fetch bars
        request = CryptoBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            limit=50,
        )
        bars_data = data_client.get_crypto_bars(request)
        
        if symbol in bars_data:
            df = bars_data[symbol].df
            if not df.empty:
                df = df.sort_index()
                latest = df.iloc[-1]
                
                # Calculate indicators
                close = df['close']
                high = df['high']
                low = df['low']
                
                # SMA
                sma_fast = close.rolling(10).mean().iloc[-1]
                sma_slow = close.rolling(30).mean().iloc[-1]
                
                # RSI
                delta = close.diff()
                gain = (delta.where(delta > 0, 0)).rolling(14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                
                # Bollinger Bands
                bb_mid = close.rolling(20).mean().iloc[-1]
                bb_std = close.rolling(20).std().iloc[-1]
                bb_upper = bb_mid + (bb_std * 2)
                bb_lower = bb_mid - (bb_std * 2)
                
                return {
                    "symbol": symbol,
                    "price": float(latest['close']),
                    "high": float(latest['high']),
                    "low": float(latest['low']),
                    "volume": float(latest['volume']),
                    "timestamp": latest.name.isoformat(),
                    "indicators": {
                        "sma_fast": float(sma_fast) if not pd.isna(sma_fast) else None,
                        "sma_slow": float(sma_slow) if not pd.isna(sma_slow) else None,
                        "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                        "bb_upper": float(bb_upper) if not pd.isna(bb_upper) else None,
                        "bb_lower": float(bb_lower) if not pd.isna(bb_lower) else None,
                        "bb_mid": float(bb_mid) if not pd.isna(bb_mid) else None,
                    }
                }
    except Exception as e:
        print(f"[ERROR] Failed to get price data: {e}")
    
    return {
        "symbol": current_strategy["symbol"],
        "price": 0,
        "high": 0,
        "low": 0,
        "volume": 0,
        "timestamp": datetime.now().isoformat(),
        "indicators": {}
    }

def get_account_info():
    """Get current account equity and P&L"""
    try:
        account = trading_client.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
        }
    except Exception as e:
        print(f"[ERROR] Failed to get account: {e}")
    return {"equity": 0, "cash": 0, "buying_power": 0}

def get_recent_trades():
    """Get recent trades from log"""
    try:
        trade_log = Path("crypto_trades.csv")
        if trade_log.exists():
            import pandas as pd
            df = pd.read_csv(trade_log)
            return df.tail(10).to_dict('records')
    except Exception as e:
        print(f"[WARNING] Could not load trades: {e}")
    return []

@app.route("/")
def index():
    """Main dashboard page"""
    return render_template("dashboard.html")

@app.route("/api/strategy")
def api_strategy():
    """Get current strategy"""
    load_strategy_from_results()
    return jsonify(current_strategy)

@app.route("/api/price")
def api_price():
    """Get current price and indicators"""
    return jsonify(get_price_data())

@app.route("/api/account")
def api_account():
    """Get account info"""
    return jsonify(get_account_info())

@app.route("/api/trades")
def api_trades():
    """Get recent trades"""
    return jsonify(get_recent_trades())

@app.route("/api/data")
def api_data():
    """Get all data"""
    load_strategy_from_results()
    return jsonify({
        "strategy": current_strategy,
        "price": get_price_data(),
        "account": get_account_info(),
        "trades": get_recent_trades(),
        "timestamp": datetime.now().isoformat(),
    })

if __name__ == "__main__":
    # Create templates directory
    Path("templates").mkdir(exist_ok=True)
    
    print("=" * 80)
    print("LIVE TRADING DASHBOARD")
    print("=" * 80)
    print("\nStarting dashboard on http://localhost:5000")
    print("Press Ctrl+C to stop\n")
    
    app.run(debug=True, host="0.0.0.0", port=5000, use_reloader=False)
