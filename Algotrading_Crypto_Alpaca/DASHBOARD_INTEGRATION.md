# Dashboard Integration - Complete ✅

## Overview
The Flask web dashboard has been fully integrated into **trade.py**. The bot now runs with a background Flask server that displays live trading information.

## How It Works

### 1. **Shared State (DashboardState)**
- Global `dashboard_state` object holds all data needed for the dashboard
- Updated continuously as the bot trades and processes data

### 2. **Live Data Syncing**
The bot automatically syncs the following data to the dashboard:

| Component | Update Point | Data |
|-----------|-------------|------|
| **Strategy Info** | Bot init | Strategy name, symbol, timeframe, combo score |
| **Price & Indicators** | Every trading loop | Current price, high, low, volume, RSI, SMA, Bollinger Bands, etc. |
| **Account Info** | After each trade/balance check | Equity, cash, buying power |
| **Trade Signals** | When BUY/SELL executed | Trade timestamp, symbol, side, quantity, price, status |

### 3. **Flask Web Server**
- Runs on **http://localhost:5000** in a background thread
- Serves `dashboard.html` template
- Provides `/api/data` endpoint with JSON data (updated every request)
- Auto-refresh on web page (5-second polling)

## Integration Changes Made to trade.py

### 1. **DashboardState Class** (Lines 63-98)
```python
class DashboardState:
    - strategy, symbol, timeframe, combo_score, win_rate, profit_factor, sharpe
    - price, high, low, volume, timestamp
    - indicators dict (RSI, SMA, EMA, MACD, Bollinger Bands, etc.)
    - equity, cash, buying_power
    - recent_trades list (timestamp, symbol, side, quantity, price, status)
    - last_signal, last_signal_time
```

### 2. **Imports Added** (Lines 28-42)
```python
import threading
from flask import Flask, render_template, jsonify
FLASK_AVAILABLE = True/False flag
```

### 3. **start_dashboard_server() Function** (Lines 261-313)
- Flask app creation
- Route: `/` → dashboard.html
- Route: `/api/data` → JSON with all dashboard state
- Runs on port 5000 with use_reloader=False (important for threading)

### 4. **Bot Initialization** (bot.__init__)
- Syncs selected strategy, symbol, timeframe to dashboard_state

### 5. **get_account() Method**
- Syncs equity, cash, buying_power to dashboard_state

### 6. **place_buy_order() Method**
- Sets `last_signal = "BUY"`
- Appends trade to `recent_trades` list with full details

### 7. **place_sell_order() Method** ✨ NEW
- Sets `last_signal = "SELL"`
- Appends trade to `recent_trades` list with full details

### 8. **run_trading_loop() Method** ✨ EXPANDED
- After fetching bars and calculating indicators:
  - Syncs current price, high, low, volume to dashboard_state
  - Syncs all indicator values to dashboard_state.indicators dict
  - Updates timestamp

### 9. **main() Function** ✨ NEW
- Creates Flask thread: `threading.Thread(target=start_dashboard_server, daemon=True)`
- Starts thread before trading loop begins
- Prints: "Dashboard: http://localhost:5000"

## Usage

### Start the Bot with Dashboard

```bash
# Run the trading bot
python trade.py
```

**Output will show:**
```
[+] Dashboard: http://localhost:5000

[MODE] PAPER Trading
Strategy: Bollinger Bands
Symbols: ['DOGE/USD']
Position Size: 10% of account
Stop Loss: $5.00
Profit Target: $2.00
```

### View the Dashboard

1. Open browser to: **http://localhost:5000**
2. Dashboard automatically updates every 5 seconds
3. Shows:
   - **Strategy Card**: Selected strategy, pair, timeframe, performance metrics
   - **Price Card**: Current price, 24h high/low, volume, indicators (RSI bar, SMA, Bollinger Bands)
   - **Account Card**: Equity, cash, buying power with status indicator
   - **Trades Table**: Last 20 trades with timestamp, side (BUY/SELL), quantity, price, status

### Example: Watch Bollinger Band Entry/Exit

When bot enters a trade with Bollinger Bands:

1. **Entry (BUY)**
   - Price drops below lower Bollinger Band
   - Dashboard shows: `BUY DOGE/USD 100 units @ $0.2345`
   - Last signal shows: `BUY` (in green)

2. **Exit (SELL)**
   - Price returns above middle or upper band
   - Dashboard shows: `SELL DOGE/USD 100 units @ $0.2456`
   - Last signal shows: `SELL` (in red)
   - Trade appears in history with +$11.10 profit

## Data Flow Diagram

```
AlpacaCryptoBot.run_trading_loop()
    ↓
fetch bars for symbol
    ↓
calculate_indicators(df)
    ↓
UPDATE dashboard_state:
  - price, high, low, volume
  - indicators (all values)
  - timestamp
    ↓
generate_signal()
    ↓
BUY/SELL decision
    ↓
place_buy_order() / place_sell_order()
    ↓
UPDATE dashboard_state:
  - last_signal ("BUY" or "SELL")
  - recent_trades (append new trade)
  - timestamp
    ↓
Flask /api/data endpoint
    ↓
Dashboard.html (polls every 5 seconds)
    ↓
Browser displays:
  - Strategy info
  - Live price & indicators
  - Account status
  - Trade history with new entry
```

## Key Files

| File | Purpose |
|------|---------|
| `trade.py` | Main bot with integrated Flask server (1142 lines) |
| `templates/dashboard.html` | Web dashboard UI (280 lines) |
| `backtest.py` | Backtesting engine (unchanged) |
| `.env` | API credentials (unchanged) |

## Dependencies

```bash
pip install flask  # Installed via install_python_packages
```

Verified installed in Python environment.

## Troubleshooting

### Dashboard not loading?
- Ensure Flask installed: `pip install flask`
- Check port 5000 is available: `netstat -an | find "5000"`
- If port in use: Edit line 313 in trade.py: `app.run(..., port=5001)`

### No data showing?
- Dashboard updates every 5 seconds
- Check Flask server started: Look for "[+] Dashboard: http://localhost:5000" in console
- F5 refresh browser if data stuck

### Trades not appearing?
- Check console output for trade signals
- Verify backtest selected a working strategy
- Monitor "Last Trade:" timestamp at bottom of dashboard

### Indicators showing as 0 or None?
- Ensure enough data bars (lookback_bars >= strategy needs)
- Check NaN handling in calculate_indicators()
- Verify symbol/timeframe has data

## Performance Notes

- Flask runs in daemon thread (won't block bot shutdown)
- JSON updates only on API request (~5 second polling)
- Memory efficient: Keeps only last 20 trades in recent_trades
- No performance impact on bot speed

## Next Steps (Optional Enhancements)

- [ ] Add WebSocket for real-time updates (instead of polling)
- [ ] Add trade statistics chart (P&L over time)
- [ ] Add strategy parameter adjustment controls
- [ ] Add alert notifications for large P&L swings
- [ ] Add export trades to CSV from dashboard

---

**Status**: ✅ **READY TO USE**

Run `python trade.py` and open http://localhost:5000 to see live trading!
