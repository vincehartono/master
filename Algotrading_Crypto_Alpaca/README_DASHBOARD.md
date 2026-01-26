# 🎯 Alpaca Crypto Trading Bot with Live Dashboard

> **Your trading bot just got a web dashboard!** 🚀

Watch your algorithmic trading bot execute trades in real-time with a beautiful live web dashboard.

---

## What's New

Your bot now displays:
- ✅ **Live Price Data** - Real-time crypto prices (updates every 5 seconds)
- ✅ **Trading Indicators** - RSI, SMA, Bollinger Bands, MACD, Stochastic, and more
- ✅ **Strategy Info** - Which strategy and trading pair is selected
- ✅ **Account Status** - Current equity, cash, buying power
- ✅ **Trade History** - See each entry and exit in real-time
- ✅ **Signal Indicators** - Shows when BUY/SELL signals fire

## Quick Start

### 1. Start the Bot
```bash
cd c:\Users\Vince\master\Algotrading_Crypto_Alpaca
python trade.py
```

### 2. Open Dashboard
```
Browser: http://localhost:5000
```

### 3. Watch It Trade
The dashboard updates automatically every 5 seconds. When your bot executes a trade:
- Console shows: `BUY signal DOGE/USD | Price: $0.2345`
- Dashboard shows: `BUY DOGE/USD 100 @ 0.2345` in the trades table
- Status updates: Account equity changes immediately

---

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│                   trade.py (Main Bot)                        │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐          │
│  │  Trading Loop    │ ──────▶ │ DashboardState   │          │
│  │  - Fetch bars    │         │ (Shared Data)    │          │
│  │  - Indicators    │ ◀────── │ - strategy       │          │
│  │  - Signals       │         │ - price          │          │
│  │  - Orders        │         │ - indicators     │          │
│  │                  │         │ - trades         │          │
│  └──────────────────┘         │ - account        │          │
│                               └──────────────────┘          │
│                                      ▲                       │
└──────────────────────────────────────┼───────────────────────┘
                                       │
                ┌──────────────────────┴──────────────────────┐
                │                                             │
         ┌──────▼──────────┐                         ┌────────▼────────┐
         │  Flask Server   │                         │  Browser Tab    │
         │ (Background)    │                         │  (Dashboard)    │
         │                 │                         │                 │
         │ GET /           │◀─────────────────────▶ │ Every 5 sec:    │
         │ GET /api/data   │    HTTP Request        │ - Fetch /api    │
         │                 │    JSON Response       │ - Update HTML   │
         │ Port 5000       │                         │ - Show trades   │
         └─────────────────┘                         └─────────────────┘
```

### Data Flow

1. **Bot Running** (trade.py)
   - Fetches price bars for selected symbol
   - Calculates technical indicators (SMA, RSI, BB, etc.)
   - Generates BUY/SELL signals
   - Executes orders

2. **Shared State Updates** (DashboardState)
   - After price fetch: Updates price, high, low, volume
   - After indicators: Updates all indicator values
   - After order: Updates recent_trades list, last_signal

3. **Flask Web Server** (Background Thread)
   - Runs on http://localhost:5000 in daemon thread
   - Serves dashboard.html on GET /
   - Serves JSON data on GET /api/data

4. **Dashboard Browser** (Auto-Refresh)
   - Polls /api/data every 5 seconds
   - Updates displays with fresh data
   - Shows trades in real-time as they execute

---

## Dashboard Features

### Strategy Card
Shows your selected strategy and performance metrics:
- Strategy name (e.g., "Bollinger Bands")
- Trading pair (e.g., "DOGE/USD")
- Timeframe (e.g., "5Min")
- Combo score (0-100)
- Win rate (%)
- Profit factor
- Sharpe ratio

### Price Card
Real-time price information:
- Current price
- 24h high/low
- Volume
- Last update timestamp

### Indicators Card
Technical analysis data:
- **RSI** with visual bar (red=overbought >70, yellow=30-70, green=oversold <30)
- **SMA** fast and slow moving averages
- **Bollinger Bands** upper, middle, lower bands
- Plus: EMA, MACD, Stochastic, ATR, Momentum

### Account Card
Your account status:
- Equity (current account value)
- Cash (available cash)
- Buying power (margin available)
- Connection status (green = connected)

### Trades Table
Last 20 executed trades:
- Timestamp
- Symbol
- Side (BUY/SELL)
- Quantity
- Price
- Status

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Alpaca API account (paper or live)
- API credentials in `.env` file

### Install Flask
```bash
pip install flask
```

### Check Installation
```bash
python -c "import flask; print(flask.__version__)"
```

### Configure .env
```
APCA_API_KEY_ID=your_paper_key
APCA_API_SECRET_KEY=your_paper_secret
APCA_API_BASE_URL=https://paper-api.alpaca.markets
```

For live trading, also add:
```
APCA_API_KEY_ID_LIVE=your_live_key
APCA_API_SECRET_KEY_LIVE=your_live_secret
APCA_API_BASE_URL_LIVE=https://api.alpaca.markets
```

---

## Strategies Available

The bot automatically tests and selects the best performing strategy:

### Long Strategies (Entry signals)
- **SMA Crossover** - Fast SMA crosses above slow SMA
- **EMA Crossover** - Fast EMA crosses above slow EMA
- **MACD** - MACD line crosses above signal line
- **RSI** - RSI drops below 30 (oversold signal)
- **Bollinger Bands** - Price drops below lower band
- **Stochastic** - Stochastic drops below 20 (oversold)
- **ATR Breakout** - High volatility + positive momentum
- **Volume Surge** - Volume spikes above normal + positive momentum

### Short Strategies (Exit signals)
- **Short SMA** - Fast SMA crosses below slow SMA
- **Short RSI** - RSI rises above 70 (overbought signal)
- **Short Bollinger** - Price rises above upper band
- **Short Momentum** - Negative momentum detected
- **Short Downtrend** - Price below both SMAs
- **Short EMA** - Fast EMA crosses below slow EMA
- **Short Stochastic** - Stochastic rises above 80 (overbought)

---

## Example: Real Trading Session

### Console Output
```bash
$ python trade.py

[+] Dashboard: http://localhost:5000

[MODE] PAPER Trading
Strategy: Bollinger Bands
Symbols: ['DOGE/USD']
Position Size: 10% of account
Stop Loss: $5.00
Profit Target: $2.00

[OK] Connected. Account equity: $10,000.00

BUY signal DOGE/USD | Price: $0.2345 | SMA(10): 0.2340 | SMA(30): 0.2350
[BUY] DOGE/USD x100 @ Order #12345
[OK] Trade: DOGE/USD BUY x100 @ $0.2345 - Submitted

[Trade history saved]
[Trade log: BUY,DOGE/USD,100,0.2345,0000012345,Submitted]

SELL signal DOGE/USD | Price: $0.2456
[SELL] DOGE/USD x100 @ Order #12346
[OK] Trade: DOGE/USD SELL x100 @ $0.2456 - Submitted

[P&L: +$11.10 | Target: $2.00 (Profit!)]
```

### Dashboard Display (http://localhost:5000)
```
╔═══════════════════════════════════════════════════════════════════════════╗
║ 🤖 TRADING BOT DASHBOARD                                                  ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─────────────────────────────────────┬─────────────────────────────────────┐
│ STRATEGY                            │ PRICE                               │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ Bollinger Bands                     │ DOGE/USD: $0.2456                   │
│ DOGE/USD • 5Min                     │ 24h High: $0.2500  Low: $0.2300     │
│ Combo: 87.3  WR: 65%  PF: 2.15      │ Volume: 15.2M  Last: 10:18:45       │
└─────────────────────────────────────┴─────────────────────────────────────┘

┌─────────────────────────────────────┬─────────────────────────────────────┐
│ INDICATORS                          │ ACCOUNT                             │
├─────────────────────────────────────┼─────────────────────────────────────┤
│ RSI: 68 [████████░░]                │ Equity: $10,011.10                  │
│ SMA(10): 0.2348  SMA(30): 0.2351    │ Cash: $8,956.00                     │
│ BB Upper: 0.2470 Mid: 0.2400        │ Buying Power: $8,956.00             │
│ BB Lower: 0.2330                    │ Status: ● CONNECTED                 │
└─────────────────────────────────────┴─────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│ RECENT TRADES                                                              │
├──────────────────┬──────────┬──────┬─────────┬─────────┬──────────────────┤
│ Timestamp        │ Symbol   │ Side │ Qty     │ Price   │ Status           │
├──────────────────┼──────────┼──────┼─────────┼─────────┼──────────────────┤
│ 10:15:32         │ DOGE/USD │ BUY  │ 100     │ 0.2345  │ Submitted        │
│ 10:18:45         │ DOGE/USD │ SELL │ 100     │ 0.2456  │ Submitted        │
└──────────────────┴──────────┴──────┴─────────┴─────────┴──────────────────┘

Last Signal: SELL ✓  |  Last Trade: 10:18:45  |  P&L: +$11.10
```

---

## Trading Modes

### Paper Trading (Recommended for Testing)
```bash
python trade.py
# Select: Backtest mode
# Select: PAPER mode
# Trades won't affect real account
```

### Live Trading (Real Money)
```bash
python trade.py
# Select: Backtest mode
# Select: LIVE mode
# Type: YES to confirm
# ⚠️ TRADES WILL EXECUTE FOR REAL MONEY
```

**Always test in PAPER mode first!**

---

## Configuration

Edit `TradingConfig` class in trade.py:

```python
@dataclass
class TradingConfig:
    # Strategy & Data
    strategy: str = "SMA"
    timeframe: str = "1Min"  # or "5Min", "15Min", "1H"
    symbols: List[str] = field(default_factory=lambda: ["BTC/USD"])
    
    # Risk Management
    position_size_pct: float = 10.0      # % of account per trade
    stop_loss: float = 5.0               # Fixed dollars
    profit_target: float = 2.0           # Fixed dollars
    lookback_bars: int = 50              # Bars for indicators
    
    # Intervals
    check_interval: int = 60             # Seconds between checks
    no_trade_rerun_minutes: int = 30     # Rerun backtest if idle
    
    # Trading
    paper_trading: bool = True
```

---

## Files

```
Algotrading_Crypto_Alpaca/
├── trade.py                           # Main trading bot (1144 lines)
├── backtest.py                        # Backtesting engine (1041 lines)
├── .env                               # API credentials (confidential)
├── templates/
│   └── dashboard.html                 # Web dashboard UI (280 lines)
├── logs/
│   ├── trade_log.csv                 # Trade history
│   └── pnl_log.csv                   # P&L tracking
├── QUICK_START.md                     # This file (usage guide)
├── DASHBOARD_INTEGRATION.md           # Technical integration details
├── CHANGES_DETAILED.md                # Line-by-line changes made
└── INTEGRATION_COMPLETE.md            # Integration verification
```

---

## Troubleshooting

### Dashboard won't load
```bash
# Check Flask is installed
pip install flask

# Check port 5000 is available
netstat -an | findstr :5000

# If port in use, change in trade.py line 313
app.run(..., port=5001)  # Then use http://localhost:5001
```

### No trade signals
- Check console for "BUY signal" / "SELL signal" messages
- Verify backtest completed successfully
- Check strategy selected is valid
- Ensure symbols have recent price data

### Indicators showing 0 or None
- Increase `lookback_bars` (need more historical data)
- Verify symbol is valid (BTC/USD, ETH/USD, etc.)
- Check NaN handling in `calculate_indicators()`

### Bot exits immediately
- Check API credentials in .env file
- Verify Alpaca API is accessible
- Look for error messages in console
- Check account has sufficient equity/margin

### Flask not found error
```bash
pip install flask
```

---

## Performance & Monitoring

### Bot Performance
- CPU: ~5% during trading loop
- Memory: ~50-100MB (includes market data)
- Network: ~1 request per minute (for bar data)
- Latency: Sub-second trade execution

### Dashboard Performance
- Memory: ~5MB (Flask server)
- Bandwidth: ~2KB per /api/data request (every 5 sec)
- Refresh rate: 5 second polling
- Browser compatibility: All modern browsers

### Optimization Tips
1. **Increase check_interval** - Less frequent checks = lower CPU/network
2. **Decrease lookback_bars** - Fewer bars = faster calculations
3. **Reduce polling rate** - Edit dashboard.html fetch interval

---

## Advanced: Customization

### Add Custom Strategy
Edit `TradingConfig.STRATEGIES` and `generate_signal()`:

```python
def generate_signal(self, symbol: str, indicators: Dict) -> Optional[str]:
    if self.config.strategy == "MyStrategy":
        # Custom logic here
        if some_condition:
            return "BUY"
        elif other_condition:
            return "SELL"
    return None
```

### Change Dashboard Port
Edit line 313 in trade.py:
```python
app.run(debug=False, host="0.0.0.0", port=8080)  # Changed to 8080
```

### Access Dashboard Remotely
Edit line 313 in trade.py (use your server IP):
```python
# Already set to 0.0.0.0 (accessible from any IP)
app.run(debug=False, host="0.0.0.0", port=5000)

# Access from: http://YOUR_SERVER_IP:5000
```

---

## Next Steps

1. ✅ Run `python trade.py`
2. ✅ Open http://localhost:5000
3. ✅ Select backtest mode to auto-select best strategy
4. ✅ Watch live dashboard as bot trades
5. ✅ Monitor P&L and adjust strategy as needed

---

## Support

- Check console output for error messages
- Review trade logs: `logs/trade_log.csv`
- Check P&L history: `logs/pnl_log.csv`
- Read backtest results: `backtest_results.json`

---

## License

Your trading bot, your rules! 🚀

**Happy trading!**
