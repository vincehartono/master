# 🚀 Quick Start: Live Dashboard with Trade.py

## What's New

Your Alpaca crypto trading bot now includes a **live web dashboard** that shows:
- ✅ Selected strategy & trading pair
- ✅ Real-time price & indicators
- ✅ Account status (equity, cash, buying power)
- ✅ Trade entries/exits as they happen
- ✅ Last 20 trade history

## Run Bot + Dashboard

```bash
# Terminal: Start the bot
cd c:\Users\Vince\master\Algotrading_Crypto_Alpaca
python trade.py
```

**You'll see:**
```
[+] Dashboard: http://localhost:5000

[MODE] PAPER Trading
Strategy: Bollinger Bands
Symbols: ['DOGE/USD']
...
```

## View Dashboard

```
Open browser → http://localhost:5000
```

**The dashboard automatically updates every 5 seconds!**

## Watch Your Trades Live

### Example: Bollinger Band Strategy on DOGE/USD

1. **Price drops below lower Bollinger Band**
   - Dashboard: Recent Trades table shows → `BUY  DOGE/USD  100  @$0.2345`
   - Last Signal: **BUY** (green indicator)

2. **Price rises back above middle band**
   - Dashboard: Recent Trades table shows → `SELL DOGE/USD  100  @$0.2456`
   - Last Signal: **SELL** (red indicator)
   - Profit: +$11.10 calculated

3. **Watch account equity update** as trades execute

## Dashboard Components

| Component | What It Shows | Updates |
|-----------|---------------|---------|
| **Strategy Card** | Name, pair, timeframe, combo score | On startup |
| **Price Card** | Current price, high, low, volume, timestamp | Every 5 sec |
| **Indicators Card** | RSI, SMA (fast/slow), Bollinger Bands | Every 5 sec |
| **Account Card** | Equity, cash, buying power, status | Every 5 sec |
| **Trades Table** | Last 20 trades (timestamp, side, qty, price) | Real-time |

## How It Works Behind the Scenes

```
1. Bot runs trading loop
2. Fetches latest price data
3. Calculates indicators (SMA, RSI, BB, etc.)
4. Generates BUY/SELL signal
5. Updates shared dashboard_state
6. Flask server serves data on port 5000
7. Browser polls /api/data every 5 seconds
8. Dashboard HTML updates display
```

**All in one Python process - no separate services needed!**

## Troubleshooting

### Dashboard won't load at http://localhost:5000

**Solution:** Check if Flask is installed
```bash
pip install flask
```

### Trades not showing in dashboard

**Check:**
1. Is bot running? (Look for "BUY signal" or "SELL signal" in console)
2. Has backtest completed? (Need to auto-select a strategy)
3. Are signals being generated? (Check console output for signal messages)

### Port 5000 already in use?

**Solution:** Edit trade.py line 313:
```python
app.run(debug=False, host="0.0.0.0", port=5001)  # Change to 5001
```

Then access: http://localhost:5001

### Indicators showing 0 or None?

**Check:**
1. Strategy needs enough bars (usually 20-50)
2. Verify `lookback_bars` setting in config
3. Ensure symbol has recent price data

## Key Files

```
Algotrading_Crypto_Alpaca/
├── trade.py                    # Main bot (now with Flask server)
├── backtest.py                 # Strategy backtesting
├── templates/
│   └── dashboard.html          # Web UI
├── DASHBOARD_INTEGRATION.md    # Full integration details
└── QUICK_START.md             # This file
```

## Supported Strategies

The bot automatically tests and selects the best performing strategy:

**Long Strategies:**
- SMA Crossover, EMA Crossover, MACD, RSI, Bollinger Bands
- Stochastic, ATR Breakout, Volume Surge

**Short Strategies:**
- Short SMA, Short RSI, Short Bollinger, Short Momentum
- Short Downtrend, Short EMA, Short Stochastic

Selection is automatic based on combo score ranking!

## Paper vs Live Trading

### Paper Trading (Safe Testing)
```bash
python trade.py
# Select: Backtest mode first
# Then: Select PAPER when asked
# Trades won't actually execute - great for testing!
```

### Live Trading (Real Money)
```bash
python trade.py
# Select: Backtest mode first
# Then: Select LIVE when asked
# CONFIRM with "YES" - trades will execute!
```

**⚠️ CAUTION:** Only enable LIVE after testing extensively in PAPER mode!

## Real Example: What You'll See

**Console Output:**
```
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

SELL signal DOGE/USD | Price: $0.2456
[SELL] DOGE/USD x100 @ Order #12346
```

**Dashboard HTML (http://localhost:5000):**
```
═══════════════════════════════════════════════════════════════════════════
BOLLINGER BANDS    DOGE/USD    1Min    Combo: 87.3    WR: 65%    PF: 2.15
───────────────────────────────────────────────────────────────────────────
Price: $0.2456    24H High: $0.2500    24H Low: $0.2300    Vol: 15.2M
───────────────────────────────────────────────────────────────────────────
RSI: 68 [████████░░] | SMA(10): 0.2348 | SMA(30): 0.2351
BB Upper: 0.2470  |  Middle: 0.2400  |  Lower: 0.2330
───────────────────────────────────────────────────────────────────────────
Equity: $10,011.10    Cash: $8,956.00    BP: $8,956.00    [● CONNECTED]
───────────────────────────────────────────────────────────────────────────
RECENT TRADES
Timestamp              Symbol    Side   Qty    Price     Status
2025-01-21 10:15:32   DOGE/USD  BUY    100    0.2345    Submitted
2025-01-21 10:18:45   DOGE/USD  SELL   100    0.2456    Submitted
```

## Performance Tips

1. **Paper mode first** - Test strategy before going live
2. **Watch the dashboard** - Monitor trades in real-time
3. **Set realistic targets** - $2 profit target with $5 stop loss
4. **Monitor P&L** - Bot auto-closes if profit/loss targets hit
5. **Check indicators** - Higher RSI (>70) = overbought, Lower (<30) = oversold

## Advanced Configuration

Edit `TradingConfig` in trade.py to customize:

```python
@dataclass
class TradingConfig:
    # Strategy & Data
    strategy: str = "SMA"           # Which strategy to use
    timeframe: str = "1Min"         # Bar timeframe (1Min, 5Min, 15Min, 1H)
    symbols: List[str] = field(     # Symbols to trade
        default_factory=lambda: ["BTC/USD", "ETH/USD"]
    )
    
    # Risk Management
    position_size_pct: float = 10.0 # % of equity per trade
    stop_loss: float = 5.0          # Fixed stop loss in dollars
    profit_target: float = 2.0      # Fixed profit target in dollars
    lookback_bars: int = 50         # Bars for indicators
```

## Need Help?

**Common Issues:**

1. **"ModuleNotFoundError: No module named 'flask'"**
   → Run: `pip install flask`

2. **"Port 5000 already in use"**
   → Change port in trade.py or stop other services

3. **"No trade signals generated"**
   → Check backtest completed successfully
   → Verify strategy logic in generate_signal()

4. **"Dashboard won't connect"**
   → Check Flask server started: `[+] Dashboard: http://localhost:5000`
   → Try http://localhost:5000 (if on same machine)
   → Or http://YOUR_IP:5000 (if on network)

---

**You're all set!** 🎉

Run `python trade.py` and open http://localhost:5000 to see your bot trade in real-time!

Questions? Check the code comments in trade.py or DASHBOARD_INTEGRATION.md for details.
