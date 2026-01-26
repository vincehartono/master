# Visual Guide: What You'll See

## 1. RUNNING THE BOT

### Terminal Output
```bash
C:\Users\Vince\master\Algotrading_Crypto_Alpaca> python trade.py

Loading backtest results for strategy selection...

============================================================================
TOP 5 STRATEGIES BY COMBO SCORE
============================================================================

1. Bollinger Bands (5Min)
   Combo Score: 87.3
   Profit Factor: 2.15
   Win Rate: 65%
   Return: 12.5%
   Sharpe: 1.23

2. RSI (1Min)
   Combo Score: 84.2
   ...

3. SMA Crossover (5Min)
   ...

[+] Selected: Bollinger Bands (5Min) on DOGE/USD

[+] Dashboard: http://localhost:5000

================================================================================
[MODE] PAPER Trading
Strategy: Bollinger Bands
Symbols: ['DOGE/USD']
Position Size: 10% of account
Stop Loss: $5.00
Profit Target: $2.00

[OK] Connected. Account equity: $10,000.00

[Trading loop started...]

[Loop 0001] Checking DOGE/USD...
[Loop 0002] Checking DOGE/USD...
```

### Browser Tab Opens
```
Navigate to: http://localhost:5000
```

---

## 2. DASHBOARD LOADS

### Page Layout
```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  🤖 TRADING BOT DASHBOARD                                                  ║
║     Status: ● CONNECTED                              Refresh: 5 seconds    ║
║                                                                            ║
╠════════════════════════════════════════════════════════════════════════════╣
║                                                                            ║
║  ┌──────────────────────────────────────┬──────────────────────────────┐ ║
║  │ 📊 STRATEGY CARD                     │ 💹 PRICE CARD                │ ║
║  ├──────────────────────────────────────┼──────────────────────────────┤ ║
║  │ Strategy: Bollinger Bands            │ Symbol: DOGE/USD             │ ║
║  │ Pair: DOGE/USD                       │ Price: $0.2345               │ ║
║  │ Timeframe: 5Min                      │ 24h High: $0.2500            │ ║
║  │ Combo Score: 87.3                    │ 24h Low: $0.2300             │ ║
║  │ Win Rate: 65%                        │ Volume: 15.2M                │ ║
║  │ Profit Factor: 2.15                  │ Updated: 10:15:30            │ ║
║  │ Sharpe Ratio: 1.23                   │                              │ ║
║  │                                      │                              │ ║
║  └──────────────────────────────────────┴──────────────────────────────┘ ║
║                                                                            ║
║  ┌──────────────────────────────────────┬──────────────────────────────┐ ║
║  │ 📈 INDICATORS CARD                   │ 💰 ACCOUNT CARD              │ ║
║  ├──────────────────────────────────────┼──────────────────────────────┤ ║
║  │                                      │ Equity: $10,000.00           │ ║
║  │ RSI(14): 48.5                        │ Cash: $10,000.00             │ ║
║  │ [████████░░░░░░░░░░] 48%             │ Buying Power: $10,000.00     │ ║
║  │                                      │                              │ ║
║  │ SMA(10): 0.2343                      │ Status: ● CONNECTED          │ ║
║  │ SMA(30): 0.2348                      │ Mode: PAPER TRADING          │ ║
║  │                                      │                              │ ║
║  │ BB Upper: 0.2450 ═════════           │                              │ ║
║  │ BB Middle: 0.2380 ────                │                              │ ║
║  │ BB Lower: 0.2310 ═════════           │                              │ ║
║  │                                      │                              │ ║
║  │ MACD: 0.0015                         │                              │ ║
║  │ Stochastic: 42.3                     │                              │ ║
║  │ ATR: 0.0045                          │                              │ ║
║  │                                      │                              │ ║
║  └──────────────────────────────────────┴──────────────────────────────┘ ║
║                                                                            ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │ 📋 RECENT TRADES                                                   │  ║
║  ├─────────────────────┬───────┬──────┬─────┬──────────┬────────────┤  ║
║  │ Timestamp           │ Symbol│ Side │ Qty │ Price    │ Status     │  ║
║  ├─────────────────────┼───────┼──────┼─────┼──────────┼────────────┤  ║
║  │ (no trades yet)     │       │      │     │          │            │  ║
║  │                     │       │      │     │          │            │  ║
║  └─────────────────────┴───────┴──────┴─────┴──────────┴────────────┘  ║
║                                                                            ║
║  Last Signal: NONE  |  Last Trade: --:--:--  |  Updates: 10:15:30        ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 3. PRICE DROPS (BUY SIGNAL)

### Terminal Output
```bash
[Loop 0047] Checking DOGE/USD...

BUY signal DOGE/USD | Price: $0.2310 | SMA(10): 0.2343 | SMA(30): 0.2348
[BUY] DOGE/USD x100 @ Order #98765432
[OK] Trade: DOGE/USD BUY x100 @ $0.2310 - Submitted
[Trade log: BUY,DOGE/USD,100,0.2310,98765432,Submitted]
```

### Dashboard Updates (5 seconds later)
```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  🤖 TRADING BOT DASHBOARD                                                  ║
║     Status: ● CONNECTED                              Refresh: 5 seconds    ║
║                                                                            ║
║  Price Card:                    Account Card:                             ║
║  ├─ Price: $0.2310 (DOWN!)      ├─ Equity: $9,769.00 (position opened)   ║
║  ├─ 24h High: $0.2500           ├─ Cash: $8,769.00 (locked in margin)    ║
║  ├─ 24h Low: $0.2300            └─ Status: ● TRADING                     ║
║  └─ Updated: 10:18:45                                                     ║
║                                                                            ║
║  Indicators Card:                                                          ║
║  ├─ RSI(14): 28.5  ◀ OVERSOLD (BUY SIGNAL!)                              ║
║  │  [████░░░░░░░░░░░░░] 28%                                              ║
║  ├─ SMA(10): 0.2343                                                       ║
║  ├─ SMA(30): 0.2348                                                       ║
║  ├─ BB Lower: 0.2310 (PRICE HERE! Buy signal)                            ║
║  └─ Updated: 10:18:45                                                     ║
║                                                                            ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │ 📋 RECENT TRADES                                                   │  ║
║  ├─────────────────────┬───────┬──────┬─────┬──────────┬────────────┤  ║
║  │ Timestamp           │ Symbol│ Side │ Qty │ Price    │ Status     │  ║
║  ├─────────────────────┼───────┼──────┼─────┼──────────┼────────────┤  ║
║  │ 10:18:45            │DOGE/US│ BUY  │100 │ $0.2310  │ Submitted  │  ║
║  │                     │       │      │    │          │            │  ║
║  └─────────────────────┴───────┴──────┴─────┴──────────┴────────────┘  ║
║                                                                            ║
║  Last Signal: BUY ✓ (green)  |  Last Trade: 10:18:45  |  Profit: $0.00   ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 4. PRICE RECOVERS (SELL SIGNAL)

### Terminal Output
```bash
[Loop 0095] Checking DOGE/USD...

SELL signal DOGE/USD | Price: $0.2435 | RSI: 71.2 (overbought)
[SELL] DOGE/USD x100 @ Order #98765433
[OK] Trade: DOGE/USD SELL x100 @ $0.2435 - Submitted
[Trade log: SELL,DOGE/USD,100,0.2435,98765433,Submitted]
[P&L: +$12.50 | Target: $2.00 (✓ PROFIT!)]
```

### Dashboard Updates (5 seconds later)
```
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║  🤖 TRADING BOT DASHBOARD                                                  ║
║     Status: ● CONNECTED                              Refresh: 5 seconds    ║
║                                                                            ║
║  Price Card:                    Account Card:                             ║
║  ├─ Price: $0.2435 (UP!)        ├─ Equity: $10,012.50 (PROFIT!)          ║
║  ├─ 24h High: $0.2500           ├─ Cash: $10,012.50 (position closed)    ║
║  ├─ 24h Low: $0.2300            └─ Status: ● CONNECTED                   ║
║  └─ Updated: 10:25:30                                                     ║
║                                                                            ║
║  Indicators Card:                                                          ║
║  ├─ RSI(14): 71.2  ◀ OVERBOUGHT (SELL SIGNAL!)                           ║
║  │  [██████████████░░░░░░░░░] 71%                                        ║
║  ├─ SMA(10): 0.2410                                                       ║
║  ├─ SMA(30): 0.2378                                                       ║
║  ├─ BB Upper: 0.2450 (PRICE NEAR, sell signal)                           ║
║  └─ Updated: 10:25:30                                                     ║
║                                                                            ║
║  ┌────────────────────────────────────────────────────────────────────┐  ║
║  │ 📋 RECENT TRADES                                                   │  ║
║  ├─────────────────────┬───────┬──────┬─────┬──────────┬────────────┤  ║
║  │ Timestamp           │ Symbol│ Side │ Qty │ Price    │ Status     │  ║
║  ├─────────────────────┼───────┼──────┼─────┼──────────┼────────────┤  ║
║  │ 10:18:45            │DOGE/US│ BUY  │100 │ $0.2310  │ Submitted  │  ║
║  │ 10:25:30            │DOGE/US│ SELL │100 │ $0.2435  │ Submitted  │  ║
║  │                     │       │      │    │          │            │  ║
║  └─────────────────────┴───────┴──────┴─────┴──────────┴────────────┘  ║
║                                                                            ║
║  Last Signal: SELL ✗ (red)  |  Last Trade: 10:25:30  |  Profit: +$12.50  ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
```

---

## 5. CONTINUING TO TRADE

### Multiple Cycles
```
Terminal shows:
[Loop 0100] Checking DOGE/USD...
[Loop 0101] Checking DOGE/USD...
BUY signal DOGE/USD | Price: $0.2278...
[BUY] DOGE/USD x109 @ Order #98765434
...
SELL signal DOGE/USD | Price: $0.2510...
[SELL] DOGE/USD x109 @ Order #98765435
[P&L: +$25.18 | Target: $2.00 (✓ PROFIT!)]

Dashboard shows:
Recent Trades table grows:
│ 10:18:45 │ DOGE/USD │ BUY  │ 100  │ 0.2310  │ Submitted │
│ 10:25:30 │ DOGE/USD │ SELL │ 100  │ 0.2435  │ Submitted │
│ 10:35:12 │ DOGE/USD │ BUY  │ 109  │ 0.2278  │ Submitted │
│ 10:41:50 │ DOGE/USD │ SELL │ 109  │ 0.2510  │ Submitted │

Account Equity: $10,037.68 (all profits!)
Total P&L: +$37.68
```

---

## 6. COLOR INDICATORS

### Status Lights
```
● CONNECTED (green)     - Bot is running and connected to Alpaca
● TRADING (yellow)      - Position currently open
● ALERT (red)           - Error or warning
○ DISCONNECTED (gray)   - Bot not running
```

### Signal Indicators
```
✓ BUY (green)     - Most recent signal was BUY
✗ SELL (red)      - Most recent signal was SELL
○ NONE (gray)     - No trades yet
```

### RSI Bar Colors
```
[████████░░] GREEN    RSI < 30 (oversold - buy opportunity)
[███████░░░░░░░░░░░░] YELLOW RSI 30-70 (neutral)
[████████████░░░░░░] RED      RSI > 70 (overbought - sell opportunity)
```

---

## 7. AUTO-REFRESH CYCLE

### Every 5 Seconds
```
Browser JavaScript:
  fetch('/api/data')
    │
    ├─ Get: strategy name, timeframe, combo score
    ├─ Get: current price, high, low, volume
    ├─ Get: all indicator values (RSI, SMA, BB, etc.)
    ├─ Get: account equity, cash, buying power
    ├─ Get: list of recent trades
    ├─ Get: last signal (BUY/SELL)
    └─ Get: current timestamp
  
  Update HTML:
    ├─ <h2>Price: $0.2435</h2>
    ├─ <div>RSI: 71.2</div>
    ├─ <table>Recent Trades...</table>
    ├─ <span>Equity: $10,012.50</span>
    └─ <span>Last: SELL at 10:25:30</span>

User sees:
  Dashboard auto-updates without page refresh!
```

---

## 8. PROFIT TARGET HIT

When profit reaches $2.00 (or more):

### Terminal Output
```bash
[+] PROFIT - Target reached! P&L: +$2.45
[CLOSE] Closing all positions...
[OK] Position closed. Trading stopped.
```

### Dashboard Shows
```
Account Card:
├─ Equity: $10,002.45 (PROFIT LOCKED!)
├─ Cash: $10,002.45
└─ Status: ✓ PROFIT TARGET HIT
```

### Bot Behavior
```
After profit target:
1. Close all open positions
2. Log P&L to file
3. Exit trading loop
4. Offer to restart or quit
5. Dashboard still accessible at http://localhost:5000
```

---

## 9. STOP LOSS HIT

If loss reaches -$5.00 (or more):

### Terminal Output
```bash
[-] STOP - Stop loss hit! P&L: -$5.00
[CLOSE] Closing all positions...
[OK] Position closed. Trading stopped.
```

### Dashboard Shows
```
Account Card:
├─ Equity: $9,995.00 (LOSS LIMITED!)
├─ Cash: $9,995.00
└─ Status: ✗ STOP LOSS HIT
```

---

## 10. EXAMPLE FULL SESSION

### Complete Scenario
```
TIME: 10:00:00
[+] Connected. Equity: $10,000.00
Strategy: Bollinger Bands on DOGE/USD

10:05:15
Price: $0.2400, RSI: 35 (oversold)
BUY signal! 100 units at $0.2400
Dashboard: [BUY]

10:15:30
Price: $0.2350 (down 2%)
No signal yet

10:22:45
Price: $0.2480 (up 3.3%)
RSI: 75 (overbought)
SELL signal! Exit at $0.2480
Dashboard: [SELL]
Profit: +$8.00

Account: $10,008.00
Dashboard shows both trades in history

10:30:00
Price: $0.2400
No signal yet

10:35:20
Price: $0.2280 (down)
RSI: 22 (very oversold)
BUY signal! 110 units at $0.2280
Dashboard: [BUY] again

...trading continues...
```

---

## Key Takeaways

✅ **Dashboard Updates in Real-Time**
- Every 5 seconds without page refresh
- Shows latest price, indicators, account, trades

✅ **Trade Visibility**
- See each entry (BUY) immediately
- See each exit (SELL) immediately
- Watch profit/loss in real-time

✅ **One Command to Start**
```bash
python trade.py
```
Then open: http://localhost:5000

✅ **No Separate Services**
- Flask server runs in background
- Dashboard integrated into same script
- Auto-stops when bot stops

**You're watching your bot trade live!** 🚀
