# Crypto Trading Bot - File Organization

## ✅ All Files Consolidated

All crypto trading files are now centralized in:
```
C:\Users\Vince\master\Algotrading_Crypto_Alpaca\
```

### Data Files

| File | Size | Purpose |
|------|------|---------|
| `crypto_trade.log` | 27.9 KB | Trading activity log |
| `crypto_trades.csv` | 710 bytes | Trade history |
| `crypto_pnl.csv` | 122 bytes | Profit/Loss tracking |
| `backtest_results.csv` | 52.9 KB | Strategy backtest results |
| `backtest_results.json` | 134.4 KB | Best strategy config |

### Key Python Files

| File | Purpose |
|------|---------|
| `trade.py` | Main trading bot (1,391 lines) |
| `backtest.py` | Strategy backtesting engine |
| `.env` | API credentials (paper + live) |
| `test_existing_position.py` | Verify position detection |

### Trade.py Features Verified

✅ **Position Detection at Startup**
- Reads existing positions with `get_all_positions()`
- Logs entry price from `avg_entry_price`
- Example: `• DOGEUSD: +720.42 @ $0.1251`

✅ **P&L Monitoring**
- Checks profit target: P&L ≥ $2.00 → SELL
- Checks stop loss: P&L ≤ -$5.00 → SELL
- Calculates P&L from entry price vs current price

✅ **No Force-Close**
- Bot preserves existing positions on startup
- Does NOT sell when fresh start detected

### File Paths (All Relative)

All file references in trade.py are relative paths:
- `crypto_trade.log` ✓
- `crypto_trades.csv` ✓
- `crypto_pnl.csv` ✓

This ensures files are created/updated in the script directory.

### Running the Bot

```bash
cd C:\Users\Vince\master\Algotrading_Crypto_Alpaca
python trade.py
```

All logs and data will be saved to the same folder.

### Verification

Check for existing positions:
```bash
python test_existing_position.py
```

Result:
```
DOGEUSD: +720.42 @ $0.1251
Bot will NOT close these positions at startup
Bot will continue monitoring for profit target or stop loss
```
