# Alpaca Crypto Algorithmic Trading System

Complete automated crypto trading platform with backtesting framework.

## 📂 Files Overview

### Core Trading
- **`trade.py`** - Main trading bot for live/paper trading
- **`backtest.py`** - 20 strategies × 5 timeframes = 100 backtests
- **`analyze_backtest.py`** - Analyze and visualize backtest results
- **`.env.example`** - API credentials template

### Documentation
- **`SETUP.md`** - Installation and configuration guide
- **`BACKTEST_GUIDE.md`** - Detailed backtesting guide
- **`README.md`** - This file

### Output Files (Auto-Generated)
- **`backtest_results.json`** - Full backtest results
- **`backtest_results.csv`** - CSV format for Excel
- **`good_strategies.csv`** - Profitable strategies only
- **`best_strategies.csv`** - Top 20 strategies by Sharpe ratio
- **`crypto_trades.csv`** - Trade log during live trading
- **`crypto_pnl.csv`** - Profit/loss tracking
- **`crypto_trade.log`** - Timestamped event log

---

## 🚀 Quick Start (5 minutes)

### 1. Install Dependencies
```bash
# Core requirements
pip install alpaca-py pandas numpy python-dotenv

# Optional: Better technical indicators
pip install talib-binary
```

### 2. Get API Credentials
1. Visit https://app.alpaca.markets/
2. Sign up or log in
3. Go to API Keys section
4. Copy API Key ID and Secret Key

### 3. Configure `.env`
```bash
cp .env.example .env

# Edit .env and add your credentials:
# APCA_API_KEY_ID=your_key_here
# APCA_API_SECRET_KEY=your_secret_here
# APCA_API_BASE_URL=https://paper-api.alpaca.markets  (for paper trading)
```

### 4. Run Backtests
```bash
python backtest.py
```

### 5. Analyze Results
```bash
python analyze_backtest.py
```

---

## 📊 What's Included

### 20 Trading Strategies

**Trend Following:**
1. SMA Crossover
2. EMA Crossover
3. MACD
4. ADX Trend
5. Turtle Trading

**Momentum:**
6. Momentum
7. RSI
8. Stochastic
9. Williams %R

**Mean Reversion:**
10. Bollinger Bands
11. Mean Reversion
12. Keltner Channels
13. Envelope

**Volume/Volatility:**
14. ATR Breakout
15. Volume Surge
16. Dual MA + Volume
17. VWAP

**Specialized:**
18. Ichimoku Cloud
19. CCI
20. Grid Trading

### 5 Timeframes
- 1 Hour (intraday)
- 4 Hours (balanced)
- 1 Day (swing trading)
- 1 Week (position trading)
- 1 Month (trend following)

### Total: 100 Backtests

---

## 💡 Workflow

### Phase 1: Backtesting
```
1. Run backtest.py
2. Tests all 20 strategies × 5 timeframes
3. Calculates performance metrics
4. Saves results to JSON/CSV
```

### Phase 2: Analysis
```
1. Run analyze_backtest.py
2. Filters for profitable strategies
3. Identifies best risk-adjusted returns
4. Exports top performers
```

### Phase 3: Paper Trading (Optional)
```
1. Edit trade.py to use best strategies
2. Run in paper trading mode (SAFE)
3. Monitor for 2-4 weeks
4. Validate results before going live
```

### Phase 4: Live Trading (Caution!)
```
1. Start with small position size
2. Use strict risk management
3. Monitor closely
4. Scale gradually if profitable
```

---

## 📈 Key Metrics

### Performance
- **Win Rate** - % of winning trades (target: >55%)
- **Total Return** - Sum of all trade profits (target: >5%)
- **Sharpe Ratio** - Risk-adjusted return (target: >1.0)

### Risk
- **Max Drawdown** - Worst losing streak (target: <20%)
- **Profit Factor** - Gross profit / gross loss (target: >1.5)

---

## 🎯 Example Results

After running `python backtest.py`, you'll see output like:

```
✓ SMA Crossover       | WR: 62.2% | Return: +8.2% | Trades: 45 | Sharpe: 1.23 (3/20)
✓ EMA Crossover       | WR: 58.5% | Return: +6.5% | Trades: 52 | Sharpe: 0.98 (4/20)
✗ RSI                 | WR: 41.3% | Return: -2.1% | Trades: 62 | Sharpe: -0.45 (5/20)
```

Legend:
- ✓ = Profitable (WR > 50%)
- ✗ = Losing (WR < 50%)

Then run `python analyze_backtest.py` for detailed analysis:

```
🏆 TOP 10 STRATEGIES BY SHARPE RATIO
1.  SMA Crossover             1D    Sharpe:   1.45  Return:  +8.2%  WR: 62.2%  DD: -5.3%
2.  EMA Crossover             4H    Sharpe:   1.23  Return:  +6.5%  WR: 58.5%  DD: -7.1%
3.  MACD                       1D    Sharpe:   1.18  Return:  +5.9%  WR: 55.1%  DD: -8.2%
...
```

---

## 🔧 Customization

### Trade Different Cryptocurrencies
```python
# In backtest.py
config = BacktestConfig(symbol="ETH/USD")  # Ethereum
config = BacktestConfig(symbol="SOL/USD")  # Solana
```

### Use Different Time Period
```python
# In backtest.py
config = BacktestConfig(lookback_days=90)  # 3 months instead of 1 year
```

### Test Specific Strategies
```python
# In backtest.py
backtester.STRATEGIES = {
    'SMA Crossover': SMACrossover,
    'MACD': MACD,
    'RSI': RSI,
}
```

### Adjust Risk Parameters
```python
# In trade.py
config = TradingConfig(
    position_size_pct=10.0,     # Risk 10% per trade instead of 5%
    stop_loss_pct=3.0,          # 3% stop loss
    profit_target_pct=10.0,     # 10% profit target
)
```

---

## ⚠️ Important Notes

### Backtesting Reality
- Past performance ≠ future results
- Backtests assume perfect execution (no slippage)
- Real trading includes commissions and delays
- Overfitting is common (strategy fits noise, not signal)

### Cryptocurrency Trading
- 24/7 market (no gaps like stocks)
- High volatility (strategies may fail suddenly)
- Correlations change quickly
- Whale movements create false signals

### Risk Management
- **Never risk more than you can afford to lose**
- Start small in live trading (0.01 BTC equivalent)
- Use position sizing and stop losses
- Monitor closely for first 2-4 weeks
- Be prepared to stop if losing

---

## 🐛 Troubleshooting

### "Missing API credentials"
- Check `.env` file exists and is in project root
- Verify credentials are pasted correctly (no extra spaces)
- Reload terminal after editing `.env`

### "No data" when backtesting
- Check internet connection
- Verify API credentials work
- Check Alpaca API status: https://status.alpaca.markets/

### Backtests complete but results look weird
- Check historical data has enough bars
- Increase `lookback_days` for more data
- Verify symbol format (e.g., "BTC/USD" not "BTCUSD")

### Trades not executing in live mode
- Verify you're in the right API base URL (paper vs live)
- Check account has sufficient buying power
- Check network connectivity
- Review error logs in `crypto_trade.log`

---

## 📊 Analysis Tips

### Find Best Strategy-Timeframe Combo
```bash
python analyze_backtest.py
# Look for: High Sharpe Ratio + Positive Return + Low Drawdown
```

### Create Custom Filters
```python
import pandas as pd

df = pd.read_csv('backtest_results.csv')

# Conservative: High win rate, low risk
conservative = df[
    (df['win_rate'] >= 0.60) & 
    (df['max_drawdown'] > -0.10)
]

# Aggressive: High returns
aggressive = df[df['total_return'] > 0.10]

print(conservative)
```

### Compare Timeframes
```python
# Which timeframe works best overall?
by_tf = df.groupby('timeframe').agg({
    'sharpe_ratio': 'mean',
    'total_return': 'mean',
    'win_rate': 'mean'
})
print(by_tf)
```

---

## 🎓 Learn More

### Technical Analysis
- https://school.stockcharts.com/
- https://www.investopedia.com/

### Alpaca Documentation
- https://docs.alpaca.markets/
- https://github.com/alpacahq/alpaca-py

### Trading Theory
- Sharpe Ratio: https://en.wikipedia.org/wiki/Sharpe_ratio
- Drawdown: https://en.wikipedia.org/wiki/Drawdown_(economics)
- Position Sizing: https://en.wikipedia.org/wiki/Kelly_criterion

---

## 📞 Support

- **Alpaca API Issues**: https://community.alpaca.markets/
- **Python Issues**: https://stackoverflow.com/
- **Crypto Market Data**: https://www.coingecko.com/

---

## ⚖️ Disclaimer

**This is for educational purposes only.**

- Cryptocurrency trading is risky
- Past performance doesn't guarantee future results
- Backtesting results are theoretical (real trading differs)
- Always paper trade first before going live
- Only risk capital you can afford to lose

---

## 🚀 Next Steps

1. ✅ Install dependencies
2. ✅ Set up API credentials
3. ✅ Run backtests: `python backtest.py`
4. ✅ Analyze results: `python analyze_backtest.py`
5. ✅ Pick best 2-3 strategies
6. ✅ Test in paper trading for 2-4 weeks
7. ✅ Start live trading with micro positions
8. ✅ Scale gradually if profitable

**Happy trading! 🎯**
