"""
Alpaca Crypto Backtesting Framework
====================================

30 Trading Strategies × 4 Timeframes = 120 backtests

LONG Strategies (1-20):
  1. SMA Crossover
  2. EMA Crossover
  3. MACD
  4. RSI Oversold/Overbought
  5. Bollinger Bands
  6. Stochastic
  7. ATR Breakout
  8. Volume Surge
  9. Momentum
  10. Mean Reversion
  11. Grid Trading
  12. Ichimoku Cloud
  13. Keltner Channels
  14. VWAP
  15. ADX Trend
  16. Williams %R
  17. CCI
  18. Dual MA + Volume
  19. Turtle Trading
  20. Envelope Strategy

SHORT Strategies (21-30):
  21. Short SMA Crossover (Inverted)
  22. Short EMA Crossover (Inverted)
  23. Short RSI (Overbought Focus)
  24. Short Bollinger Bands (Upper Band)
  25. Short Stochastic (Overbought)
  26. Short Momentum (Negative)
  27. Short Breakdown (Support Break)
  28. Short Mean Reversion (High Extremes)
  29. Short Volatility (High VIX-like)
  30. Short Downtrend (Lower Highs & Lows)

Timeframes:
  - 1 minute
  - 5 minutes
  - 15 minutes
  - 30 minutes

Install:
  pip install alpaca-py pandas numpy ta-lib python-dotenv
"""

import os
import sys
import json
import csv
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Alpaca
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # talib is optional - strategies work fine without it

load_dotenv()

# Import symbols from trade.py to avoid duplication
try:
    from trade import TradeConfig
    DEFAULT_SYMBOLS = TradeConfig().symbols
except ImportError:
    DEFAULT_SYMBOLS = ["ETH/USD", "SHIB/USD", "DOGE/USD", "SOL/USD"]

# Load paper trading credentials (for backtest)
API_KEY = os.getenv("APCA_API_KEY_ID_PAPER")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY_PAPER")

if not API_KEY or not SECRET_KEY:
    print("[ERROR] Missing API credentials in .env file")
    sys.exit(1)

# Set environment for CryptoHistoricalDataClient
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BacktestConfig:
    symbols: List[str] = None
    initial_cash: float = 10000.0
    commission: float = 0.001  # 0.1% per trade
    timeframes: List[TimeFrame] = None
    lookback_days: int = 90  # 3 months (faster than 365 days)
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = DEFAULT_SYMBOLS
        if self.timeframes is None:
            self.timeframes = [
                TimeFrame(amount=1, unit=TimeFrameUnit.Minute),     # 1 minute
                TimeFrame(amount=5, unit=TimeFrameUnit.Minute),     # 5 minutes
                TimeFrame(amount=15, unit=TimeFrameUnit.Minute),    # 15 minutes
                TimeFrame(amount=30, unit=TimeFrameUnit.Minute),    # 30 minutes
            ]


# ============================================================================
# Strategy Base Class
# ============================================================================

class Strategy(ABC):
    """Base strategy class"""
    
    def __init__(self, name: str, df: pd.DataFrame):
        self.name = name
        self.df = df.copy()
        self.signals = pd.Series(0, index=df.index)  # -1=SELL, 0=HOLD, 1=BUY
    
    @abstractmethod
    def generate_signals(self) -> pd.Series:
        """Generate buy/sell signals. Return series with -1/0/1"""
        pass
    
    def _add_indicators(self) -> None:
        """Subclass overrides to add their indicators"""
        pass


# ============================================================================
# 20 Strategies
# ============================================================================

class SMACrossover(Strategy):
    """1. SMA Crossover"""
    def generate_signals(self) -> pd.Series:
        fast = self.df['close'].rolling(10).mean()
        slow = self.df['close'].rolling(30).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals.fillna(0).astype(int)


class EMACrossover(Strategy):
    """2. EMA Crossover"""
    def generate_signals(self) -> pd.Series:
        fast = self.df['close'].ewm(span=12).mean()
        slow = self.df['close'].ewm(span=26).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[fast > slow] = 1
        signals[fast < slow] = -1
        return signals.fillna(0).astype(int)


class MACD(Strategy):
    """3. MACD"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close'].values
        
        if TALIB_AVAILABLE:
            macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
            # Convert to Series with correct index for proper alignment
            hist_series = pd.Series(hist, index=self.df.index)
            signals = pd.Series(0, index=self.df.index)
            signals[hist_series > 0] = 1
            signals[hist_series < 0] = -1
        else:
            # Manual MACD
            close_series = pd.Series(close, index=self.df.index)
            ema12 = close_series.ewm(span=12).mean()
            ema26 = close_series.ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            hist = macd - signal
            
            signals = pd.Series(0, index=self.df.index)
            signals[hist > 0] = 1
            signals[hist < 0] = -1
        
        return signals.fillna(0).astype(int)


class RSI(Strategy):
    """4. RSI Oversold/Overbought"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close'].values
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=14)
            # Convert to Series with correct index for proper alignment
            rsi_series = pd.Series(rsi, index=self.df.index)
        else:
            close_series = pd.Series(close, index=self.df.index)
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=self.df.index)
        signals[rsi_series < 30] = 1  # Oversold = buy
        signals[rsi_series > 70] = -1  # Overbought = sell
        
        return signals.fillna(0).astype(int)


class BollingerBands(Strategy):
    """5. Bollinger Bands"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close']
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        signals = pd.Series(0, index=self.df.index)
        signals[close < lower] = 1  # Buy at lower band
        signals[close > upper] = -1  # Sell at upper band
        
        return signals.fillna(0).astype(int)


class Stochastic(Strategy):
    """6. Stochastic Oscillator"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close']
        low = self.df['low'].rolling(14).min()
        high = self.df['high'].rolling(14).max()
        
        k = 100 * (close - low) / (high - low)
        d = k.rolling(3).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[k < 20] = 1  # Oversold
        signals[k > 80] = -1  # Overbought
        
        return signals.fillna(0).astype(int)


class ATRBreakout(Strategy):
    """7. ATR Breakout"""
    def generate_signals(self) -> pd.Series:
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        atr = pd.Series(tr).rolling(14).mean()
        
        prev_close = close.shift(1)
        signals = pd.Series(0, index=self.df.index)
        signals[close > prev_close + atr] = 1  # Buy on breakout
        signals[close < prev_close - atr] = -1  # Sell on breakdown
        
        return signals.fillna(0).astype(int)


class VolumeSurge(Strategy):
    """8. Volume Surge"""
    def generate_signals(self) -> pd.Series:
        volume = self.df['volume']
        vol_ma = volume.rolling(20).mean()
        price_ma = self.df['close'].rolling(20).mean()
        
        vol_surge = volume > vol_ma * 1.5
        price_up = self.df['close'] > price_ma
        
        signals = pd.Series(0, index=self.df.index)
        signals[vol_surge & price_up] = 1
        signals[vol_surge & ~price_up] = -1
        
        return signals.fillna(0).astype(int)


class Momentum(Strategy):
    """9. Momentum"""
    def generate_signals(self) -> pd.Series:
        momentum = self.df['close'].diff(10)
        
        signals = pd.Series(0, index=self.df.index)
        signals[momentum > 0] = 1
        signals[momentum < 0] = -1
        
        return signals.fillna(0).astype(int)


class MeanReversion(Strategy):
    """10. Mean Reversion"""
    def generate_signals(self) -> pd.Series:
        sma = self.df['close'].rolling(50).mean()
        std = self.df['close'].rolling(50).std()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] < sma - std] = 1  # Buy when below mean
        signals[self.df['close'] > sma + std] = -1  # Sell when above mean
        
        return signals.fillna(0).astype(int)


class GridTrading(Strategy):
    """11. Grid Trading"""
    def generate_signals(self) -> pd.Series:
        # Buy when price drops, sell when price rises
        min_50 = self.df['low'].rolling(50).min()
        max_50 = self.df['high'].rolling(50).max()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] <= (min_50 + max_50) / 2] = 1  # Buy in lower half
        signals[self.df['close'] >= (min_50 + max_50) / 2] = -1  # Sell in upper half
        
        return signals.fillna(0).astype(int)


class Ichimoku(Strategy):
    """12. Ichimoku Cloud"""
    def generate_signals(self) -> pd.Series:
        high9 = self.df['high'].rolling(9).max()
        low9 = self.df['low'].rolling(9).min()
        tenkan = (high9 + low9) / 2
        
        high26 = self.df['high'].rolling(26).max()
        low26 = self.df['low'].rolling(26).min()
        kijun = (high26 + low26) / 2
        
        signals = pd.Series(0, index=self.df.index)
        signals[tenkan > kijun] = 1
        signals[tenkan < kijun] = -1
        
        return signals.fillna(0).astype(int)


class KeltnerChannels(Strategy):
    """13. Keltner Channels"""
    def generate_signals(self) -> pd.Series:
        ema = self.df['close'].ewm(span=20).mean()
        atr = pd.Series(
            np.maximum(
                self.df['high'] - self.df['low'],
                np.maximum(
                    abs(self.df['high'] - self.df['close'].shift()),
                    abs(self.df['low'] - self.df['close'].shift())
                )
            )
        ).rolling(10).mean()
        
        upper = ema + 2 * atr
        lower = ema - 2 * atr
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] < lower] = 1
        signals[self.df['close'] > upper] = -1
        
        return signals.fillna(0).astype(int)


class VWAP(Strategy):
    """14. VWAP"""
    def generate_signals(self) -> pd.Series:
        vwap = (self.df['close'] * self.df['volume']).rolling(20).sum() / self.df['volume'].rolling(20).sum()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] > vwap] = 1
        signals[self.df['close'] < vwap] = -1
        
        return signals.fillna(0).astype(int)


class ADXTrend(Strategy):
    """15. ADX Trend"""
    def generate_signals(self) -> pd.Series:
        high = self.df['high']
        low = self.df['low']
        close = self.df['close']
        
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift()), abs(low - close.shift())))
        atr = pd.Series(tr).rolling(14).mean()
        
        plus_di = 100 * plus_dm.rolling(14).sum() / (atr * 14)
        minus_di = 100 * minus_dm.rolling(14).sum() / (atr * 14)
        
        signals = pd.Series(0, index=self.df.index)
        signals[plus_di > minus_di] = 1
        signals[plus_di < minus_di] = -1
        
        return signals.fillna(0).astype(int)


class WilliamsR(Strategy):
    """16. Williams %R"""
    def generate_signals(self) -> pd.Series:
        high = self.df['high'].rolling(14).max()
        low = self.df['low'].rolling(14).min()
        close = self.df['close']
        
        wr = -100 * (high - close) / (high - low)
        
        signals = pd.Series(0, index=self.df.index)
        signals[wr < -80] = 1  # Oversold
        signals[wr > -20] = -1  # Overbought
        
        return signals.fillna(0).astype(int)


class CCI(Strategy):
    """17. CCI (Commodity Channel Index)"""
    def generate_signals(self) -> pd.Series:
        tp = (self.df['high'] + self.df['low'] + self.df['close']) / 3
        sma = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        
        cci = (tp - sma) / (0.015 * mad)
        
        signals = pd.Series(0, index=self.df.index)
        signals[cci < -100] = 1
        signals[cci > 100] = -1
        
        return signals.fillna(0).astype(int)


class DualMAVolume(Strategy):
    """18. Dual MA + Volume"""
    def generate_signals(self) -> pd.Series:
        fast_ma = self.df['close'].ewm(span=12).mean()
        slow_ma = self.df['close'].ewm(span=26).mean()
        vol_ma = self.df['volume'].rolling(20).mean()
        
        signals = pd.Series(0, index=self.df.index)
        buy = (fast_ma > slow_ma) & (self.df['volume'] > vol_ma)
        sell = (fast_ma < slow_ma) & (self.df['volume'] > vol_ma)
        
        signals[buy] = 1
        signals[sell] = -1
        
        return signals.fillna(0).astype(int)


class TurtleTrading(Strategy):
    """19. Turtle Trading"""
    def generate_signals(self) -> pd.Series:
        high20 = self.df['high'].rolling(20).max()
        low20 = self.df['low'].rolling(20).min()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] > high20.shift(1)] = 1  # Breakout up
        signals[self.df['close'] < low20.shift(1)] = -1  # Breakout down
        
        return signals.fillna(0).astype(int)


class Envelope(Strategy):
    """20. Envelope Strategy"""
    def generate_signals(self) -> pd.Series:
        sma = self.df['close'].rolling(30).mean()
        envelope_pct = 0.05  # 5% envelope
        
        upper = sma * (1 + envelope_pct)
        lower = sma * (1 - envelope_pct)
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] < lower] = 1
        signals[self.df['close'] > upper] = -1
        
        return signals.fillna(0).astype(int)


# ============================================================================
# 10 SHORT-FOCUSED STRATEGIES
# ============================================================================

class ShortSMA(Strategy):
    """21. Short SMA Crossover (Inverted)"""
    def generate_signals(self) -> pd.Series:
        fast = self.df['close'].rolling(10).mean()
        slow = self.df['close'].rolling(30).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[fast < slow] = 1  # Go short when fast MA below slow MA
        signals[fast > slow] = -1  # Cover when fast MA above slow MA
        return signals.fillna(0).astype(int)


class ShortEMA(Strategy):
    """22. Short EMA Crossover (Inverted)"""
    def generate_signals(self) -> pd.Series:
        fast = self.df['close'].ewm(span=12).mean()
        slow = self.df['close'].ewm(span=26).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[fast < slow] = 1  # Go short when fast EMA below slow EMA
        signals[fast > slow] = -1  # Cover when fast EMA above slow EMA
        return signals.fillna(0).astype(int)


class ShortRSI(Strategy):
    """23. Short RSI (Overbought Focus)"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close'].values
        
        if TALIB_AVAILABLE:
            rsi = talib.RSI(close, timeperiod=14)
            rsi_series = pd.Series(rsi, index=self.df.index)
        else:
            close_series = pd.Series(close, index=self.df.index)
            delta = close_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
        
        signals = pd.Series(0, index=self.df.index)
        signals[rsi_series > 70] = 1  # Go short on overbought
        signals[rsi_series < 30] = -1  # Cover on oversold
        return signals.fillna(0).astype(int)


class ShortBollinger(Strategy):
    """24. Short Bollinger Bands (Upper Band)"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close']
        sma = close.rolling(20).mean()
        std = close.rolling(20).std()
        
        upper = sma + 2 * std
        lower = sma - 2 * std
        
        signals = pd.Series(0, index=self.df.index)
        signals[close > upper] = 1  # Short at upper band
        signals[close < lower] = -1  # Cover at lower band
        return signals.fillna(0).astype(int)


class ShortStochastic(Strategy):
    """25. Short Stochastic (Overbought)"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close']
        low = self.df['low'].rolling(14).min()
        high = self.df['high'].rolling(14).max()
        
        k = 100 * (close - low) / (high - low)
        d = k.rolling(3).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[k > 80] = 1  # Go short on overbought
        signals[k < 20] = -1  # Cover on oversold
        return signals.fillna(0).astype(int)


class ShortMomentum(Strategy):
    """26. Short Momentum (Negative)"""
    def generate_signals(self) -> pd.Series:
        momentum = self.df['close'].diff(10)
        
        signals = pd.Series(0, index=self.df.index)
        signals[momentum < 0] = 1  # Go short when momentum negative
        signals[momentum > 0] = -1  # Cover when momentum positive
        return signals.fillna(0).astype(int)


class ShortBreakdown(Strategy):
    """27. Short Breakdown (Support Break)"""
    def generate_signals(self) -> pd.Series:
        low20 = self.df['low'].rolling(20).min()
        high20 = self.df['high'].rolling(20).max()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] < low20.shift(1)] = 1  # Short on breakdown
        signals[self.df['close'] > high20.shift(1)] = -1  # Cover on recovery
        return signals.fillna(0).astype(int)


class ShortMeanReversion(Strategy):
    """28. Short Mean Reversion (High Extremes)"""
    def generate_signals(self) -> pd.Series:
        sma = self.df['close'].rolling(50).mean()
        std = self.df['close'].rolling(50).std()
        
        signals = pd.Series(0, index=self.df.index)
        signals[self.df['close'] > sma + std] = 1  # Short when extremely high
        signals[self.df['close'] < sma - std] = -1  # Cover when extremely low
        return signals.fillna(0).astype(int)


class ShortVolatility(Strategy):
    """29. Short Volatility (High VIX-like)"""
    def generate_signals(self) -> pd.Series:
        # High volatility periods tend to reverse
        volatility = self.df['close'].rolling(14).std()
        vol_ma = volatility.rolling(20).mean()
        
        signals = pd.Series(0, index=self.df.index)
        signals[volatility > vol_ma * 1.5] = 1  # Go short on high volatility
        signals[volatility < vol_ma * 0.8] = -1  # Cover on low volatility
        return signals.fillna(0).astype(int)


class ShortDowntrend(Strategy):
    """30. Short Downtrend (Lower Highs & Lows)"""
    def generate_signals(self) -> pd.Series:
        close = self.df['close']
        close_shift1 = close.shift(1)
        close_shift2 = close.shift(2)
        
        # Downtrend: each close lower than previous closes
        downtrend = (close < close_shift1) & (close_shift1 < close_shift2)
        uptrend = (close > close_shift1) & (close_shift1 > close_shift2)
        
        signals = pd.Series(0, index=self.df.index)
        signals[downtrend] = 1  # Go short in downtrend
        signals[uptrend] = -1  # Cover in uptrend
        return signals.fillna(0).astype(int)


# ============================================================================
# Backtester
# ============================================================================

@dataclass
class BacktestResult:
    strategy_name: str
    timeframe: str
    symbol: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    cumulative_return_pct: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade: float
    best_trade: float
    worst_trade: float
    profit_factor: float
    
    def calculate_combo_score(self) -> float:
        """
        Calculate weighted combo score from multiple metrics
        Weights: Profit Factor 40%, Win Rate 30%, Return 20%, Sharpe 10%
        """
        # Normalize metrics to 0-100 scale
        
        # 1. Profit Factor (higher is better, cap at 100)
        pf_score = min(self.profit_factor * 50, 100)  # 2.0 PF = 100
        
        # 2. Win Rate (already 0-100)
        wr_score = self.win_rate * 100
        
        # 3. Return (higher is better, normalize to 0-100)
        # Assume returns range from -50% to +50%, cap at 100
        ret_score = max(0, min((self.total_return * 100 + 50) / 1, 100))
        
        # 4. Sharpe Ratio (higher is better, normalize)
        # Sharpe ranges from -5 to +5, normalize to 0-100
        sharpe_score = max(0, min((self.sharpe_ratio + 5) / 10 * 100, 100))
        
        # Weighted combo score
        combo = (
            pf_score * 0.40 +
            wr_score * 0.30 +
            ret_score * 0.20 +
            sharpe_score * 0.10
        )
        
        return combo


class Backtester:
    """Backtest strategies"""
    
    STRATEGIES = {
        'SMA Crossover': SMACrossover,
        'EMA Crossover': EMACrossover,
        'MACD': MACD,
        'RSI': RSI,
        'Bollinger Bands': BollingerBands,
        'Stochastic': Stochastic,
        'ATR Breakout': ATRBreakout,
        'Volume Surge': VolumeSurge,
        'Short SMA': ShortSMA,
        'Short RSI': ShortRSI,
        'Short Bollinger': ShortBollinger,
        'Short Momentum': ShortMomentum,
        'Short Downtrend': ShortDowntrend,
        'Short EMA': ShortEMA,
        'Short Stochastic': ShortStochastic,
    }
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.data_client = CryptoHistoricalDataClient()
        self.results: List[BacktestResult] = []
    
    def fetch_data(self, symbol: str, timeframe: TimeFrame, days: int) -> pd.DataFrame:
        """Fetch historical crypto data"""
        try:
            from datetime import datetime, timedelta, timezone
            
            # Calculate dates
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days)
            
            print(f"  Requesting: {symbol}, {timeframe.amount}{timeframe.unit}, {days} days")
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start_date,
                end=end_date,
                limit=10000,  # API max
            )
            bars = self.data_client.get_crypto_bars(request)
            
            # BarSet object has .df property that includes all symbols
            if bars and hasattr(bars, 'df'):
                df = bars.df
                if not df.empty:
                    # Reset index to remove symbol level if present
                    if 'symbol' in df.index.names:
                        df = df.reset_index(level='symbol', drop=True)
                    df = df.sort_index()
                    print(f"  Got {len(df)} bars")
                    return df
                else:
                    print(f"  Empty dataframe returned")
            else:
                print(f"  No data in response")
        except Exception as e:
            print(f"  [ERROR] Failed to fetch data: {e}")
            import traceback
            traceback.print_exc()
        
        return pd.DataFrame()
    
    def run_backtest(
        self,
        symbol: str,
        strategy_class,
        strategy_name: str,
        df: pd.DataFrame,
        timeframe_str: str,
    ) -> Optional[BacktestResult]:
        """Run backtest for single strategy"""
        
        if df.empty or len(df) < 50:
            return None
        
        try:
            strategy = strategy_class(strategy_name, df)
            signals = strategy.generate_signals()
            
            # Simulate trades
            position = 0
            trades = []
            entry_price = 0
            
            for i in range(1, len(df)):
                signal = signals.iloc[i]
                price = df['close'].iloc[i]
                
                # Entry
                if signal == 1 and position == 0:
                    entry_price = price
                    position = 1
                
                # Exit
                elif signal == -1 and position == 1:
                    pnl = (price - entry_price) / entry_price
                    trades.append(pnl)
                    position = 0
            
            if not trades:
                return None
            
            # Calculate metrics
            trades_arr = np.array(trades)
            winning = np.sum(trades_arr > 0)
            losing = np.sum(trades_arr < 0)
            total = len(trades)
            win_rate = winning / total if total > 0 else 0
            
            total_return = np.sum(trades_arr)
            avg_trade = np.mean(trades_arr)
            best_trade = np.max(trades_arr)
            worst_trade = np.min(trades_arr)
            
            # Sharpe ratio
            if len(trades_arr) > 1 and np.std(trades_arr) > 0:
                sharpe = np.mean(trades_arr) / np.std(trades_arr) * np.sqrt(252)
            else:
                sharpe = 0.0
            
            # Max drawdown
            cumul = np.cumprod(1 + trades_arr)
            running_max = np.maximum.accumulate(cumul)
            drawdown = (cumul - running_max) / running_max
            max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
            
            # Profit factor
            gross_profit = np.sum(trades_arr[trades_arr > 0])
            gross_loss = abs(np.sum(trades_arr[trades_arr < 0]))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            return BacktestResult(
                strategy_name=strategy_name,
                timeframe=timeframe_str,
                symbol=symbol,
                total_trades=total,
                winning_trades=winning,
                losing_trades=losing,
                win_rate=win_rate,
                total_return=total_return,
                cumulative_return_pct=total_return * 100,
                sharpe_ratio=sharpe,
                max_drawdown=max_drawdown,
                avg_trade=avg_trade,
                best_trade=best_trade,
                worst_trade=worst_trade,
                profit_factor=profit_factor,
            )
        
        except Exception as e:
            print(f"[ERROR] Backtest failed for {strategy_name}: {e}")
            return None
    
    def run_all_backtests(self) -> List[BacktestResult]:
        """Run all strategies × symbols × timeframes"""
        
        timeframe_map = {
            TimeFrame(amount=1, unit=TimeFrameUnit.Minute): "1Min",
            TimeFrame(amount=5, unit=TimeFrameUnit.Minute): "5Min",
            TimeFrame(amount=15, unit=TimeFrameUnit.Minute): "15Min",
            TimeFrame(amount=30, unit=TimeFrameUnit.Minute): "30Min",
        }
        
        total = len(self.STRATEGIES) * len(self.config.timeframes) * len(self.config.symbols)
        completed = 0
        
        for symbol in self.config.symbols:
            print(f"\n{'='*80}")
            print(f"SYMBOL: {symbol}")
            print(f"{'='*80}")
            
            for timeframe in self.config.timeframes:
                timeframe_str = timeframe_map.get(timeframe, str(timeframe.value))
                print(f"\n[FETCH] {symbol} {timeframe_str}...", end=" ", flush=True)
                
                df = self.fetch_data(symbol, timeframe, self.config.lookback_days)
                if df.empty:
                    print(f"[SKIP]")
                    completed += len(self.STRATEGIES)
                    continue
                
                print(f"OK ({len(df)} bars)")
                print(f"{'-'*80}")
                
                for strategy_name, strategy_class in self.STRATEGIES.items():
                    completed += 1
                    pct = completed / total * 100
                    bar_length = 20
                    filled = int(bar_length * completed / total)
                    bar = "=" * filled + "-" * (bar_length - filled)
                    
                    result = self.run_backtest(
                        symbol,
                        strategy_class,
                        strategy_name,
                        df,
                        timeframe_str,
                    )
                    
                    if result:
                        self.results.append(result)
                        status = "[+]" if result.win_rate > 0.5 else "[-]"
                        print(
                            f"  {status} {strategy_name:25} WR:{result.win_rate:>5.1%} "
                            f"Return:{result.total_return:>+6.2%} "
                            f"Trades:{result.total_trades:>2d} "
                            f"[{bar}] {pct:>5.1f}%"
                        )
        
        return self.results
    
    def save_results(self, filename: str = "backtest_results.json") -> None:
        """Save results to JSON"""
        import numpy as np
        
        # Convert numpy types to Python native types
        def convert_types(obj):
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_types(item) for item in obj]
            return obj
        
        data = [convert_types(asdict(r)) for r in self.results]
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\n[+] Saved {len(self.results)} results to {filename}")
    
    def export_csv(self, filename: str = "backtest_results.csv") -> None:
        """Export to CSV for analysis"""
        if not self.results:
            return
        
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=asdict(self.results[0]).keys())
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        print(f"[+] Exported to {filename}")
    
    def print_summary(self) -> None:
        """Print summary statistics"""
        if not self.results:
            print("No results to summarize")
            return
        
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        print("\n" + "="*80)
        print("BACKTEST SUMMARY")
        print("="*80)
        
        # Best by strategy
        print("\n[STRATEGIES] BEST (by Return):")
        best_by_strat = df.groupby('strategy_name')['total_return'].mean().sort_values(ascending=False)
        for strat, ret in best_by_strat.head(5).items():
            print(f"  {strat:25} {ret:+.2%}")
        
        # Best by timeframe
        print("\n[TIMEFRAMES] BEST (by Sharpe Ratio):")
        best_by_tf = df.groupby('timeframe')['sharpe_ratio'].mean().sort_values(ascending=False)
        for tf, sharpe in best_by_tf.items():
            print(f"  {tf:10} {sharpe:.2f}")
        
        # Overall stats
        print("\n[STATISTICS] OVERALL:")
        print(f"  Total Backtests: {len(df)}")
        print(f"  Avg Win Rate: {df['win_rate'].mean():.1%}")
        print(f"  Avg Return: {df['total_return'].mean():+.2%}")
        print(f"  Avg Sharpe: {df['sharpe_ratio'].mean():.2f}")
        print(f"  Avg Max Drawdown: {df['max_drawdown'].mean():.2%}")
        
        # Profitable combos
        print("\n💰 PROFITABLE COMBINATIONS (return > 0):")
        profitable = df[df['total_return'] > 0].sort_values('sharpe_ratio', ascending=False)
        print(f"  {len(profitable)}/{len(df)} combinations profitable")
        for _, row in profitable.head(5).iterrows():
            print(
                f"  {row['strategy_name']:25} {row['timeframe']:3} "
                f"Return: {row['total_return']:+.2%} Sharpe: {row['sharpe_ratio']:.2f}"
            )


# ============================================================================
# Main
# ============================================================================

def main():
    import time
    
    print("="*80)
    print("ALPACA CRYPTO BACKTESTING FRAMEWORK")
    print("="*80)
    
    config = BacktestConfig(lookback_days=90)
    
    backtester = Backtester(config)
    
    total_backtests = len(backtester.STRATEGIES) * len(config.timeframes) * len(config.symbols)
    est_time = total_backtests * 0.5  # ~0.5s per backtest estimate
    
    print(f"\nSymbols: {', '.join(config.symbols)}")
    print(f"Lookback: {config.lookback_days} days")
    print(f"Strategies: {len(backtester.STRATEGIES)}")
    print(f"Timeframes: {len(config.timeframes)}")
    print(f"Total Backtests: {total_backtests}")
    print(f"Estimated Time: ~{est_time/60:.1f} minutes\n")
    
    # Run backtests
    start_time = time.time()
    print("[START] Beginning backtests...\n")
    results = backtester.run_all_backtests()
    elapsed = time.time() - start_time
    print(f"\n[DONE] Completed in {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    
    # Save & export
    backtester.save_results()
    backtester.export_csv()
    
    # Summary
    backtester.print_summary()


if __name__ == "__main__":
    main()
