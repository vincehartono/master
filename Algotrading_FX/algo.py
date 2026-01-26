"""
IBKR Automated Strategy Optimization & Trading System

What this script gives you (cleaned + rewritten):
1) Connect to IBKR (TWS/IB Gateway) via ib_insync
2) Pull historical bars for FX (or stocks) from IBKR
3) Backtest a set of Backtrader strategies across:
      strategies x timeframes x pairs
   and rank results by Sharpe (primary) then total return
4) Live trading loop with profit target / stop loss
5) Trade + PnL CSV logging

Notes
- Backtrader is optional. If it's not installed, optimization/backtesting is disabled.
- pandas_ta is optional. (You can extend strategies with it later.)
- This is a template. Always paper trade first.

Install:
  pip install ib_insync pandas numpy
  pip install backtrader matplotlib   # optional (for backtests)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import csv
import json
import time
from datetime import datetime
import math

import numpy as np
import pandas as pd
from ib_insync import IB, Forex, Stock, util, MarketOrder, LimitOrder

# ----------------------------
# Optional dependencies
# ----------------------------
try:
    import backtrader as bt

    BACKTRADER_AVAILABLE = True
except Exception:
    BACKTRADER_AVAILABLE = False
    bt = None  # type: ignore

try:
    import pandas_ta as pta  # noqa: F401

    PANDAS_TA_AVAILABLE = True
except Exception:
    PANDAS_TA_AVAILABLE = False


# ============================================================================
# Config objects
# ============================================================================

@dataclass(frozen=True)
class IBKRConnectionConfig:
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 1


@dataclass(frozen=True)
class StrategyConfig:
    strategy: str          # e.g. "SMA"
    pair: str              # e.g. "AUD/USD"
    timeframe: str         # e.g. "1 hour"


@dataclass(frozen=True)
class LiveRiskConfig:
    profit_target: float = 10.0  # dollars
    stop_loss: float = 5.0       # dollars
    quantity: int = 1000         # Legacy fixed size (used if balance_risk_pct <= 0)
    balance_risk_pct: float = 1.0  # % of account balance to allocate per trade (not risk %, just notional)
    min_quantity: int = 1000       # Minimum trade size in units (e.g. 1000 ~ micro lot)
    check_interval: int = 60     # seconds between checks


# ============================================================================
# Utilities
# ============================================================================

def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


LOG_FILE = Path("algo.log")


def log_message(msg: str) -> None:
    """
    Append a timestamped message to algo.log (in addition to normal prints where used).
    """
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(f"{now_ts()} {msg}\n")
    except Exception:
        pass


# ============================================================================
# IBKR Trader (ib_insync)
# ============================================================================

class IBKRAlgoTrader:
    def __init__(self, cfg: IBKRConnectionConfig):
        self.cfg = cfg
        self.ib = IB()

    def connect(self) -> bool:
        try:
            self.ib.connect(self.cfg.host, self.cfg.port, clientId=self.cfg.client_id)
            msg = f"[OK] Connected to IBKR @ {self.cfg.host}:{self.cfg.port}"
            print(msg)
            log_message(msg)
            try:
                accounts = self.ib.managedAccounts()
                info = f"[INFO] Managed accounts: {accounts} (ensure this is your PAPER or LIVE account as intended)"
                print(info)
                log_message(info)
            except Exception:
                pass
            return True
        except Exception as e:
            msg = f"[ERROR] IBKR connection failed: {e}"
            print(msg)
            log_message(msg)
            return False

    def disconnect(self) -> None:
        try:
            self.ib.disconnect()
        except Exception:
            pass

    def get_contract(
        self,
        symbol: str,
        sec_type: str = "CASH",
        exchange: str = "IDEALPRO",
        currency: str = "USD",
    ):
        """
        sec_type:
          - 'CASH' for FX (Forex)
          - 'STK'  for Stock
        """
        if sec_type.upper() == "STK":
            return Stock(symbol, exchange, currency)
        if sec_type.upper() == "CASH":
            # For ib_insync Forex, pass "EURUSD" or Forex("EURUSD")
            return Forex(f"{symbol}{currency}".replace("/", ""))
        raise ValueError(f"Unsupported sec_type: {sec_type}")

    def get_historical_data(
        self,
        contract,
        duration: str = "1 M",
        bar_size: str = "1 hour",
        what_to_show: str = "MIDPOINT",
        use_rth: bool = False,
    ) -> pd.DataFrame:
        """
        Returns a DataFrame with datetime index and OHLCV columns if available.
        """
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        df = util.df(bars)
        if df is None or df.empty:
            return pd.DataFrame()

        # Normalize index to datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # Backtrader typically expects columns: open/high/low/close/volume
        # IB sometimes provides volume=0 for FX midpoint
        for c in ["open", "high", "low", "close", "volume"]:
            if c not in df.columns:
                df[c] = np.nan
        df = df[["open", "high", "low", "close", "volume"]].copy()
        df.index = pd.to_datetime(df.index)
        return df

    def place_market_order(self, contract, action: str, quantity: int):
        """
        action: 'BUY' or 'SELL'
        Uses GTC (Good Till Cancelled) to prevent auto-cancellation at market close
        """
        order = MarketOrder(action.upper(), quantity)
        order.tif = 'GTC'  # Good Till Cancelled (don't expire at market close)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def place_limit_order(self, contract, action: str, quantity: int, limit_price: float):
        """
        Limit order helper to reduce use of pure market orders.
        """
        order = LimitOrder(action.upper(), quantity, limit_price)
        trade = self.ib.placeOrder(contract, order)
        return trade

    def get_positions(self):
        return self.ib.positions()

    def get_account_value(self) -> float:
        """
        Uses NetLiquidation from account summary (cached for 10 seconds).
        Reduces excessive account summary queries that cause Error 322.
        """
        now = time.time()
        # Return cached value if available and < 10 seconds old
        if hasattr(self, '_account_cache') and hasattr(self, '_account_cache_time'):
            if now - self._account_cache_time < 10:
                return self._account_cache
        
        try:
            summary = self.ib.accountSummary()
            for item in summary:
                if getattr(item, "tag", "") == "NetLiquidation":
                    value = safe_float(item.value, 0.0)
                    self._account_cache = value
                    self._account_cache_time = now
                    return value
        except Exception as e:
            if "Maximum number" in str(e):
                # Rate limited, return cached value if available
                if hasattr(self, '_account_cache'):
                    return self._account_cache
        
        return 0.0


# ============================================================================
# Logging
# ============================================================================

class TradeLogger:
    def __init__(self, trade_log_file: str = "trade_log.csv", pnl_log_file: str = "pnl_log.csv"):
        self.trade_log_file = Path(trade_log_file)
        self.pnl_log_file = Path(pnl_log_file)
        self._init_trade_log()
        self._init_pnl_log()

    def _init_trade_log(self) -> None:
        if not self.trade_log_file.exists():
            with self.trade_log_file.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "Timestamp",
                    "Strategy",
                    "Pair",
                    "Timeframe",
                    "Action",
                    "Quantity",
                    "Price",
                    "Order_ID",
                    "Fee",
                    "Status",
                ])

    def _init_pnl_log(self) -> None:
        if not self.pnl_log_file.exists():
            with self.pnl_log_file.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "Timestamp", "Cycle", "Strategy", "Pair", "Timeframe",
                    "Entry_Time", "Exit_Time", "Entry_Price", "Exit_Price",
                    "Quantity", "Gross_PnL", "Commission", "Net_PnL",
                    "Account_Balance", "Cumulative_PnL", "Status"
                ])

    def log_trade(
        self,
        strategy: str,
        pair: str,
        timeframe: str,
        action: str,
        quantity: int,
        price: Optional[float],
        order_id: Any,
        fee: float = 0.0,
        status: str = "Executed",
    ) -> None:
        with self.trade_log_file.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now_ts(),
                strategy,
                pair,
                timeframe,
                action,
                quantity,
                f"{price:.5f}" if price is not None else "N/A",
                order_id if order_id is not None else "N/A",
                f"{fee:.5f}",
                status,
            ])
        print(f"[LOG] Trade logged: {action} {quantity} {pair}")

    def log_pnl(
        self,
        cycle: int,
        strategy: str,
        pair: str,
        timeframe: str,
        entry_time: str,
        exit_time: str,
        entry_price: Optional[float],
        exit_price: Optional[float],
        quantity: int,
        gross_pnl: float,
        commission: float,
        net_pnl: float,
        account_balance: float,
        cumulative_pnl: float,
        status: str,
    ) -> None:
        with self.pnl_log_file.open("a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                now_ts(),
                cycle,
                strategy,
                pair,
                timeframe,
                entry_time,
                exit_time,
                f"{entry_price:.5f}" if entry_price is not None else "N/A",
                f"{exit_price:.5f}" if exit_price is not None else "N/A",
                quantity,
                f"{gross_pnl:.2f}",
                f"{commission:.2f}",
                f"{net_pnl:.2f}",
                f"{account_balance:.2f}",
                f"{cumulative_pnl:.2f}",
                status,
            ])
        print(f"[LOG] P&L logged: {status} | Net: {net_pnl:+.2f}")

    def get_summary(self) -> Optional[Dict[str, float]]:
        try:
            df = pd.read_csv(self.pnl_log_file)
            if df.empty:
                return None
            net = pd.to_numeric(df["Net_PnL"], errors="coerce").fillna(0.0)
            wins = net[net > 0]
            losses = net[net < 0]
            return {
                "total_trades": float(len(df)),
                "winning_trades": float(len(wins)),
                "losing_trades": float(len(losses)),
                "total_pnl": float(net.sum()),
                "avg_win": float(wins.mean()) if len(wins) else 0.0,
                "avg_loss": float(losses.mean()) if len(losses) else 0.0,
                "win_rate": float(len(wins) / len(df) * 100.0) if len(df) else 0.0,
            }
        except Exception as e:
            print(f"Error getting summary: {e}")
            return None


# ============================================================================
# Backtrader Strategies (examples)
# ============================================================================

# Backtrader Strategies (examples)
# ============================================================================
if BACKTRADER_AVAILABLE:

    class SMAStrategy(bt.Strategy):
        """Strategy 1: Simple Moving Average Crossover"""
        params = (("fast", 10), ("slow", 30), ("printlog", False))

        def __init__(self):
            self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

        def next(self):
            if self.crossover > 0 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.position:
                self.close()

    class EMAStrategy(bt.Strategy):
        """Strategy 2: Exponential Moving Average Crossover"""
        params = (("fast", 12), ("slow", 26), ("printlog", False))

        def __init__(self):
            self.fast_ema = bt.indicators.EMA(self.data.close, period=self.p.fast)
            self.slow_ema = bt.indicators.EMA(self.data.close, period=self.p.slow)
            self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)

        def next(self):
            if self.crossover > 0 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.position:
                self.close()

    class MACDStrategy(bt.Strategy):
        """Strategy 3: MACD Crossover"""
        params = (("printlog", False),)

        def __init__(self):
            self.macd = bt.indicators.MACD(self.data.close)
            self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        def next(self):
            if self.crossover > 0 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.position:
                self.close()

    class RSIStrategy(bt.Strategy):
        """Strategy 4: RSI Overbought/Oversold"""
        params = (("period", 14), ("oversold", 30), ("overbought", 70), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

        def next(self):
            if self.rsi < self.p.oversold and not self.position:
                self.buy()
            elif self.rsi > self.p.overbought and self.position:
                self.close()

    class BollingerStrategy(bt.Strategy):
        """Strategy 5: Bollinger Bands Bounce"""
        params = (("period", 20), ("devfactor", 2.0), ("printlog", False))

        def __init__(self):
            self.bbands = bt.indicators.BollingerBands(
                self.data.close, period=self.p.period, devfactor=self.p.devfactor
            )

        def next(self):
            if self.data.close[0] < self.bbands.lines.bot[0] and not self.position:
                self.buy()
            elif self.data.close[0] > self.bbands.lines.top[0] and self.position:
                self.close()

    class StochasticStrategy(bt.Strategy):
        """Strategy 6: Stochastic Oscillator"""
        params = (("period", 14), ("oversold", 20), ("overbought", 80), ("printlog", False))

        def __init__(self):
            self.stoch = bt.indicators.Stochastic(self.data, period=self.p.period)

        def next(self):
            if self.stoch.percK[0] < self.p.oversold and not self.position:
                self.buy()
            elif self.stoch.percK[0] > self.p.overbought and self.position:
                self.close()

    class RSIMACDStrategy(bt.Strategy):
        """Strategy 7: RSI + MACD Combined"""
        params = (("rsi_period", 14), ("rsi_low", 30), ("rsi_high", 70), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.macd = bt.indicators.MACD(self.data.close)
            self.crossover = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

        def next(self):
            if self.crossover > 0 and self.rsi < 50 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.rsi > 50 and self.position:
                self.close()

    class SMACloseAboveEMAStrategy(bt.Strategy):
        """Strategy 8: Price close above/below EMA filter"""
        params = (("ema_period", 50), ("printlog", False))

        def __init__(self):
            self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)

        def next(self):
            if self.data.close[0] > self.ema[0] and not self.position:
                self.buy()
            elif self.data.close[0] < self.ema[0] and self.position:
                self.close()

    class RSIMeanReversionStrategy(bt.Strategy):
        """Strategy 9: RSI mean reversion at extremes"""
        params = (("period", 14), ("very_low", 20), ("very_high", 80), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.period)

        def next(self):
            if self.rsi[0] < self.p.very_low and not self.position:
                self.buy()
            elif self.rsi[0] > self.p.very_high and self.position:
                self.close()
    class BollingerBreakoutStrategy(bt.Strategy):
        """Strategy 11: Bollinger band breakout"""
        params = (("period", 20), ("devfactor", 2.5), ("printlog", False))

        def __init__(self):
            self.bbands = bt.indicators.BollingerBands(
                self.data.close, period=self.p.period, devfactor=self.p.devfactor
            )

        def next(self):
            if self.data.close[0] > self.bbands.lines.top[0] and not self.position:
                self.buy()
            elif self.data.close[0] < self.bbands.lines.mid[0] and self.position:
                self.close()
            elif self.data.close[0] < self.bbands.lines.mid[0] and self.position:
                self.close()
    class DoubleSMAStrategy(bt.Strategy):
        """Strategy 12: Short/long SMA crossover (different params)"""
        params = (("fast", 5), ("slow", 50), ("printlog", False))

        def __init__(self):
            self.fast_sma = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.slow_sma = bt.indicators.SMA(self.data.close, period=self.p.slow)
            self.crossover = bt.indicators.CrossOver(self.fast_sma, self.slow_sma)

        def next(self):
            if self.crossover > 0 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.position:
                self.close()

    class TripleSMAStrategy(bt.Strategy):
        """Strategy 13: 3-SMA trend filter (fast > medium > slow)"""
        params = (("fast", 5), ("medium", 20), ("slow", 50), ("printlog", False))

        def __init__(self):
            self.fast = bt.indicators.SMA(self.data.close, period=self.p.fast)
            self.medium = bt.indicators.SMA(self.data.close, period=self.p.medium)
            self.slow = bt.indicators.SMA(self.data.close, period=self.p.slow)

        def next(self):
            if self.fast[0] > self.medium[0] > self.slow[0] and not self.position:
                self.buy()
            elif self.fast[0] < self.medium[0] and self.position:
                self.close()

    class RSIBollingerStrategy(bt.Strategy):
        """Strategy 14: RSI filter + Bollinger bounce"""
        params = (("rsi_period", 14), ("rsi_floor", 40), ("period", 20), ("devfactor", 2.0), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.bbands = bt.indicators.BollingerBands(
                self.data.close, period=self.p.period, devfactor=self.p.devfactor
            )

        def next(self):
            if self.data.close[0] < self.bbands.lines.bot[0] and self.rsi[0] > self.p.rsi_floor and not self.position:
                self.buy()
            elif self.data.close[0] > self.bbands.lines.mid[0] and self.position:
                self.close()

    class EMACrossoverTrendStrategy(bt.Strategy):
        """Strategy 15: EMA(20)/EMA(50) trend following"""
        params = (("fast", 20), ("slow", 50), ("printlog", False))

        def __init__(self):
            self.fast_ema = bt.indicators.EMA(self.data.close, period=self.p.fast)
            self.slow_ema = bt.indicators.EMA(self.data.close, period=self.p.slow)
            self.crossover = bt.indicators.CrossOver(self.fast_ema, self.slow_ema)

        def next(self):
            if self.crossover > 0 and not self.position:
                self.buy()
            elif self.crossover < 0 and self.position:
                self.close()

    class RSIMACDTrendStrategy(bt.Strategy):
        """Strategy 16: RSI trend filter + MACD histogram"""
        params = (("rsi_period", 14), ("rsi_trend", 50), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.macd = bt.indicators.MACD(self.data.close)

        def next(self):
            if self.rsi[0] > self.p.rsi_trend and self.macd.histo[0] > 0 and not self.position:
                self.buy()
            elif self.rsi[0] < self.p.rsi_trend and self.position:
                self.close()

    class StochasticRSIStrategy(bt.Strategy):
        """Strategy 17: Stochastic + RSI agreement"""
        params = (("stoch_period", 14), ("rsi_period", 14), ("oversold", 20), ("overbought", 80), ("printlog", False))

        def __init__(self):
            self.stoch = bt.indicators.Stochastic(self.data, period=self.p.stoch_period)
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

        def next(self):
            if self.stoch.percK[0] < self.p.oversold and self.rsi[0] < 50 and not self.position:
                self.buy()
            elif self.stoch.percK[0] > self.p.overbought and self.rsi[0] > 50 and self.position:
                self.close()

    class VolatilityBreakoutStrategy(bt.Strategy):
        """Strategy 18: ATR-based breakout above yesterday's range"""
        params = (("atr_period", 14), ("mult", 1.0), ("printlog", False))

        def __init__(self):
            self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)

        def next(self):
            if len(self.data) < 2:
                return
            upper_level = self.data.high[-1] + self.p.mult * self.atr[-1]
            if self.data.close[0] > upper_level and not self.position:
                self.buy()
            elif self.data.close[0] < self.data.low[-1] and self.position:
                self.close()

    class SimpleTrendFollowingStrategy(bt.Strategy):
        """Strategy 19: Price above/below SMA(50) trend filter"""
        params = (("period", 50), ("printlog", False))

        def __init__(self):
            self.sma = bt.indicators.SMA(self.data.close, period=self.p.period)

        def next(self):
            if self.data.close[0] > self.sma[0] and not self.position:
                self.buy()
            elif self.data.close[0] < self.sma[0] and self.position:
                self.close()

    class RangeBoundRSIStrategy(bt.Strategy):
        """Strategy 20: Range-bound RSI"""
        params = (("rsi_period", 14), ("range_period", 20), ("rsi_recover", 40), ("printlog", False))

        def __init__(self):
            self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
            self.lowest = bt.indicators.Lowest(self.data.low, period=self.p.range_period)

        def next(self):
            if self.data.close[0] > self.lowest[0] and self.rsi[0] > self.p.rsi_recover and not self.position:
                self.buy()
            elif self.rsi[0] < 50 and self.position:
                self.close()



# Map a short name -> strategy class
# Map a short name -> strategy class
def get_strategy_registry() -> Dict[str, Any]:
    if not BACKTRADER_AVAILABLE:
        return {}
    return {
        "SMA": SMAStrategy,
        "EMA": EMAStrategy,
        "MACD": MACDStrategy,
        "RSI": RSIStrategy,
        "BOLL": BollingerStrategy,
        "STOCH": StochasticStrategy,
        "RSI_MACD": RSIMACDStrategy,
        "SMA_EMA_FILTER": SMACloseAboveEMAStrategy,
        "RSI_MEANREV": RSIMeanReversionStrategy,
        "BOLL_BREAKOUT": BollingerBreakoutStrategy,
        "DOUBLE_SMA": DoubleSMAStrategy,
        "TRIPLE_SMA": TripleSMAStrategy,
        "RSI_BOLL": RSIBollingerStrategy,
        "EMA_TREND": EMACrossoverTrendStrategy,
        "STOCH_RSI": StochasticRSIStrategy,
        "ATR_BREAKOUT": VolatilityBreakoutStrategy,
        "TREND_SMA50": SimpleTrendFollowingStrategy,
        "RANGE_RSI": RangeBoundRSIStrategy,
    }

# ============================================================================
# Optimizer / Backtester
# ============================================================================

class StrategyOptimizer:
    """
    Runs backtests across strategies x timeframes x pairs, stores results, ranks and saves.
    """
    def __init__(
        self,
        trader: IBKRAlgoTrader,
        strategies: Sequence[Type] | Sequence[str],
        timeframes: Sequence[str],
        pairs: Sequence[Tuple[str, str]],  # (symbol, currency)
    ):
        if not BACKTRADER_AVAILABLE:
            raise RuntimeError("Backtrader not installed. Install backtrader to use StrategyOptimizer.")

        self.trader = trader
        self.timeframes = list(timeframes)
        self.pairs = list(pairs)
        self.results: List[Dict[str, Any]] = []

        registry = get_strategy_registry()
        resolved: List[Type] = []
        for s in strategies:
            if isinstance(s, str):
                if s not in registry:
                    raise ValueError(f"Unknown strategy name: {s}. Available: {list(registry)}")
                resolved.append(registry[s])
            else:
                resolved.append(s)
        self.strategies = resolved

    @staticmethod
    def _duration_for_timeframe(timeframe: str) -> str:
        """
        Duration mapping to ensure at least 50+ bars available.
        Matches IBKR API constraints (e.g., 1-min data only goes back ~1 month).
        """
        duration_map = {
            "1 min": "1 D",        # 1440 bars in 24h = plenty for 50 bar min
            "5 mins": "1 W",       # 288 bars in 1 week
            "15 mins": "2 W",      # 96 bars in 2 weeks
            "30 mins": "1 M",      # 48 bars in 1 month
        }
        return duration_map.get(timeframe, "1 M")

    def run_optimization(self, initial_cash: float = 1000.0) -> List[Dict[str, Any]]:
        total_tests = len(self.strategies) * len(self.timeframes) * len(self.pairs)
        completed = 0

        for strategy_class in self.strategies:
            for timeframe in self.timeframes:
                for (symbol, currency) in self.pairs:
                    completed += 1
                    try:
                        result = self._backtest_single(strategy_class, symbol, currency, timeframe, initial_cash)
                        self.results.append(result)
                    except Exception as e:
                        print(f"[ERROR] Backtest failed for {strategy_class.__name__} {timeframe} {symbol}/{currency}: {e}")

                    if completed % 10 == 0:
                        pct = completed / total_tests * 100
                        print(f"Progress: {completed}/{total_tests} ({pct:.1f}%)")

        print(f"\n[OK] Optimization complete! {len(self.results)} successful backtests")
        return self.results

    def _backtest_single(
        self,
        strategy_class: Type,
        symbol: str,
        currency: str,
        timeframe: str,
        initial_cash: float,
    ) -> Dict[str, Any]:
        contract = self.trader.get_contract(symbol, "CASH", "IDEALPRO", currency)
        duration = self._duration_for_timeframe(timeframe)
        df = self.trader.get_historical_data(contract, duration=duration, bar_size=timeframe)

        if df.empty or len(df) < 20:  # Relaxed from 50 to 20 bars (still enough for SMA/EMA)
            raise ValueError(f"Insufficient data: got {len(df) if not df.empty else 0} bars")

        # Prepare data for backtrader feed
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        data = bt.feeds.PandasData(dataname=df)

        cerebro = bt.Cerebro()
        cerebro.addstrategy(strategy_class)
        cerebro.adddata(data)
        cerebro.broker.setcash(initial_cash)
        cerebro.broker.setcommission(commission=0.0001)  # placeholder for FX costs

        # Analyzers (as in your screenshots)
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.Returns, _name="returns")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")

        results = cerebro.run()
        strat = results[0]

        final_value = cerebro.broker.getvalue()
        total_return = (final_value / initial_cash - 1.0) * 100.0

        sharpe = strat.analyzers.sharpe.get_analysis().get("sharperatio")
        if sharpe is None:
            sharpe = 0.0

        drawdown_info = strat.analyzers.drawdown.get_analysis()
        max_drawdown = drawdown_info.get("max", {}).get("drawdown", 0.0)

        trades_info = strat.analyzers.trades.get_analysis()
        total_trades = trades_info.get("total", {}).get("closed", 0)

        return {
            "strategy": strategy_class.__name__,
            "pair": f"{symbol}/{currency}",
            "timeframe": timeframe,
            "final_value": float(final_value),
            "total_return": float(total_return),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_drawdown),
            "total_trades": int(total_trades),
            "data_points": int(len(df)),
        }

    def get_ranked_strategies(self, top_n: int = 10, min_trades: int = 5) -> List[Dict[str, Any]]:
        valid = [r for r in self.results if r.get("total_trades", 0) >= min_trades]
        ranked = sorted(valid, key=lambda x: (x["sharpe_ratio"], x["total_return"]), reverse=True)
        return ranked[:top_n]

    def save_results(self, filename: str = "optimization_results.json") -> None:
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"[OK] Results saved to {filename}")


# ============================================================================
# Live Trader (profit target / stop loss)
# ============================================================================

class LiveTrader:
    """
    Live trading loop:
      - Monitor NetLiquidation for profit/stop
      - If no open position, execute strategy signal and open position
      - Log trades
    """
    def __init__(
        self,
        trader: IBKRAlgoTrader,
        strategy_config: StrategyConfig,
        risk: LiveRiskConfig = LiveRiskConfig(),
        logger: Optional[TradeLogger] = None,
        cycle: int = 1,
    ):
        self.trader = trader
        self.cfg = strategy_config
        self.risk = risk
        self.logger = logger or TradeLogger()
        self.cycle = cycle

        self.symbol, self.currency = self.cfg.pair.split("/")
        self.timeframe = self.cfg.timeframe

        self.initial_balance: Optional[float] = None
        self.position_opened: bool = False
        self.entry_time: Optional[str] = None

    def run(self) -> Dict[str, Any]:
        print("\n" + "=" * 80)
        print(f"LIVE TRADING: {self.cfg.strategy} on {self.cfg.pair} ({self.timeframe})")
        self.initial_balance = self.trader.get_account_value()
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Trade Size: {self.risk.balance_risk_pct}% of balance (min {self.risk.min_quantity} units)")
        print("=" * 80 + "\n")





        try:
            while True:
                current_balance = self.trader.get_account_value()
                pnl = current_balance - (self.initial_balance or current_balance)

                t = datetime.now().strftime("%H:%M:%S")
                print(f"[{t}] Balance: ${current_balance:,.2f} | P&L: ${pnl:+.2f}")

                # Profit target
                if pnl >= self.risk.profit_target:
                    print(f"\n[OK] PROFIT TARGET HIT! P&L: ${pnl:+.2f}")
                    self.close_all_positions()
                    return {"status": "profit_hit", "pnl": pnl}

                # Stop loss
                if pnl <= -self.risk.stop_loss:
                    print(f"\n[STOP] STOP LOSS HIT! P&L: ${pnl:+.2f}")
                    self.close_all_positions()
                    return {"status": "stop_loss_hit", "pnl": pnl}

                # Only execute entry if no open position
                if not self.position_opened:
                    self.execute_strategy_entry()

                time.sleep(self.risk.check_interval)

        except KeyboardInterrupt:
            msg = "\n[STOP] Manual stop requested. Leaving any open positions unchanged."
            print(msg)
            log_message(msg)
            return {"status": "manual_stop", "pnl": pnl}

    def execute_strategy_entry(self) -> None:
        """
        Minimal example signal (mirrors your screenshot): SMA(10) vs SMA(30) crossover check
        You can replace this with a dispatch based on cfg.strategy.
        """
        try:
            contract = self.trader.get_contract(self.symbol, "CASH", "IDEALPRO", self.currency)

            # Request a reasonably long history using IB-safe units.
            # For intraday bars, ask for 7 days; for daily, 180 days.
            tf = self.timeframe.lower()
            if "min" in tf or "hour" in tf:
                duration = "7 D"
            elif "day" in tf:
                duration = "180 D"
            else:
                duration = "7 D"

            df = self.trader.get_historical_data(contract, duration=duration, bar_size=self.timeframe)
            if df.empty or len(df) < 10:
                msg = "[WARN] Insufficient data for signal"
                print(msg)
                log_message(msg)
                return

            df = df.copy()
            df["sma_fast"] = df["close"].rolling(10).mean()
            df["sma_slow"] = df["close"].rolling(30).mean()
            latest = df.iloc[-1]

            if pd.isna(latest["sma_fast"]) or pd.isna(latest["sma_slow"]):
                msg = "[WARN] SMA not ready yet"
                print(msg)
                log_message(msg)
                return

            if latest["sma_fast"] > latest["sma_slow"]:
                print("BUY signal detected")

                account_balance = self.trader.get_account_value()
                if account_balance <= 0:
                    msg = "[WARN] Cannot size trade: account balance is zero or negative"
                    print(msg)
                    log_message(msg)
                    return

                close_price = float(latest["close"])
                if close_price <= 0:
                    msg = "[WARN] Cannot size trade: last close price is zero or negative"
                    print(msg)
                    log_message(msg)
                    return

                # FX has margin requirement (typically 50:1, so 2% of notional needed)
                margin_requirement = 0.02  # 2% (50:1 leverage)
                max_notional = account_balance / margin_requirement
                max_quantity = int(max_notional / close_price)
                
                risk_fraction = max(self.risk.balance_risk_pct, 0.0) / 100.0
                if risk_fraction > 0:
                    notional = account_balance * risk_fraction
                    quantity = int(notional / close_price)
                    # Cap to max allowed by margin
                    quantity = min(quantity, max_quantity)
                    quantity = max(quantity, self.risk.min_quantity)
                else:
                    # Fallback to fixed quantity if percentage is disabled
                    quantity = min(self.risk.quantity, max_quantity)

                # Final check: ensure we have enough margin
                required_cash = quantity * close_price * margin_requirement
                if required_cash > account_balance:
                    msg = f"[WARN] Insufficient margin. Need ${required_cash:.2f}, have ${account_balance:.2f}. Skipping trade."
                    print(msg)
                    log_message(msg)
                    return

                msg = (
                    f"Placing MARKET BUY for {quantity} units "
                    f"(balance=${account_balance:.2f}, price=${close_price:.5f}, margin_req=${required_cash:.2f})"
                )
                print(msg)
                log_message(msg)
                
                # Use market order to guarantee execution on entry signal
                trade = self.trader.place_market_order(contract, "BUY", quantity)
                
                # Wait a moment for order to be processed
                time.sleep(2)
                
                # Verify position was actually filled
                positions = self.trader.get_positions()
                position_filled = False
                for p in positions:
                    if p.contract.symbol == self.symbol:
                        pos_size = getattr(p, "position", 0)
                        if pos_size > 0:
                            position_filled = True
                            self.position_quantity = pos_size
                            break
                
                if position_filled:
                    self.position_opened = True
                    self.entry_time = now_ts()
                    self.entry_price = float(latest["close"])
                    print(f"[OK] Position FILLED: {self.position_quantity} units @ {close_price:.5f}")
                else:
                    print("[WARN] Market order placed but position not yet filled. Retrying next cycle...")
                    self.position_opened = False

        except Exception as e:
            print(f"[ERROR] Strategy execution error: {e}")

    def close_all_positions(self) -> None:
        """
        Closes ALL open positions in the account.
        (Use with care - you can narrow to just this symbol if you prefer.)
        """
        positions = self.trader.get_positions()
        for p in positions:
            try:
                pos_size = getattr(p, "position", 0)
                if not pos_size:
                    continue

                contract = p.contract
                action = "SELL" if pos_size > 0 else "BUY"
                qty = int(abs(pos_size))
                self.trader.place_market_order(contract, action, qty)

                self.logger.log_trade(
                    strategy=self.cfg.strategy,
                    pair=self.cfg.pair,
                    timeframe=self.timeframe,
                    action=action,
                    quantity=qty,
                    price=None,
                    order_id="N/A",
                    fee=0.0,
                    status="Closed",
                )
            except Exception as e:
                print(f"[ERROR] Failed closing position: {e}")


# ============================================================================
# Example usage
# ============================================================================

def main():
    # Ask at runtime (better for .exe use)
    live_answer = input("Use LIVE IBKR account? (y/N): ").strip().lower()
    USE_LIVE_ACCOUNT = live_answer.startswith("y")

    rerun_answer = input("Re-run strategy optimization? (y/N, 'N' reuses last results if available): ").strip().lower()
    RERUN_OPTIMIZATION = rerun_answer.startswith("y")

    # ---- Connect ----
    if USE_LIVE_ACCOUNT:
        print("[MODE] LIVE trading mode selected. Ensure LIVE TWS/Gateway is running and API is enabled.")
        conn = IBKRConnectionConfig(host="127.0.0.1", port=7496, client_id=2)
    else:
        print("[MODE] PAPER trading mode selected.")
        conn = IBKRConnectionConfig(host="127.0.0.1", port=7497, client_id=1)
    trader = IBKRAlgoTrader(conn)
    if not trader.connect():
        return

    logger = TradeLogger()

    # ---- Optimization (Backtrader required) ----
    # Either re-run optimization or reuse the last saved results.
    if BACKTRADER_AVAILABLE and RERUN_OPTIMIZATION:
        registry = get_strategy_registry()
        strategies = list(registry.keys())  # e.g. ["SMA", "EMA", ...]
        timeframes = [
            "1 min",
            "5 mins",
            "15 mins",
            "30 mins",
        ]
        pairs = [("EUR", "USD"), ("GBP", "CHF"), ("AUD", "NZD")]
        # pairs = [("USD", "CAD"), ("GBP", "AUD")]

        opt = StrategyOptimizer(trader, strategies, timeframes, pairs)
        opt.run_optimization(initial_cash=1000.0)
        ranked = opt.get_ranked_strategies(top_n=20, min_trades=5)
        print("\nTop strategies:")
        for r in ranked:
            print(r)
        opt.save_results("optimization_results.json")

    elif BACKTRADER_AVAILABLE and not RERUN_OPTIMIZATION and Path("optimization_results.json").exists():
        print("[INFO] Reusing strategies from optimization_results.json")
        with open("optimization_results.json", "r") as f:
            all_results = json.load(f)
        # Apply the same ranking rules
        valid = [r for r in all_results if r.get("total_trades", 0) >= 5]
        ranked = sorted(valid, key=lambda x: (x["sharpe_ratio"], x["total_return"]), reverse=True)[:5]
        print("\nTop strategies (reused):")
        for r in ranked:
            print(r)
    else:
        ranked = []

    # Pick the single best config for live trading
    if ranked:
        best = ranked[0]
        best_strategy = best["strategy"]
        best_pair = best["pair"]          # already like "AUD/NZD"
        best_timeframe = best["timeframe"]
        print(
            f"\n[INFO] Best config from optimization: "
            f"{best_strategy} on {best_pair} ({best_timeframe}) "
            f"Sharpe={best['sharpe_ratio']:.2f}, Return={best['total_return']:.2%}"
        )
    else:
        print("[WARN] No valid strategies from optimization; falling back to default live config")
        best_strategy = "SMA"
        best_pair = "AUD/NZD"
        best_timeframe = "1 hour"

    # ---- Live Trading ----
    # Uses the best config from optimization (or default if optimization skipped)
    live_cfg = StrategyConfig(strategy=best_strategy, pair=best_pair, timeframe=best_timeframe)
    risk = LiveRiskConfig(
        profit_target=1,       # close all when P&L >= +$1
        stop_loss=3,           # close all when P&L <= -$3
        quantity=1000,         # legacy / unused as long as balance_risk_pct > 0
        balance_risk_pct=10.0, # 10% of account balance per trade
        min_quantity=1,       # at least 1 units
        check_interval=60,     # check every 60 seconds
    )
    live = LiveTrader(trader, live_cfg, risk=risk, logger=logger, cycle=1)

    # Optional: simple paper-test of BUY then SELL at market for the live pair.
    # Uncomment to fire a one-off round-trip test trade.
    #
    # def test_paper_roundtrip():
    #     contract = trader.get_contract(symbol=best_pair.split("/")[0], sec_type="CASH", exchange="IDEALPRO", currency=best_pair.split("/")[1])
    #     qty = 1000
    #     print(f"[TEST] Placing test BUY for {qty} units on {best_pair}")
    #     trader.place_market_order(contract, "BUY", qty)
    #     time.sleep(5)
    #     print(f"[TEST] Placing test SELL for {qty} units on {best_pair}")
    #     trader.place_market_order(contract, "SELL", qty)
    
    # test_paper_roundtrip()

    # IMPORTANT: keep this active only when you are sure you're on a PAPER account
    live.run()

    trader.disconnect()


if __name__ == "__main__":
    main()
