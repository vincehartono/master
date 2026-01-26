"""
Alpaca Crypto Algorithmic Trading System
=========================================

Features:
- Real-time crypto trading (BTC, ETH, SOL, etc.)
- Multiple strategies: SMA Crossover, MACD, RSI
- Risk management: position sizing, stop loss, profit target
- Trade logging & P&L tracking
- Paper and live trading modes
- Live web dashboard (http://localhost:5000)

Install:
  pip install alpaca-py pandas numpy python-dotenv flask

Setup:
  1. Get Alpaca API keys from https://app.alpaca.markets/
  2. Create .env file:
     APCA_API_KEY_ID=your_key
     APCA_API_SECRET_KEY=your_secret
     APCA_API_BASE_URL=https://paper-api.alpaca.markets  (for paper)
  3. Run: python trade.py
  4. Open: http://localhost:5000 in browser
"""

import os
import sys
import csv
import json
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Flask for dashboard
try:
    from flask import Flask, render_template, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# Alpaca API
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    MarketOrderRequest,
    LimitOrderRequest,
    OrderSide,
    TimeInForce,
    ClosePositionRequest,
)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ============================================================================
# Shared Dashboard State
# ============================================================================

class DashboardState:
    """Shared state for live dashboard"""
    def __init__(self):
        self.strategy = "SMA"
        self.symbol = "BTC/USD"
        self.timeframe = "1Min"
        self.combo_score = 0.0
        self.win_rate = 0.0
        self.profit_factor = 1.0
        self.sharpe = 0.0
        
        self.price = 0.0
        self.high = 0.0
        self.low = 0.0
        self.volume = 0.0
        self.timestamp = datetime.now().isoformat()
        
        self.indicators = {
            'sma_fast': None,
            'sma_slow': None,
            'rsi': 50,
            'bb_upper': None,
            'bb_lower': None,
            'bb_mid': None,
            'stoch': 50,
        }
        
        self.equity = 1000.0
        self.cash = 1000.0
        self.buying_power = 1000.0
        
        self.recent_trades = []
        self.last_signal = None
        self.last_signal_time = None

# Global dashboard state
dashboard_state = DashboardState()

# ============================================================================
# Configuration
# ============================================================================

load_dotenv()

# Alpaca API credentials - will be set based on trading mode toggle
API_KEY = None
SECRET_KEY = None
BASE_URL = None

# Credential check will happen in set_credentials()


def set_credentials(is_live: bool) -> None:
    """Set API credentials based on trading mode"""
    global API_KEY, SECRET_KEY, BASE_URL
    
    if is_live:
        API_KEY = os.getenv("APCA_API_KEY_ID_LIVE")
        SECRET_KEY = os.getenv("APCA_API_SECRET_KEY_LIVE")
        BASE_URL = "https://api.alpaca.markets"
        os.environ["APCA_API_BASE_URL"] = BASE_URL
    else:
        API_KEY = os.getenv("APCA_API_KEY_ID_PAPER")
        SECRET_KEY = os.getenv("APCA_API_SECRET_KEY_PAPER")
        BASE_URL = "https://paper-api.alpaca.markets"
        os.environ["APCA_API_BASE_URL"] = BASE_URL


def get_script_dir() -> str:
    """Get the correct script directory (handles both EXE and Python execution)"""
    # Check if running from PyInstaller bundle
    if getattr(sys, 'frozen', False):
        # Running as EXE - look for files in the EXE directory, not temp
        return os.path.dirname(sys.executable)
    else:
        # Running as Python script
        return os.path.dirname(os.path.abspath(__file__))
    
    if not API_KEY or not SECRET_KEY:
        print("[ERROR] Missing API credentials in .env file")
        sys.exit(1)


@dataclass(frozen=True)
class TradingConfig:
    """Trading parameters"""
    # Strategy
    strategy: str = "SMA"  # SMA, MACD, RSI
    symbols: List[str] = field(default_factory=lambda: ["ETH/USD", "SHIB/USD", "DOGE/USD", "SOL/USD"])
    timeframe: str = "1Min"  # Store as string for flexibility
    lookback_bars: int = 100  # Historical bars for indicators
    
    # Risk management
    position_size_pct: float = 10.0  # % of account to trade per trade
    stop_loss: float = 5.0  # dollars (fixed amount, not %)
    profit_target: float = 2.0  # dollars (fixed amount, not %)
    max_positions: int = 5  # Max open positions
    
    # Execution
    check_interval: int = 60  # Seconds between checks
    paper_trading: bool = True  # Default to paper
    
    # SMA Strategy params
    sma_fast: int = 10
    sma_slow: int = 30


# ============================================================================
# Trading Fee Structure (Alpaca Crypto Spot)
# ============================================================================
# Level 1: 0 - $100,000 volume - Maker: 0.15%, Taker: 0.25%
# Level 2: $100,000+ volume - Maker: 0.10%, Taker: 0.20%
# etc.

MAKER_FEE = 0.0015  # 0.15% for maker orders (post to order book)
TAKER_FEE = 0.0025  # 0.25% for taker orders (immediate execution)

def calculate_trading_fees(order_value: float, order_type: str = "market") -> float:
    """
    Calculate trading fees based on order type
    
    Args:
        order_value: Total value of the order in USD
        order_type: "market" (taker) or "limit" (maker)
    
    Returns:
        Fee amount in USD
    """
    # Market orders are typically taker orders
    if order_type.lower() == "market":
        return order_value * TAKER_FEE
    else:
        # Limit orders are typically maker orders
        return order_value * MAKER_FEE


# ============================================================================
# Logging
# ============================================================================

def log_message(msg: str) -> None:
    """Log to console and file"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    
    with open("crypto_trade.log", "a") as f:
        f.write(f"[{ts}] {msg}\n")


class TradeLogger:
    """Log trades and P&L"""
    
    def __init__(self, trade_log: str = "crypto_trades.csv", pnl_log: str = "crypto_pnl.csv"):
        self.trade_log = Path(trade_log)
        self.pnl_log = Path(pnl_log)
        self._init_logs()
    
    def _init_logs(self) -> None:
        if not self.trade_log.exists():
            with open(self.trade_log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "Timestamp", "Symbol", "Side", "Quantity", "Price",
                    "Order_ID", "Status", "Commission"
                ])
        
        if not self.pnl_log.exists():
            with open(self.pnl_log, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "Timestamp", "Symbol", "Entry_Time", "Exit_Time",
                    "Entry_Price", "Exit_Price", "Quantity", "Gross_PnL",
                    "Commission", "Net_PnL", "Account_Equity", "Status"
                ])
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        order_id: str,
        status: str,
        commission: float = 0.0,
    ) -> None:
        """Log individual trade"""
        with open(self.trade_log, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(),
                symbol,
                side,
                qty,
                f"{price:.2f}",
                order_id,
                status,
                f"{commission:.4f}",
            ])
    
    def log_pnl(
        self,
        symbol: str,
        entry_time: str,
        exit_time: str,
        entry_price: float,
        exit_price: float,
        qty: float,
        gross_pnl: float,
        commission: float,
        account_equity: float,
        status: str = "Closed",
    ) -> None:
        """Log closed trade P&L"""
        net_pnl = gross_pnl - commission
        with open(self.pnl_log, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(),
                symbol,
                entry_time,
                exit_time,
                f"{entry_price:.2f}",
                f"{exit_price:.2f}",
                f"{qty:.4f}",
                f"{gross_pnl:.2f}",
                f"{commission:.2f}",
                f"{net_pnl:.2f}",
                f"{account_equity:.2f}",
                status,
            ])


# ============================================================================
# Flask Dashboard Server
# ============================================================================

def start_dashboard_server():
    """Start Flask dashboard in background thread"""
    if not FLASK_AVAILABLE:
        return
    
    # Create Flask app with templates folder
    template_folder = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_folder)
    
    @app.route("/")
    def index():
        return render_template("dashboard.html")
    
    @app.route("/api/data")
    def api_data():
        return jsonify({
            "strategy": {
                "name": dashboard_state.strategy,
                "symbol": dashboard_state.symbol,
                "timeframe": dashboard_state.timeframe,
                "combo_score": dashboard_state.combo_score,
                "win_rate": dashboard_state.win_rate,
                "profit_factor": dashboard_state.profit_factor,
                "sharpe": dashboard_state.sharpe,
            },
            "price": {
                "symbol": dashboard_state.symbol,
                "price": dashboard_state.price,
                "high": dashboard_state.high,
                "low": dashboard_state.low,
                "volume": dashboard_state.volume,
                "timestamp": dashboard_state.timestamp,
                "indicators": dashboard_state.indicators,
            },
            "account": {
                "equity": dashboard_state.equity,
                "cash": dashboard_state.cash,
                "buying_power": dashboard_state.buying_power,
            },
            "trades": dashboard_state.recent_trades[-20:],  # Last 20 trades
            "last_signal": dashboard_state.last_signal,
            "last_signal_time": dashboard_state.last_signal_time,
            "timestamp": datetime.now().isoformat(),
        })
    
    # Run on port 5000
    try:
        app.run(debug=False, host="0.0.0.0", port=5000, use_reloader=False)
    except Exception as e:
        print(f"[WARNING] Dashboard server error: {e}")

# ============================================================================
# Alpaca Trading Client
# ============================================================================

class AlpacaCryptoBot:
    """Crypto trading bot using Alpaca API"""
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.logger = TradeLogger()
        
        # Initialize clients (uses APCA_API_BASE_URL from environment)
        self.trading_client = TradingClient(API_KEY, SECRET_KEY)
        self.data_client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=SECRET_KEY)
        
        # State tracking
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.last_prices: Dict[str, float] = {}
        self.bar_cache: Dict[str, pd.DataFrame] = {}
        
        # Update dashboard state
        dashboard_state.strategy = config.strategy
        dashboard_state.symbol = config.symbols[0] if config.symbols else "BTC/USD"
        dashboard_state.timeframe = config.timeframe
        
        msg = f"[{'PAPER' if config.paper_trading else 'LIVE'}] Alpaca Crypto Bot initialized"
        log_message(msg)
    
    def get_account(self) -> Dict[str, Any]:
        """Get account info"""
        try:
            account = self.trading_client.get_account()
            # Parse fields which may be strings
            equity = float(account.equity) if account.equity else 0.0
            cash = float(account.cash) if account.cash else 0.0
            buying_power = float(account.buying_power) if account.buying_power else 0.0
            
            # Update dashboard state
            dashboard_state.equity = equity
            dashboard_state.cash = cash
            dashboard_state.buying_power = buying_power
            
            return {
                "equity": equity,
                "cash": cash,
                "buying_power": buying_power,
                "returns": 0.0,  # Simplified for crypto
            }
        except Exception as e:
            log_message(f"[ERROR] Failed to get account: {e}")
            return {}
    
    def _parse_timeframe(self, timeframe_str: str) -> TimeFrame:
        """Parse timeframe string like '5Min' to TimeFrame object"""
        # Extract number from string like '30Min', '5Min', etc.
        try:
            amount = int(''.join(c for c in timeframe_str if c.isdigit()))
            return TimeFrame(amount=amount, unit=TimeFrameUnit.Minute)
        except:
            return TimeFrame(amount=1, unit=TimeFrameUnit.Minute)  # Default fallback
    
    def get_historical_bars(self, symbol: str, bars: int) -> pd.DataFrame:
        """Fetch historical crypto bars"""
        try:
            # Parse timeframe from config
            tf = self._parse_timeframe(self.config.timeframe)
            
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                limit=bars,
            )
            bars_data = self.data_client.get_crypto_bars(request)
            
            # Extract dataframe from BarSet object
            df = bars_data.df
            if df is not None and not df.empty:
                df = df.sort_index()
                return df
            
            return pd.DataFrame()
        except Exception as e:
            log_message(f"[ERROR] Failed to fetch bars for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate all indicators needed for strategies"""
        if df.empty or len(df) < 30:
            return {}
        
        df = df.copy()
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # SMA/EMA
        df['sma_fast'] = close.rolling(self.config.sma_fast).mean()
        df['sma_slow'] = close.rolling(self.config.sma_slow).mean()
        df['ema_fast'] = close.ewm(span=self.config.sma_fast).mean()
        df['ema_slow'] = close.ewm(span=self.config.sma_slow).mean()
        
        # MACD
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_mid'] = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # Momentum
        df['momentum'] = close - close.shift(10)
        
        # Stochastic
        min_low = low.rolling(14).min()
        max_high = high.rolling(14).max()
        df['stoch'] = 100 * (close - min_low) / (max_high - min_low)
        df['stoch_signal'] = df['stoch'].rolling(3).mean()
        
        # Volume SMA
        df['volume_sma'] = volume.rolling(20).mean()
        
        latest = df.iloc[-1]
        
        return {
            'close': float(latest['close']),
            'high': float(latest['high']),
            'low': float(latest['low']),
            'volume': float(latest['volume']),
            'sma_fast': float(latest['sma_fast']) if not pd.isna(latest['sma_fast']) else None,
            'sma_slow': float(latest['sma_slow']) if not pd.isna(latest['sma_slow']) else None,
            'ema_fast': float(latest['ema_fast']) if not pd.isna(latest['ema_fast']) else None,
            'ema_slow': float(latest['ema_slow']) if not pd.isna(latest['ema_slow']) else None,
            'macd': float(latest['macd']) if not pd.isna(latest['macd']) else 0,
            'macd_signal': float(latest['macd_signal']) if not pd.isna(latest['macd_signal']) else 0,
            'rsi': float(latest['rsi']) if not pd.isna(latest['rsi']) else 50,
            'bb_upper': float(latest['bb_upper']) if not pd.isna(latest['bb_upper']) else None,
            'bb_lower': float(latest['bb_lower']) if not pd.isna(latest['bb_lower']) else None,
            'bb_mid': float(latest['bb_mid']) if not pd.isna(latest['bb_mid']) else None,
            'atr': float(latest['atr']) if not pd.isna(latest['atr']) else 0,
            'momentum': float(latest['momentum']) if not pd.isna(latest['momentum']) else 0,
            'stoch': float(latest['stoch']) if not pd.isna(latest['stoch']) else 50,
            'stoch_signal': float(latest['stoch_signal']) if not pd.isna(latest['stoch_signal']) else 50,
            'volume_sma': float(latest['volume_sma']) if not pd.isna(latest['volume_sma']) else 0,
            'bars': len(df),
        }
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        if len(prices) < period:
            return 50.0
        
        deltas = prices.diff()
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        
        rs = up / down if down != 0 else 0
        rsi = 100 - (100 / (1 + rs)) if rs else 50
        return float(rsi)
    
    def generate_signal(self, symbol: str, indicators: Dict[str, Any]) -> Optional[str]:
        """Dispatch to strategy-specific signal generator"""
        if not indicators:
            return None
        
        strategy = self.config.strategy
        
        # Long strategies
        if strategy == "SMA Crossover":
            return self._signal_sma_crossover(indicators)
        elif strategy == "EMA Crossover":
            return self._signal_ema_crossover(indicators)
        elif strategy == "MACD":
            return self._signal_macd(indicators)
        elif strategy == "RSI":
            return self._signal_rsi(indicators)
        elif strategy == "Bollinger Bands":
            return self._signal_bollinger(indicators)
        elif strategy == "Stochastic":
            return self._signal_stochastic(indicators)
        elif strategy == "ATR Breakout":
            return self._signal_atr_breakout(indicators)
        elif strategy == "Volume Surge":
            return self._signal_volume_surge(indicators)
        # Short strategies
        elif strategy == "Short SMA":
            return self._signal_short_sma(indicators)
        elif strategy == "Short RSI":
            return self._signal_short_rsi(indicators)
        elif strategy == "Short Bollinger":
            return self._signal_short_bollinger(indicators)
        elif strategy == "Short Momentum":
            return self._signal_short_momentum(indicators)
        elif strategy == "Short Downtrend":
            return self._signal_short_downtrend(indicators)
        elif strategy == "Short EMA":
            return self._signal_short_ema(indicators)
        elif strategy == "Short Stochastic":
            return self._signal_short_stochastic(indicators)
        
        return None
    
    def _signal_sma_crossover(self, ind: Dict[str, Any]) -> Optional[str]:
        """SMA Crossover: Fast SMA above Slow SMA = BUY"""
        if not ind.get('sma_fast') or not ind.get('sma_slow'):
            return None
        if ind['sma_fast'] > ind['sma_slow'] * 1.001:
            return "BUY"
        elif ind['sma_fast'] < ind['sma_slow'] * 0.999:
            return "SELL"
        return None
    
    def _signal_ema_crossover(self, ind: Dict[str, Any]) -> Optional[str]:
        """EMA Crossover: Fast EMA above Slow EMA = BUY"""
        if not ind.get('ema_fast') or not ind.get('ema_slow'):
            return None
        if ind['ema_fast'] > ind['ema_slow'] * 1.001:
            return "BUY"
        elif ind['ema_fast'] < ind['ema_slow'] * 0.999:
            return "SELL"
        return None
    
    def _signal_macd(self, ind: Dict[str, Any]) -> Optional[str]:
        """MACD: MACD crosses above signal line = BUY"""
        if ind.get('macd') is None or ind.get('macd_signal') is None:
            return None
        if ind['macd'] > ind['macd_signal'] and ind.get('macd', 0) > 0:
            return "BUY"
        elif ind['macd'] < ind['macd_signal'] and ind.get('macd', 0) < 0:
            return "SELL"
        return None
    
    def _signal_rsi(self, ind: Dict[str, Any]) -> Optional[str]:
        """RSI: Oversold (<30) = BUY, Overbought (>70) = SELL"""
        rsi = ind.get('rsi', 50)
        if rsi < 30:
            return "BUY"
        elif rsi > 70:
            return "SELL"
        return None
    
    def _signal_bollinger(self, ind: Dict[str, Any]) -> Optional[str]:
        """Bollinger Bands: Price below lower band = BUY, above upper = SELL"""
        price = ind.get('close')
        bb_upper = ind.get('bb_upper')
        bb_lower = ind.get('bb_lower')
        bb_middle = ind.get('bb_middle')
        
        # DEBUG: Print Bollinger Bands status (only if values exist)
        if price and bb_upper and bb_lower and bb_middle:
            ts = datetime.now().strftime("%H:%M:%S")
            print(f"[{ts}] [DEBUG BB] Price: ${price:.4f} | Upper: ${bb_upper:.4f} | Mid: ${bb_middle:.4f} | Lower: ${bb_lower:.4f}")
        
        if not price or not bb_upper or not bb_lower:
            return None
        if price < bb_lower:
            signal = "BUY"
            print(f"  └─ Signal: {signal} (price < lower band)")
            return signal
        elif price > bb_upper:
            signal = "SELL"
            print(f"  └─ Signal: {signal} (price > upper band)")
            return signal
        else:
            print(f"  └─ Signal: HOLD (price between bands)")
        return None
    
    def _signal_short_sma(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short SMA: Fast SMA below Slow SMA = SHORT (SELL)"""
        if not ind.get('sma_fast') or not ind.get('sma_slow'):
            return None
        if ind['sma_fast'] < ind['sma_slow'] * 0.999:
            return "SELL"  # SHORT signal
        return None
    
    def _signal_short_rsi(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short RSI: Overbought (>70) = SHORT (SELL)"""
        rsi = ind.get('rsi', 50)
        if rsi > 70:
            return "SELL"  # SHORT signal
        return None
    
    def _signal_short_bollinger(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short Bollinger: Price above upper band = SHORT (SELL)"""
        price = ind.get('close')
        bb_upper = ind.get('bb_upper')
        if not price or not bb_upper:
            return None
        if price > bb_upper:
            return "SELL"  # SHORT signal
        return None
    
    def _signal_short_momentum(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short Momentum: Negative momentum = SHORT (SELL)"""
        momentum = ind.get('momentum', 0)
        if momentum < 0:
            return "SELL"  # SHORT signal
        return None
    
    def _signal_short_downtrend(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short Downtrend: When price drops below both SMAs = SHORT signal (SELL to initiate)"""
        if not ind.get('sma_fast') or not ind.get('sma_slow'):
            return None
        sma_fast = ind['sma_fast']
        sma_slow = ind['sma_slow']
        price = ind['close']
        
        # Downtrend setup: price below both SMAs = SELL (short entry)
        if price < sma_fast < sma_slow:
            return "SELL"  # SHORT entry signal
        # No exit signal - hold until trend changes
        return None
    
    def _signal_stochastic(self, ind: Dict[str, Any]) -> Optional[str]:
        """Stochastic: Oversold (<20) = BUY, Overbought (>80) = SELL"""
        stoch = ind.get('stoch', 50)
        if stoch < 20:
            return "BUY"
        elif stoch > 80:
            return "SELL"
        return None
    
    def _signal_atr_breakout(self, ind: Dict[str, Any]) -> Optional[str]:
        """ATR Breakout: Price breaks above prev high + ATR = BUY"""
        # Simplified: High volume + price up = BUY
        atr = ind.get('atr', 0)
        momentum = ind.get('momentum', 0)
        if atr > 0 and momentum > 0:
            return "BUY"
        elif atr > 0 and momentum < 0:
            return "SELL"
        return None
    
    def _signal_volume_surge(self, ind: Dict[str, Any]) -> Optional[str]:
        """Volume Surge: Volume > Volume SMA + price up = BUY"""
        volume = ind.get('volume', 0)
        volume_sma = ind.get('volume_sma', 0)
        close = ind.get('close', 0)
        momentum = ind.get('momentum', 0)
        
        if volume_sma > 0 and volume > volume_sma * 1.5 and momentum > 0:
            return "BUY"
        elif volume_sma > 0 and volume > volume_sma * 1.5 and momentum < 0:
            return "SELL"
        return None
    
    def _signal_short_ema(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short EMA: Fast EMA below Slow EMA = SHORT (SELL)"""
        if not ind.get('ema_fast') or not ind.get('ema_slow'):
            return None
        if ind['ema_fast'] < ind['ema_slow'] * 0.999:
            return "SELL"  # SHORT signal
        return None
    
    def _signal_short_stochastic(self, ind: Dict[str, Any]) -> Optional[str]:
        """Short Stochastic: Overbought (>80) = SHORT (SELL)"""
        stoch = ind.get('stoch', 50)
        if stoch > 80:
            return "SELL"  # SHORT signal
        return None
    
    def get_position_size(self, symbol: str, price: float) -> float:
        """Calculate position size based on account equity and risk"""
        account = self.get_account()
        if not account or account['equity'] <= 0:
            return 0.0
        
        # Risk = position_size_pct % of account
        risk_amount = account['equity'] * (self.config.position_size_pct / 100.0)
        
        # Quantity = risk / price
        qty = risk_amount / price
        
        # Round to reasonable precision (crypto usually 4 decimals)
        return round(qty, 4)
    
    def place_buy_order(self, symbol: str, qty: float, price: float = None) -> Optional[str]:
        """Place market buy order"""
        try:
            # Use provided price or fall back to dashboard state price
            if price is None:
                price = dashboard_state.price
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.IOC,  # Immediate or cancel
            )
            order = self.trading_client.submit_order(request)
            
            # Calculate order value and fees
            order_value = qty * price
            fee = calculate_trading_fees(order_value, "market")  # Market orders are takers
            
            msg = f"[BUY] {symbol} x{qty} @ ${price:.2f} = ${order_value:.2f} | Fee: ${fee:.4f} ({TAKER_FEE*100:.2f}%) | Order #{order.id}"
            log_message(msg)
            self.logger.log_trade(symbol, "BUY", qty, fee, str(order.id), "Submitted")
            
            # Update dashboard
            dashboard_state.last_signal = "BUY"
            dashboard_state.last_signal_time = datetime.now().isoformat()
            dashboard_state.recent_trades.append({
                "Timestamp": datetime.now().isoformat(),
                "Symbol": symbol,
                "Side": "BUY",
                "Quantity": qty,
                "Price": price,
                "Value": order_value,
                "Fee": fee,
                "Status": "Submitted",
            })
            
            return str(order.id)
        except Exception as e:
            log_message(f"[ERROR] Failed to place BUY order for {symbol}: {e}")
            return None
    
    def place_sell_order(self, symbol: str, qty: float, price: float = None) -> Optional[str]:
        """Place market sell order"""
        try:
            # Use provided price or fall back to dashboard state price
            if price is None:
                price = dashboard_state.price
            
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.IOC,
            )
            order = self.trading_client.submit_order(request)
            
            # Calculate order value and fees
            order_value = qty * price
            fee = calculate_trading_fees(order_value, "market")  # Market orders are takers
            
            msg = f"[SELL] {symbol} x{qty} @ ${price:.2f} = ${order_value:.2f} | Fee: ${fee:.4f} ({TAKER_FEE*100:.2f}%) | Order #{order.id}"
            log_message(msg)
            self.logger.log_trade(symbol, "SELL", qty, fee, str(order.id), "Submitted")
            
            # Update dashboard
            dashboard_state.last_signal = "SELL"
            dashboard_state.last_signal_time = datetime.now().isoformat()
            dashboard_state.recent_trades.append({
                "Timestamp": datetime.now().isoformat(),
                "Symbol": symbol,
                "Side": "SELL",
                "Quantity": qty,
                "Price": price,
                "Value": order_value,
                "Fee": fee,
                "Status": "Submitted",
            })
            
            return str(order.id)
        except Exception as e:
            log_message(f"[ERROR] Failed to place SELL order for {symbol}: {e}")
            return None
    
    def close_all_positions(self) -> None:
        """Close all open positions"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                qty = abs(float(position.qty))
                if qty > 0:
                    request = ClosePositionRequest(qty=qty)
                    self.trading_client.close_position(position.symbol, request)
                    log_message(f"[CLOSE] {position.symbol} x{qty}")
        except Exception as e:
            log_message(f"[ERROR] Failed to close positions: {e}")
    
    def run_trading_loop(self) -> None:
        """Main trading loop"""
        account_initial = self.get_account()
        log_message(f"Starting with equity: ${account_initial.get('equity', 0):.2f}")
        log_message(f"Configuration: {self.config.strategy} strategy, {self.config.symbols}")
        log_message(f"Stop Loss: ${self.config.stop_loss:.2f} | Profit Target: ${self.config.profit_target:.2f}")
        log_message("=" * 80)
        
        try:
            loop_count = 0
            last_trade_time = datetime.now()
            no_trade_threshold = 60 * 60  # 1 hour in seconds
            
            while True:
                loop_count += 1
                account = self.get_account()
                if not account:
                    time.sleep(self.config.check_interval)
                    continue
                
                # Calculate P&L from initial balance
                current_pnl = account['equity'] - account_initial.get('equity', account['equity'])
                
                # Display account status with progress indicator
                ts = datetime.now().strftime("%H:%M:%S")
                target_pnl = self.config.profit_target
                stop_loss_pnl = -self.config.stop_loss
                progress_range = target_pnl - stop_loss_pnl
                progress_pct = ((current_pnl - stop_loss_pnl) / progress_range * 100) if progress_range > 0 else 0
                progress_pct = max(0, min(100, progress_pct))  # Clamp 0-100
                
                bar_length = 15
                filled = int(bar_length * progress_pct / 100)
                progress_bar = "█" * filled + "░" * (bar_length - filled)
                
                # Check time since last trade
                time_since_trade = (datetime.now() - last_trade_time).total_seconds()
                minutes_since = time_since_trade / 60
                
                print(
                    f"[{ts}] Loop #{loop_count} | Equity: ${account['equity']:,.2f} | "
                    f"P&L: ${current_pnl:+.2f} | "
                    f"No trade: {minutes_since:.0f}m"
                )
                
                # [RERUN BACKTEST] If 1 hour without trade
                if time_since_trade > no_trade_threshold:
                    msg = f"\n[!] No trades in 1 hour. Rerunning backtest for new strategy..."
                    print(msg)
                    log_message(msg)
                    
                    # Rerun backtest
                    import subprocess
                    import json
                    
                    script_dir = get_script_dir()
                    try:
                        process = subprocess.Popen(
                            ["python", "backtest.py"],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=script_dir
                        )
                        
                        # Stream output
                        for line in process.stdout:
                            print(line, end="", flush=True)
                        
                        process.wait(timeout=300)
                        
                        # Load new results
                        backtest_file = os.path.join(script_dir, "backtest_results.json")
                        with open(backtest_file, "r") as f:
                            results = json.load(f)
                        
                        if results:
                            # Calculate combo scores
                            for result in results:
                                pf = float(result.get('profit_factor', 1.0))
                                wr = float(result.get('win_rate', 50.0))
                                ret = float(result.get('total_return', 0.0))
                                sharpe = float(result.get('sharpe_ratio', 0.0))
                                
                                pf_score = min(pf * 50, 100) if pf > 0 else 0
                                wr_score = wr
                                ret_score = max(0, min((ret * 100 + 50) / 1, 100))
                                sharpe_score = max(0, min((sharpe + 5) / 10 * 100, 100))
                                
                                combo = (pf_score * 0.40 + wr_score * 0.30 + ret_score * 0.20 + sharpe_score * 0.10)
                                result['combo_score'] = combo
                            
                            sorted_results = sorted(results, key=lambda x: float(x.get("combo_score", -999)), reverse=True)
                            best = sorted_results[0]
                            self.config = TradingConfig(strategy=best['strategy_name'])
                            
                            msg = f"[+] New strategy selected: {best['strategy_name']} ({best['timeframe']}) | Score: {best['combo_score']:.1f}"
                            print(msg)
                            log_message(msg)
                            last_trade_time = datetime.now()
                    except Exception as e:
                        log_message(f"[ERROR] Backtest rerun failed: {e}")
                
                # [+] PROFIT TARGET CHECK
                if current_pnl >= self.config.profit_target:
                    msg = f"\n[+] PROFIT - Target reached! P&L: ${current_pnl:+.2f}"
                    print(msg)
                    log_message(msg)
                    self.close_all_positions()
                    return
                
                # [-] STOP LOSS CHECK
                if current_pnl <= -self.config.stop_loss:
                    msg = f"\n[-] STOP - Stop loss hit! P&L: ${current_pnl:+.2f}"
                    print(msg)
                    log_message(msg)
                    self.close_all_positions()
                    return
                
                # Check each symbol
                for symbol in self.config.symbols:
                    try:
                        df = self.get_historical_bars(symbol, self.config.lookback_bars)
                        if df.empty:
                            continue
                        
                        indicators = self.calculate_indicators(df)
                        
                        # Update dashboard with current price and indicators
                        dashboard_state.price = float(indicators['close'])
                        dashboard_state.high = float(df['high'].iloc[-1]) if 'high' in df.columns else 0
                        dashboard_state.low = float(df['low'].iloc[-1]) if 'low' in df.columns else 0
                        dashboard_state.volume = float(df['volume'].iloc[-1]) if 'volume' in df.columns else 0
                        dashboard_state.timestamp = datetime.now().isoformat()
                        dashboard_state.indicators = {
                            'rsi': float(indicators.get('rsi', 0) or 0),
                            'sma_fast': float(indicators.get('sma_fast', 0) or 0),
                            'sma_slow': float(indicators.get('sma_slow', 0) or 0),
                            'ema_fast': float(indicators.get('ema_fast', 0) or 0),
                            'ema_slow': float(indicators.get('ema_slow', 0) or 0),
                            'macd': float(indicators.get('macd', 0) or 0),
                            'macd_signal': float(indicators.get('macd_signal', 0) or 0),
                            'bb_upper': float(indicators.get('bb_upper', 0) or 0),
                            'bb_middle': float(indicators.get('bb_middle', 0) or 0),
                            'bb_lower': float(indicators.get('bb_lower', 0) or 0),
                            'stochastic': float(indicators.get('stochastic', 0) or 0),
                            'atr': float(indicators.get('atr', 0) or 0),
                            'momentum': float(indicators.get('momentum', 0) or 0),
                        }
                        
                        # ========== CHECK PROFIT TARGET & STOP LOSS ==========
                        try:
                            position = self.trading_client.get_open_position(symbol)
                            if position and float(position.qty) != 0:
                                try:
                                    entry_price = float(position.avg_fill_price) if position.avg_fill_price else None
                                    if not entry_price:
                                        # Skip if we can't determine entry price
                                        pass
                                    else:
                                        current_price = indicators['close']
                                        position_qty = abs(float(position.qty))
                                        
                                        # Calculate P&L for this position
                                        if float(position.qty) > 0:  # Long position
                                            position_pnl = (current_price - entry_price) * position_qty
                                        else:  # Short position
                                            position_pnl = (entry_price - current_price) * position_qty
                                        
                                        # Check profit target
                                        if self.config.profit_target and position_pnl >= self.config.profit_target:
                                            ts = datetime.now().strftime("%H:%M:%S")
                                            msg = (
                                                f"[{ts}] 🎯 PROFIT TARGET HIT for {symbol}! "
                                                f"P&L: ${position_pnl:.2f} (target: ${self.config.profit_target:.2f})"
                                            )
                                            print(msg)
                                            log_message(msg)
                                            self.place_sell_order(symbol, position_qty)
                                            last_trade_time = datetime.now()
                                            continue  # Skip signal processing, position closed
                                        
                                        # Check stop loss
                                        elif self.config.stop_loss and position_pnl <= -self.config.stop_loss:
                                            ts = datetime.now().strftime("%H:%M:%S")
                                            msg = (
                                                f"[{ts}] ⛔ STOP LOSS HIT for {symbol}! "
                                                f"P&L: ${position_pnl:.2f} (stop: ${-self.config.stop_loss:.2f})"
                                            )
                                            print(msg)
                                            log_message(msg)
                                            self.place_sell_order(symbol, position_qty)
                                            last_trade_time = datetime.now()
                                            continue  # Skip signal processing, position closed
                                except (ValueError, TypeError) as e:
                                    pass  # Skip if values can't be converted
                        except Exception as e:
                            pass  # No position or error checking
                        
                        signal = self.generate_signal(symbol, indicators)
                        
                        # DEBUG: Show what happened with signal generation
                        if signal is None:
                            ts = datetime.now().strftime("%H:%M:%S")
                            # Only print occasionally to avoid spam (every 10 loops)
                            if loop_count % 10 == 0:
                                print(f"[{ts}] [DEBUG] No signal for {symbol} (waiting for setup)")
                        
                        # Check if we already have an open position
                        try:
                            existing_position = self.trading_client.get_open_position(symbol)
                            has_position = existing_position and float(existing_position.qty) != 0
                        except Exception:
                            has_position = False
                        
                        if signal == "BUY":
                            # Only BUY if we don't have an open position
                            if has_position:
                                if loop_count % 10 == 0:
                                    ts = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{ts}] [SKIP] Already have position in {symbol}. Monitoring for exit signal.")
                            else:
                                qty = self.get_position_size(symbol, indicators['close'])
                                if qty > 0:
                                    price = indicators['close']
                                    msg = (
                                        f"BUY signal {symbol} | "
                                        f"Price: ${price:.2f} | SMA({self.config.sma_fast}): {indicators['sma_fast']:.2f} | "
                                        f"SMA({self.config.sma_slow}): {indicators['sma_slow']:.2f}"
                                    )
                                    log_message(msg)
                                    self.place_buy_order(symbol, qty)
                                    last_trade_time = datetime.now()
                        
                        elif signal == "SELL":
                            # Check if we have position to close
                            if has_position:
                                qty = abs(float(existing_position.qty))
                                msg = (
                                    f"SELL signal {symbol} | "
                                    f"Price: ${indicators['close']:.2f}"
                                )
                                log_message(msg)
                                self.place_sell_order(symbol, qty)
                                last_trade_time = datetime.now()
                            else:
                                if loop_count % 10 == 0:
                                    ts = datetime.now().strftime("%H:%M:%S")
                                    print(f"[{ts}] [SKIP] No position to close for {symbol}")
                    
                    except Exception as e:
                        log_message(f"[ERROR] Processing {symbol}: {e}")
                
                time.sleep(self.config.check_interval)
        
        except KeyboardInterrupt:
            msg = "\n[STOP] Manual stop. Positions remain open."
            log_message(msg)


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the trading bot"""
    print("=" * 80)
    print("Alpaca Crypto Algorithmic Trading System")
    print("=" * 80)
    
    # ================================================================
    # TOGGLE 1: Paper vs Live Trading (FIRST - must set credentials)
    # ================================================================
    trading_mode = input("\n[?] Trading mode - (p)aper or (l)ive? [p]: ").strip().lower()
    IS_LIVE = trading_mode.startswith("l")
    
    # Set credentials based on mode
    set_credentials(IS_LIVE)
    
    if IS_LIVE:
        print("\n[!] LIVE TRADING MODE SELECTED")
    else:
        print("\n[OK] PAPER TRADING MODE (Safe)")
    
    # ================================================================
    # TOGGLE 2: Run Backtest?
    # ================================================================
    backtest_answer = input("[?] Run backtesting first? (y/N): ").strip().lower()
    RUN_BACKTEST = backtest_answer.startswith("y")
    
    if RUN_BACKTEST:
        print("\n[INFO] Running backtest to find best strategy...")
        import subprocess
        
        # Get the correct script directory (works for both EXE and Python)
        script_dir = get_script_dir()
        
        # Run backtest.py with real-time output
        try:
            process = subprocess.Popen(
                ["python", "backtest.py"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=script_dir
            )
            
            # Stream output in real-time
            stdout_data = ""
            for line in process.stdout:
                print(line, end="", flush=True)
                stdout_data += line
            
            process.wait(timeout=300)
            
            stderr_data = process.stderr.read()
            if stderr_data:
                print("[ERROR]", stderr_data)
            
            # Try to load and display results
            backtest_file = os.path.join(script_dir, "backtest_results.json")
            try:
                with open(backtest_file, "r") as f:
                    results = json.load(f)
                    
                if results:
                    # Calculate combo score for each result
                    for result in results:
                        pf = float(result.get('profit_factor', 1.0))
                        wr = float(result.get('win_rate', 50.0))
                        ret = float(result.get('total_return', 0.0))
                        sharpe = float(result.get('sharpe_ratio', 0.0))
                        
                        # Normalize to 0-100 scale
                        pf_score = min(pf * 50, 100) if pf > 0 else 0
                        wr_score = wr  # already 0-100
                        ret_score = max(0, min((ret * 100 + 50) / 1, 100))
                        sharpe_score = max(0, min((sharpe + 5) / 10 * 100, 100))
                        
                        # Weighted combo score
                        combo = (pf_score * 0.40 + wr_score * 0.30 + ret_score * 0.20 + sharpe_score * 0.10)
                        result['combo_score'] = combo
                    
                    # Sort by combo_score (highest profit first)
                    sorted_results = sorted(
                        results, 
                        key=lambda x: float(x.get("combo_score", -999)), 
                        reverse=True
                    )
                    
                    print("\n" + "="*80)
                    print("TOP 5 STRATEGIES BY COMBO SCORE")
                    print("="*80)
                    for i, result in enumerate(sorted_results[:5], 1):
                        print(f"\n{i}. {result['strategy_name']} ({result['timeframe']})")
                        print(f"   Combo Score: {result.get('combo_score', 0):.1f}")
                        print(f"   Profit Factor: {result.get('profit_factor', 'N/A')}")
                        print(f"   Win Rate: {result.get('win_rate', 'N/A')}%")
                        print(f"   Return: {result.get('total_return', 'N/A')}%")
                        print(f"   Sharpe: {result.get('sharpe_ratio', 'N/A')}")
                    
                    best = sorted_results[0]
                    selected_strategy = best['strategy_name']
                    selected_symbol = best['symbol']
                    selected_timeframe = best['timeframe']
                    print(f"\n[+] Selected: {selected_strategy} ({selected_timeframe}) on {selected_symbol}")
            except Exception as e:
                print(f"[WARNING] Could not load backtest results: {e}")
                selected_strategy = "SMA"
                selected_symbol = "BTC/USD"
                selected_timeframe = "1Min"
        except Exception as e:
            print(f"[ERROR] Backtest failed: {e}")
            selected_strategy = "SMA"
            selected_symbol = "BTC/USD"
            selected_timeframe = "1Min"
    else:
        # Try to load from existing backtest results JSON
        print("\n[INFO] Skipped backtest. Checking for existing backtest results...")
        results_file = "backtest_results.json"
        selected_strategy = None
        selected_symbol = None
        selected_timeframe = None
        
        if os.path.exists(results_file):
            try:
                with open(results_file, "r") as f:
                    results = json.load(f)
                
                if results:
                    # Calculate combo scores for existing results
                    for result in results:
                        pf = float(result.get("profit_factor", 1.0))
                        wr = float(result.get("win_rate", 50))
                        ret = float(result.get("total_return", 0)) / 100
                        sharpe = float(result.get("sharpe_ratio", 0))
                        
                        pf_score = min(pf * 50, 100) if pf > 0 else 0
                        wr_score = wr
                        ret_score = max(0, min((ret * 100 + 50) / 1, 100))
                        sharpe_score = max(0, min((sharpe + 5) / 10 * 100, 100))
                        
                        combo = (pf_score * 0.40 + wr_score * 0.30 + ret_score * 0.20 + sharpe_score * 0.10)
                        result['combo_score'] = combo
                    
                    # Get best result
                    sorted_results = sorted(
                        results, 
                        key=lambda x: float(x.get("combo_score", -999)), 
                        reverse=True
                    )
                    
                    best = sorted_results[0]
                    selected_strategy = best['strategy_name']
                    selected_symbol = best['symbol']
                    selected_timeframe = best['timeframe']
                    print(f"[+] Loaded best strategy: {selected_strategy} ({selected_timeframe}) on {selected_symbol}")
            except Exception as e:
                print(f"[WARNING] Could not load backtest results: {e}")
        
        # Fallback if no strategy selected
        if not selected_strategy:
            print("[WARNING] No backtest results found. Please run backtest first.")
            print("[INFO] To run backtest, restart and answer 'y' to backtest question.")
            selected_strategy = "SMA"
            selected_symbol = "BTC/USD"
            selected_timeframe = "1Min"
            print(f"[FALLBACK] Using: {selected_strategy} on {selected_symbol}")
    
    # Create fresh config with user choices - use the selected strategy, symbol, and timeframe
    config = TradingConfig(paper_trading=not IS_LIVE, strategy=selected_strategy, symbols=[selected_symbol], timeframe=selected_timeframe)

    
    # Display configuration
    mode = "LIVE" if IS_LIVE else "PAPER"
    print(f"\n{'='*80}")
    print(f"[MODE] {mode} Trading")
    print(f"Strategy: {config.strategy}")
    print(f"Symbols: {config.symbols}")
    print(f"Position Size: {config.position_size_pct}% of account")
    print(f"Stop Loss: ${config.stop_loss:.2f}")
    print(f"Profit Target: ${config.profit_target:.2f}\n")
    
    # Final confirmation for LIVE
    if IS_LIVE:
        confirm = input("[!] LIVE TRADING MODE. Type 'YES' to confirm: ").strip().upper()
        if confirm != "YES":
            print("Aborted. Exiting.")
            return
    
    # Start Flask dashboard server in background thread
    if FLASK_AVAILABLE:
        dashboard_thread = threading.Thread(target=start_dashboard_server, daemon=True)
        dashboard_thread.start()
        print("[+] Dashboard: http://localhost:5000\n")
    else:
        print("[!] Flask not installed. Dashboard disabled. (pip install flask)\n")
    
    # Start bot
    bot = AlpacaCryptoBot(config)
    
    # Check connectivity
    try:
        account = bot.get_account()
        log_message(f"[OK] Connected. Account equity: ${account['equity']:,.2f}")
    except Exception as e:
        log_message(f"[ERROR] Failed to connect: {e}")
        return
    
    # Run trading loop
    bot.run_trading_loop()


if __name__ == "__main__":
    main()
