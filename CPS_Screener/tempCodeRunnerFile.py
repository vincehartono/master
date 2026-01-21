"""
Reddit-style Cash-Secured Put (CSP) screener using yfinance.

What it does
- Pulls spot price + option chains from yfinance
- Computes: put delta (Black–Scholes using contract IV), premium(mid), return, annualized yield (AY),
  PoP ~ 1-|delta|, bid/ask spread %, cushion %, RSI(14), ADX(14), collateral
- Filters for "redditor-like" trades (delta band, DTE band, min return, max spread, min cushion)
- Ranks with a "RedditScore" that rewards AY/PoP/cushion and penalizes spread + extreme RSI/very high ADX

Notes
- yfinance sometimes has missing bid/ask or IV. The code falls back gracefully.
- Put delta is computed from IV; broker greeks will differ slightly.
- Risk-free rate r is a parameter; adjust as needed.

Install
pip install yfinance pandas numpy
"""

import math
from datetime import datetime, date
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------
# Math helpers
# ----------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_put_delta(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black–Scholes put delta = N(d1) - 1."""
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    return norm_cdf(d1) - 1.0


# ----------------------------
# Indicators: RSI(14), ADX(14)
# ----------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    # Directional movement
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    # True range
    tr1 = (high - low)
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()

    plus_di = 100 * (plus_dm.rolling(period).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr.replace(0, np.nan))

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


# ----------------------------
# Scoring (tuned to your samples)
# ----------------------------
def redditor_score(row: pd.Series) -> float:
    """
    Higher is better.
    Uses values stored as percents for AY/PoP/Spread/Cushion in the dataframe.
    """
    ay = float(row["AY"]) / 100.0
    pop = float(row["PoP"]) / 100.0
    cushion = float(row["Cushion"]) / 100.0

    spread = float(row["Spread"]) / 100.0 if pd.notna(row["Spread"]) else 0.20
    rsi_val = float(row["RSI"]) if pd.notna(row["RSI"]) else 50.0
    adx_val = float(row["ADX"]) if pd.notna(row["ADX"]) else 20.0

    # Reward: annualized yield + PoP + cushion
    score = 2.2 * ay + 0.6 * pop + 0.8 * cushion

    # Penalize wide spreads strongly
    score -= 1.5 * spread

    # Penalize RSI extremes (soft)
    if rsi_val > 65:
        score -= 0.08 * ((rsi_val - 65) / 35)  # scaled penalty
    if rsi_val < 35:
        score -= 0.08 * ((35 - rsi_val) / 35)

    # Mild penalty for very high ADX (strong trend can mean more gap risk); not a hard filter
    if adx_val > 35:
        score -= 0.05 * ((adx_val - 35) / 35)

    return float(score)


# ----------------------------
# Core screener
# ----------------------------
def screen_csp(
    tickers: List[str],
    # Filters approximating the redditor's style
    min_dte: int = 7,
    max_dte: int = 45,
    delta_low: float = -0.30,
    delta_high: float = -0.20,
    min_return: float = 0.005,   # 0.50% (premium/strike)
    max_spread: float = 0.10,    # 10% of mid
    min_cushion: float = 0.01,   # 1% OTM
    # Data / assumptions
    r: float = 0.045,            # risk-free rate assumption (annual)
    hist_lookback: str = "6mo",
    min_volume: Optional[int] = None,   # set e.g. 50 to require option volume
    min_oi: Optional[int] = None,       # set e.g. 200 to require open interest
    only_otm: bool = True,
    top_n_per_ticker: int = 10
) -> pd.DataFrame:
    rows = []
    today = date.today()

    for tkr in tickers:
        try:
            tk = yf.Ticker(tkr)

            # --- Spot price (fallback chain)
            S = None
            try:
                S = float(tk.fast_info.get("last_price", None))
            except Exception:
                S = None

            if not S or np.isnan(S):
                try:
                    S = float(tk.info.get("regularMarketPrice", None))
                except Exception:
                    S = None

            if not S or np.isnan(S):
                hist_short = tk.history(period="5d")
                if hist_short.empty:
                    continue
                S = float(hist_short["Close"].iloc[-1])

            # --- Technicals
            hist = tk.history(period=hist_lookback)
            if hist.empty or len(hist) < 50:
                rsi_val = np.nan
                adx_val = np.nan
            else:
                rsi_series = rsi(hist["Close"], 14)
                adx_series = adx(hist["High"], hist["Low"], hist["Close"], 14)
                rsi_val = float(rsi_series.iloc[-1]) if pd.notna(rsi_series.iloc[-1]) else np.nan
                adx_val = float(adx_series.iloc[-1]) if pd.notna(adx_series.iloc[-1]) else np.nan

            expirations = tk.options
            if not expirations:
                continue

            for exp in expirations:
                exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                dte = (exp_dt - today).days
                if dte < min_dte or dte > max_dte:
                    continue

                chain = tk.option_chain(exp)
                puts = chain.puts.copy()
                if puts.empty:
                    continue

                # Optional liquidity filters
                if min_volume is not None and "volume" in puts.columns:
                    puts = puts[puts["volume"].fillna(0).astype(int) >= min_volume]
                if min_oi is not None and "openInterest" in puts.columns:
                    puts = puts[puts["openInterest"].fillna(0).astype(int) >= min_oi]

                # OTM filter
                if only_otm:
                    puts = puts[puts["strike"] <= S].copy()

                if puts.empty:
                    continue

                # Compute per-option metrics, then keep best few per ticker/expiry to reduce noise
                tmp = []
                for _, opt in puts.iterrows():
                    K = float(opt["strike"])
                    bid = float(opt.get("bid", np.nan))
                    ask = float(opt.get("ask", np.nan))
                    last = float(opt.get("lastPrice", np.nan))
                    iv = float(opt.get("impliedVolatility", np.nan))

                    # Premium mid
                    if np.isfinite(bid) and np.isfinite(ask) and ask > 0:
                        mid = (bid + ask) / 2.0
                    else:
                        mid = last

                    if not np.isfinite(mid) or mid <= 0:
                        continue
                    if not np.isfinite(iv) or iv <= 0:
                        continue

                    T = dte / 365.0
                    delta = bs_put_delta(S, K, T, r, iv)
                    if not np.isfinite(delta):
                        continue

                    # Filter: delta band
                    if not (delta_low <= delta <= delta_high):
                        continue

                    ret = mid / K
                    if ret < min_return:
                        continue

                    ay = ret * (365.0 / dte)

                    spread = np.nan
                    if np.isfinite(bid) and np.isfinite(ask) and mid > 0:
                        spread = (ask - bid) / mid
                        if spread < 0:
                            spread = np.nan

                    cushion = (S - K) / S
                    if cushion < min_cushion:
                        continue

                    if np.isfinite(spread) and spread > max_spread:
                        continue

                    pop = 1.0 - abs(delta)  # common shortcut

                    tmp.append({
                        "Ticker": tkr,
                        "ExpiryISO": exp_dt.isoformat(),
                        "Expiry": exp_dt.strftime("%-m/%-d"),
                        "DTE": dte,
                        "Spot": round(S, 2),
                        "Strike": K,
                        "Δ": round(delta, 2),
                        "Premium": round(mid, 2),
                        "IV": round(iv * 100, 0),
                        "Return": round(ret * 100, 2),        # %
                        "AY": round(ay * 100, 0),             # %
                        "PoP": round(pop * 100, 0),           # %
                        "Spread": (round(spread * 100, 0) if np.isfinite(spread) else np.nan),  # %
                        "Cushion": round(cushion * 100, 0),   # %
                        "RSI": (round(rsi_val, 0) if np.isfinite(rsi_val) else np.nan),
                        "ADX": (round(adx_val, 0) if np.isfinite(adx_val) else np.nan),
                        "Collat": f"${K*100:,.0f}",
                        "Volume": int(opt.get("volume", 0) or 0),
                        "OI": int(opt.get("openInterest", 0) or 0),
                    })

                if not tmp:
                    continue

                tmp_df = pd.DataFrame(tmp)

                # Keep top candidates per (ticker, expiry) by AY first, then lower spread, then higher cushion
                tmp_df = tmp_df.sort_values(by=["AY", "Spread", "Cushion"], ascending=[False, True, False]).head(top_n_per_ticker)
                rows.extend(tmp_df.to_dict("records"))

        except Exception:
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Add score + final sort
    df["RedditScore"] = df.apply(redditor_score, axis=1)
    df = df.sort_values(
        by=["RedditScore", "AY", "Spread", "Cushion"],
        ascending=[False, False, True, False]
    ).reset_index(drop=True)

    # Match your column order style (plus a few extras you might like)
    cols = [
        "Ticker", "Expiry", "Strike", "Δ", "Premium", "IV", "Return", "AY", "PoP",
        "Spread", "Cushion", "RSI", "ADX", "Collat",
        "DTE", "Spot", "Volume", "OI", "RedditScore", "ExpiryISO"
    ]
    df = df[[c for c in cols if c in df.columns]]
    return df


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # Use your observed universe; you can expand this to SP500, watchlist, etc.
    tickers = [
        "GOOG", "MSFT", "DHR", "HAL", "BABA", "DVN", "ATI", "IBM", "WYNN", "NEE", "GE",
        "RCL", "ANET", "BBY", "BWXT", "EQT", "CEG", "EBAY",
        "NUE", "GM", "AAPL", "BAC", "PSX", "NU", "GILD", "AMAT", "RL"
    ]

    df = screen_csp(
        tickers=tickers,
        min_dte=7,
        max_dte=45,
        delta_low=-0.30,
        delta_high=-0.20,
        min_return=0.005,
        max_spread=0.10,
        min_cushion=0.01,
        r=0.045,
        hist_lookback="6mo",
        # Optional liquidity constraints:
        # min_volume=50,
        # min_oi=200,
        only_otm=True,
        top_n_per_ticker=10
    )

    if df.empty:
        print("No matches found with current filters.")
    else:
        # Print in a compact table
        print(df.to_string(index=False))

        # Save to CSV
        df.to_csv("csp_candidates.csv", index=False)
        print("\nSaved: csp_candidates.csv")
