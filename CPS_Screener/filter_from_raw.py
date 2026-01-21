import os
from datetime import datetime, date

import numpy as np
import pandas as pd


def bs_put_delta(spot: float, strike: float, dte: int, r: float, iv: float) -> float:
    import math

    T = dte / 365.0
    if T <= 0 or iv <= 0 or spot <= 0 or strike <= 0:
        return float("nan")
    d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * T) / (iv * math.sqrt(T))
    return 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0))) - 1.0


def main() -> None:
    raw_path = os.path.join("CPS_Screener", "all_puts_raw.csv")
    if not os.path.exists(raw_path):
        print("Raw file not found:", raw_path)
        return

    df = pd.read_csv(raw_path)
    if df.empty:
        print("Raw file is empty.")
        return

    # Basic derived fields
    bid = df.get("bid")
    ask = df.get("ask")
    last = df.get("lastPrice")
    strike = df.get("strike")
    dte = df.get("DTE")
    spot = df.get("Spot")
    iv = df.get("impliedVolatility")

    mid = (bid + ask) / 2.0
    mid = mid.where((bid.notna()) & (ask.notna()) & (ask > 0), last)

    ret = mid / strike
    spread = (ask - bid) / mid
    spread = spread.where(spread >= 0)

    cushion = (spot - strike) / spot

    # Delta
    deltas = []
    r = 0.045
    for s, k, d, v in zip(spot, strike, dte, iv):
        try:
            deltas.append(bs_put_delta(float(s), float(k), int(d), r, float(v)))
        except Exception:
            deltas.append(float("nan"))
    df["Delta"] = deltas

    df["Mid"] = mid
    df["Return"] = ret
    df["SpreadPct"] = spread
    df["CushionPct"] = cushion

    # Filters – tightened to original screener style
    mask = pd.Series(True, index=df.index)
    # DTE: 7–45 days
    mask &= df["DTE"].between(7, 45)
    # Delta band: -0.30 to -0.20
    mask &= df["Delta"].between(-0.30, -0.20)
    # Minimum premium/strike: 0.5%
    mask &= df["Return"] >= 0.005
    # Bid/ask spread ≤ 10% of mid (or NaN allowed)
    mask &= (df["SpreadPct"].isna()) | (df["SpreadPct"] <= 0.10)
    # At least 1% OTM
    mask &= df["CushionPct"] >= 0.01

    filtered = df[mask].copy()
    if filtered.empty:
        print("No matches after filtering.")
        out_path = os.path.join("CPS_Screener", "csp_candidates_from_raw.csv")
        filtered.to_csv(out_path, index=False)
        print("Saved empty filtered file to", out_path)
        return

    print("Filtered rows:", len(filtered))

    # Compute a RedditScore-like rank: AY + PoP + Cushion minus Spread
    # Use percentages from our derived fields.
    ay = (filtered["Return"] * (365.0 / filtered["DTE"])).fillna(0.0)
    pop = (1.0 - filtered["Delta"].abs()).fillna(0.0)
    cushion_pct = filtered["CushionPct"].fillna(0.0)
    spread_pct = filtered["SpreadPct"].fillna(0.0)

    score = 2.2 * ay + 0.6 * pop + 0.8 * cushion_pct - 1.5 * spread_pct
    filtered["Score"] = score

    # Sort by score and assign Rank (1 = best)
    filtered = filtered.sort_values(by="Score", ascending=False).reset_index(drop=True)
    filtered["Rank"] = filtered.index + 1

    out_path = os.path.join("CPS_Screener", "csp_candidates_from_raw.csv")
    filtered.to_csv(out_path, index=False)
    print("Saved filtered candidates with ranks to", out_path)

    # Also print the top 10 CSPs with key fields for quick review
    top10 = filtered.head(10)
    desired_cols = ["Ticker", "ExpiryISO", "strike", "lastPrice", "Return"]
    missing = [c for c in desired_cols if c not in top10.columns]

    if not missing:
        print("\nTop 10 ranked CSPs (Ticker, ExpiryISO, strike, lastPrice, Return):")
        print(top10[desired_cols].to_string(index=False))
    else:
        print("\nTop 10 rows (some desired columns missing:", ", ".join(missing), "):")
        print(top10.to_string(index=False))


if __name__ == "__main__":
    main()
