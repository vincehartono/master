import os
from datetime import date, datetime
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf


TICKERS: List[str] = [
    "GOOG", "MSFT", "DHR", "HAL", "BABA", "DVN", "ATI", "IBM", "WYNN", "NEE", "GE",
    "RCL", "ANET", "BBY", "BWXT", "EQT", "CEG", "EBAY", "NUE", "GM", "AAPL", "BAC",
    "PSX", "NU", "GILD", "AMAT", "RL",
    "RKLB", "RDDT", "WDC", "LDI", "JOBY", "U", "NTLA", "OPEN", "PINS", "SEDG",
    "PLTR", "ENPH", "MRNA", "RXRX", "KTOS", "STX", "INTC", "SOFI", "NET", "TTD",
    "HOOD", "ROKU", "TOST", "COIN", "DNA", "BILL", "AMD", "MU", "CMPS", "AAP",
    "CRSP", "LYB", "SHOP", "CZR", "PATH", "PSKY", "CMG", "BBWI", "ORCL", "UAL",
    "APH", "ACHR", "FTNT", "CCJ", "TSLA", "TXN", "ISRG", "AVGO", "LUMN", "MOH",
    "FSLR", "NVDA", "GDX", "LLY", "PSTG", "NOW", "NEM", "NFLX", "TDOC", "KLAC",
    "META", "ALB", "FISV", "NCLH", "CSX", "TSM", "FCX", "GOOGL", "CVS", "SE", "F",
    "CCL", "DHI", "PYPL", "ARKK", "DOW", "BA", "UPS", "AES", "AAL", "ADP", "XOP",
    "OXY", "AMZN", "AXP", "GS", "SLB", "SMH", "CMCSA",
]


def main() -> None:
    today = date.today()
    rows = []

    print("Collecting raw puts for", len(TICKERS), "tickers")

    for tkr in TICKERS:
        try:
            print(f"--- {tkr} ---")
            tk = yf.Ticker(tkr)

            # Spot (best-effort)
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

            expirations = tk.options
            if not expirations:
                print(f"{tkr}: no expirations")
                continue

            for exp in expirations:
                try:
                    exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
                    dte = (exp_dt - today).days

                    chain = tk.option_chain(exp)
                    puts = chain.puts.copy()
                    if puts.empty:
                        continue

                    puts = puts.copy()
                    puts["Ticker"] = tkr
                    puts["ExpiryISO"] = exp_dt.isoformat()
                    puts["DTE"] = dte
                    puts["Spot"] = S

                    rows.append(puts)
                except Exception as inner_e:
                    print(f"{tkr} {exp}: error {inner_e}")
                    continue

        except Exception as outer_e:
            print(f"{tkr}: outer error {outer_e}")
            continue

    if not rows:
        print("No puts collected at all.")
        return

    df = pd.concat(rows, ignore_index=True)

    os.makedirs("CPS_Screener", exist_ok=True)
    out_path = os.path.join("CPS_Screener", "all_puts_raw.csv")
    df.to_csv(out_path, index=False)
    print("Saved raw puts to", out_path, "rows:", len(df))


if __name__ == "__main__":
    main()

