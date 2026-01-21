import yfinance as yf
import pandas as pd
from datetime import date, datetime


def main() -> None:
    tk = yf.Ticker("AAPL")
    today = date.today()

    print("Fetching AAPL spot...")
    S = None
    try:
        S = float(tk.fast_info.get("last_price", None))
    except Exception:
        S = None
    print("fast_info last_price:", S)

    if not S or pd.isna(S):
        try:
            S = float(tk.info.get("regularMarketPrice", None))
        except Exception:
            S = None
    print("info regularMarketPrice:", S)

    if not S or pd.isna(S):
        hist_short = tk.history(period="5d")
        print("5d history shape:", hist_short.shape)
        if not hist_short.empty:
            S = float(hist_short["Close"].iloc[-1])
    print("final S:", S)

    expirations = tk.options
    print("expirations:", expirations)

    all_puts = []
    if expirations:
        for exp in expirations:
            exp_dt = datetime.strptime(exp, "%Y-%m-%d").date()
            dte = (exp_dt - today).days
            print(f"  exp {exp} DTE {dte}")
            chain = tk.option_chain(exp)
            puts = chain.puts.copy()
            print("    puts shape:", puts.shape)
            if not puts.empty:
                puts["ExpiryISO"] = exp_dt.isoformat()
                all_puts.append(puts)

    if all_puts:
        df_all = pd.concat(all_puts, ignore_index=True)
        print("total puts rows:", df_all.shape)
        df_all.to_csv("CPS_Screener/aapl_puts_raw_debug.csv", index=False)
        print("saved CPS_Screener/aapl_puts_raw_debug.csv")
    else:
        print("no puts collected")


if __name__ == "__main__":
    main()

