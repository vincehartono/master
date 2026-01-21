from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime, timedelta
import pandas as pd
from dotenv import load_dotenv
import os

# Load API key and secret from .env
load_dotenv()

# Alpaca will automatically pick up the keys from environment variables
client = StockHistoricalDataClient()

# Define parameters
symbol = "ASML"
end_date = datetime.now()
start_date = end_date - timedelta(days=5)

# Request 1-min bars, will resample to 5-min
request_params = StockBarsRequest(
    symbol_or_symbols=symbol,
    timeframe=TimeFrame.Minute,
    start=start_date,
    end=end_date
)

# Fetch data
bars = client.get_stock_bars(request_params).df

# Filter for ASML (index is multi-index: [symbol, timestamp])
df = bars.loc[symbol].copy()
df.index = pd.to_datetime(df.index)

# Resample to 5-minute
df_5min = df.resample("5min").agg({
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum"
}).dropna()

# Save to Parquet
parquet_path = r"C:\Users\Vince\master\Betting\Stocks\asml_alpaca.parquet"
df_5min.to_parquet(parquet_path)

print(f"✅ Saved {len(df_5min)} rows to {parquet_path}")
