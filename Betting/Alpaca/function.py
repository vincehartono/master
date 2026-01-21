import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import alpaca_trade_api as tradeapi
import os
from dotenv import load_dotenv
from alpaca.data import StockHistoricalDataClient, TimeFrame, TimeFrameUnit
from alpaca.data.requests import StockBarsRequest

def fetch_stock_data(symbol, start_date, end_date):
    # Load environment variables
    load_dotenv()
    API_KEY = os.getenv('APCA_API_KEY_ID')
    SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

    # Instantiate a data client
    data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    # Define the time frame for daily data
    timeframe_daily = TimeFrame(1, TimeFrameUnit.Day)

    # Create the request parameters
    request_params = StockBarsRequest(
        symbol_or_symbols=[symbol],
        timeframe=timeframe_daily,
        start=start_date,
        end=end_date
    )

    # Fetch the daily bars data
    response = data_client.get_stock_bars(request_params)

    # Extract the data from the 'data' attribute
    bars_data = response.data[symbol]

    # Convert to DataFrame
    bars_df = pd.DataFrame([bar.__dict__ for bar in bars_data])

    # Convert timestamp to datetime and set index
    bars_df['timestamp'] = pd.to_datetime(bars_df['timestamp'])
    bars_df.set_index(['timestamp'], inplace=True)

    # Convert to desired timezone
    bars_df = bars_df.tz_convert('America/New_York')

    return bars_df

def adjust_for_splits(stock_data, split_ratios):
    for split_ratio, split_date in split_ratios:
        split_date = pd.to_datetime(split_date).tz_localize('America/New_York')
        # Adjust prices
        stock_data.loc[:split_date, 'close'] /= split_ratio
        # Adjust volumes
        stock_data.loc[:split_date, 'volume'] *= split_ratio
    return stock_data

def main():
    symbol = 'ASML'
    parquet_path = r'C:\Users\Vince\master\Betting\Alpaca\stock_data.parquet'
    split_ratios = [(10, '2024-07-20')]  # 10-to-1 split on 2024-07-20

    # Load existing data or initialize empty DataFrame
    if os.path.exists(parquet_path):
        existing_data = pd.read_parquet(parquet_path)
        last_timestamp = existing_data.index.max()
        start_date = (last_timestamp + timedelta(days=1)).tz_localize(None)
    else:
        existing_data = pd.DataFrame()
        start_date = datetime(2024, 1, 1)

    # Use today as the end date
    end_date = datetime.now()

    # Fetch new stock data
    new_data = fetch_stock_data(symbol, start_date, end_date)

    # Combine and drop duplicates
    combined_data = pd.concat([existing_data, new_data])
    combined_data = combined_data[~combined_data.index.duplicated(keep='last')]

    # Adjust for splits
    combined_data = adjust_for_splits(combined_data, split_ratios)

    # Save updated data
    combined_data.to_parquet(parquet_path)

    # Print stats
    print(f"Stock data updated and saved to {parquet_path}")
    print(f"Total records: {len(combined_data)}")
    print(f"Date range: {combined_data.index.min().date()} to {combined_data.index.max().date()}")

if __name__ == "__main__":
    main()
