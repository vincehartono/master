import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import re

def parse_option_description(symbol, description):
    """Parse option info from symbol and description"""
    
    ticker = None
    expiry_date = None
    strike = None
    option_type = None
    
    if not description or pd.isna(description):
        return None, None, None, None
    
    desc = str(description).strip().upper()
    
    # Extract option type (CALL or PUT)
    if 'PUT' in desc:
        option_type = 'P'
    elif 'CALL' in desc:
        option_type = 'C'
    else:
        return None, None, None, None
    
    # Try to get ticker from Symbol column first (cleaner source)
    if symbol and not pd.isna(symbol):
        symbol_str = str(symbol).strip().upper()
        # Extract first word/letters as ticker
        ticker_match = re.match(r'^([A-Z]+)', symbol_str)
        if ticker_match:
            ticker = ticker_match.group(1)
    
    # If no ticker from Symbol, extract from Description
    if not ticker:
        words = desc.split()
        for i, word in enumerate(words):
            if word in ['PUT', 'CALL']:
                if i + 1 < len(words):
                    # Take first letters of next word
                    next_word = words[i + 1]
                    ticker_match = re.match(r'^([A-Z]+)', next_word)
                    if ticker_match:
                        ticker = ticker_match.group(1)
                    break
    
    # Extract strike from "IN$25" or "$6" or "TECHNOLOGIE$6" patterns
    strike_match = re.search(r'\$?([\d.]+)(?:\s|$|[A-Z])', desc)
    if strike_match:
        try:
            strike = float(strike_match.group(1))
        except:
            pass
    
    # Extract expiry from "EXP 01/23/26" or "EXP 01/30/26" pattern
    expiry_match = re.search(r'EXP\s+(\d{1,2})/(\d{1,2})/(\d{2,4})', desc)
    if expiry_match:
        month = expiry_match.group(1)
        day = expiry_match.group(2)
        year = expiry_match.group(3)
        
        # Handle 2-digit year
        if len(year) == 2:
            year = f"20{year}"
        
        try:
            expiry_date = datetime.strptime(f"{month}/{day}/{year}", "%m/%d/%Y").date()
        except:
            pass
    
    return ticker, expiry_date, strike, option_type

def main():
    # Get the Options folder path
    options_dir = Path(__file__).parent
    
    # Create results folder
    results_dir = options_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Find CSV files in the directory
    csv_files = list(options_dir.glob("*.csv"))
    
    if not csv_files:
        print("No CSV files found in Options folder")
        return
    
    print(f"Found {len(csv_files)} CSV file(s):\n")
    
    combined_data = []
    
    # Merge and deduplicate multiple CSV files if present
    merged_df = None
    for csv_file in csv_files:
        df_temp = pd.read_csv(csv_file)
        print(f"  {csv_file.name}: {len(df_temp)} rows")
        if merged_df is None:
            merged_df = df_temp
        else:
            merged_df = pd.concat([merged_df, df_temp], ignore_index=True)
    
    # Remove duplicates, keeping the first occurrence
    if merged_df is not None:
        print(f"\nBefore deduplication: {len(merged_df)} total rows")
        merged_df = merged_df.drop_duplicates(keep='first')
        print(f"After deduplication: {len(merged_df)} unique rows")
        print(f"Duplicates removed: {sum(1 for csv_file in csv_files for _ in pd.read_csv(csv_file)) - len(merged_df)}\n")
    
    # Process the merged dataframe
    if merged_df is not None:
        df = merged_df
        
        # Check if this looks like a transaction file
        if 'Action' in df.columns and 'Description' in df.columns:
            # This is a transaction file
            
            output_rows = []
            
            for idx, row in df.iterrows():
                symbol = row.get('Symbol', '')
                description = row.get('Description', '')
                
                # Skip non-option transactions
                if not description or pd.isna(description):
                    continue
                if 'call' not in str(description).lower() and 'put' not in str(description).lower():
                    continue
                
                ticker, expiry_date, strike, option_type = parse_option_description(symbol, description)
                
                if not ticker or not expiry_date or not strike or not option_type:
                    continue
                
                # Determine action (BTO, STO, etc.)
                action = str(row.get('Action', '')).upper()
                is_sell = 'SELL' in action
                is_buy = 'BUY' in action
                
                # Get amount (premium received or paid)
                amount = None
                for amt_col in ['Amount', 'amount', 'Price', 'price', 'Premium', 'premium']:
                    if amt_col in row and not pd.isna(row[amt_col]):
                        try:
                            # Remove $ and convert to float
                            amt_str = str(row[amt_col]).replace('$', '').replace(',', '').strip()
                            amount = float(amt_str)
                            break
                        except Exception as e:
                            pass
                
                if amount is None:
                    amount = 0
                
                # Get transaction date
                try:
                    trans_date = pd.to_datetime(row.get('Date', None)).date()
                except:
                    trans_date = None
                
                output_rows.append({
                    'ticker': ticker,
                    'expiry': expiry_date,
                    'option_type': option_type,
                    'strike': strike,
                    'action': action,
                    'is_sell': is_sell,
                    'is_buy': is_buy,
                    'trans_date': trans_date,
                    'amount': amount,
                    'description': description
                })
            
            # Combine all trades for the same ticker+expiry+type into ONE row (complete position)
            combined_rows = []
            processed = set()
            
            # Group by ticker+expiry+type (entire position)
            position_groups = {}
            for i, trans in enumerate(output_rows):
                key = (trans['ticker'], trans['expiry'], trans['option_type'])
                if key not in position_groups:
                    position_groups[key] = []
                position_groups[key].append((i, trans))
            
            # Process each position
            for position_key, transactions in position_groups.items():
                # Get all buys and sells
                buys = [(i, t) for i, t in transactions if t['is_buy']]
                sells = [(i, t) for i, t in transactions if t['is_sell']]
                
                if buys and sells:
                    # Calculate totals and dates
                    buy_amount = sum(t['amount'] for i, t in buys)
                    sell_amount = sum(t['amount'] for i, t in sells)
                    
                    # Find first buy date and last sell date (or vice versa)
                    buy_dates = [t['trans_date'] for i, t in buys]
                    sell_dates = [t['trans_date'] for i, t in sells]
                    
                    if buy_dates:
                        first_buy = min(buy_dates)
                        last_buy = max(buy_dates)
                    if sell_dates:
                        first_sell = min(sell_dates)
                        last_sell = max(sell_dates)
                    
                    # Determine overall date_bought and date_sold
                    # Use: earliest buy (if exists), latest sell (if exists)
                    date_bought = first_buy if buy_dates else None
                    date_sold = last_sell if sell_dates else None
                    
                    # P&L calculation
                    premium = abs(sell_amount)  # Total amount received from sells
                    profit = sell_amount + buy_amount  # Received - Paid
                    
                    # Days
                    days = 0
                    if date_bought and date_sold:
                        days = (date_sold - date_bought).days
                    elif date_bought and position_key[1]:  # Use expiry if no sell date
                        days = (position_key[1] - date_bought).days
                    
                    # Get all strikes
                    all_strikes = sorted(set([t['strike'] for i, t in transactions]))
                    
                    combined_rows.append({
                        'ticker': position_key[0],
                        'expiry': position_key[1],
                        'option_type': position_key[2],
                        'date_bought': date_bought,
                        'date_sold': date_sold,
                        'low_K': all_strikes[0],
                        'high_K': all_strikes[-1] if len(all_strikes) > 1 else None,
                        'premium': premium,
                        'profit': profit,
                        'days': days
                    })
                    
                    for i, t in transactions:
                        processed.add(i)
                else:
                    # Only buys or only sells
                    for i, trans in transactions:
                        combined_rows.append({
                            'ticker': trans['ticker'],
                            'expiry': trans['expiry'],
                            'option_type': trans['option_type'],
                            'date_bought': trans['trans_date'] if trans['is_buy'] else None,
                            'date_sold': trans['trans_date'] if trans['is_sell'] else None,
                            'low_K': trans['strike'],
                            'high_K': None,
                            'premium': abs(trans['amount']),
                            'profit': trans['amount'],
                            'days': 0
                        })
                        processed.add(i)
            
            combined_data.extend(combined_rows)
    
    # Save to CSV
    if combined_data:
        combined_df = pd.DataFrame(combined_data)
        
        # Save combined CSV
        output_file = results_dir / "options_combined.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"\n{'='*80}")
        print(f"Saved combined options to {output_file.name} ({len(combined_df)} rows)\n")
        print(combined_df)
        
        # Create ticker summary
        print(f"\n{'='*80}")
        print(f"Creating ticker summary...")
        print(f"{'='*80}\n")
        
        ticker_summary = combined_df.groupby('ticker', as_index=False).agg({
            'profit': 'sum'
        }).rename(columns={'profit': 'profit_loss'})
        
        ticker_summary = ticker_summary.sort_values('profit_loss', ascending=False)
        
        # Save ticker summary
        ticker_file = results_dir / "ticker_summary.csv"
        ticker_summary.to_csv(ticker_file, index=False)
        print(f"Saved ticker summary to {ticker_file.name} ({len(ticker_summary)} tickers)\n")
        print(ticker_summary)
    else:
        print("No option data to process")

if __name__ == "__main__":
    main()
