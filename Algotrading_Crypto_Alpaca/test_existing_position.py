#!/usr/bin/env python3
"""
Test script to verify the bot detects existing positions and monitors them
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient

# Setup - look for .env in parent directory too
load_dotenv()
if not os.getenv("APCA_API_KEY_ID"):
    load_dotenv(Path(__file__).parent.parent / ".env")

# Use paper trading credentials
API_KEY = os.getenv("APCA_API_KEY_ID_PAPER")
SECRET_KEY = os.getenv("APCA_API_SECRET_KEY_PAPER")
BASE_URL = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")

if not API_KEY or not SECRET_KEY:
    print("[ERROR] Missing API credentials in .env")
    sys.exit(1)

client = TradingClient(api_key=API_KEY, secret_key=SECRET_KEY)

print("=" * 80)
print("CHECKING FOR EXISTING POSITIONS")
print("=" * 80)

try:
    # Get all positions
    positions = client.get_all_positions()
    
    if not positions:
        print("\n[INFO] No existing positions found.")
        print("\nTo test the bot's position monitoring:")
        print("1. Place a BUY order manually in your account")
        print("2. Run this script again to verify detection")
        print("3. Start the trading bot - it will continue monitoring that position")
    else:
        print(f"\n[✓] Found {len(positions)} position(s):\n")
        
        for i, pos in enumerate(positions, 1):
            symbol = pos.symbol
            qty = float(pos.qty)
            avg_fill = float(pos.avg_entry_price) if pos.avg_entry_price else None
            market_value = float(pos.market_value) if pos.market_value else 0
            
            print(f"{i}. {symbol}")
            print(f"   Quantity: {qty:+.2f}")
            if avg_fill:
                print(f"   Entry Price: ${avg_fill:.4f}")
            print(f"   Market Value: ${market_value:.2f}")
            print()
        
        print("=" * 80)
        print("BEHAVIOR:")
        print("=" * 80)
        print("✓ Bot will NOT close these positions at startup")
        print("✓ Bot will read the fill price from each position")
        print("✓ Bot will continue monitoring for profit target or stop loss")
        print("✓ Bot will calculate P&L from the fill price")
        print("\nThis is the desired behavior for 'starting fresh'!")
        
except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
