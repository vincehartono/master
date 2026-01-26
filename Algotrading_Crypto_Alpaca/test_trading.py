"""
Test Trading Bot - Buy/Sell Signal Testing
Allows testing signals and order logic without running full bot
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Import from trade.py
from trade import (
    AlpacaCryptoBot, 
    TradingConfig, 
    set_credentials,
    dashboard_state,
    log_message
)

# ============================================================================
# TEST: Generate Signals for All Strategies
# ============================================================================

def test_all_strategies():
    """Test signal generation for all 15 strategies"""
    print("\n" + "="*80)
    print("TEST: Signal Generation for All Strategies")
    print("="*80)
    
    strategies = [
        # Long strategies
        "SMA", "EMA", "MACD", "RSI", "Bollinger", 
        "Stochastic", "ATR", "Volume",
        # Short strategies
        "Short_SMA", "Short_RSI", "Short_Bollinger", 
        "Short_Momentum", "Short_Downtrend", "Short_EMA", "Short_Stochastic"
    ]
    
    # Set credentials
    set_credentials(is_live=False)
    
    # Test each strategy
    for strategy in strategies:
        print(f"\n[TEST] {strategy}")
        try:
            config = TradingConfig(strategy=strategy, symbols=["BTC/USD"], paper_trading=True)
            bot = AlpacaCryptoBot(config)
            
            # Get historical data
            df = bot.get_historical_bars("BTC/USD", 50)
            if df.empty:
                print(f"  ✗ No data available")
                continue
            
            # Calculate indicators
            indicators = bot.calculate_indicators(df)
            
            # Generate signal
            signal = bot.generate_signal("BTC/USD", indicators)
            
            # Show result
            price = indicators.get('close', 0)
            rsi = indicators.get('rsi', 0)
            print(f"  ✓ Price: ${price:.4f} | RSI: {rsi:.1f} | Signal: {signal}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")

# ============================================================================
# TEST: Buy/Sell Order Simulation
# ============================================================================

def test_buy_sell_orders():
    """Test buy and sell order placement logic"""
    print("\n" + "="*80)
    print("TEST: Buy/Sell Order Placement (Simulated)")
    print("="*80)
    
    set_credentials(is_live=False)
    
    try:
        config = TradingConfig(strategy="Bollinger", symbols=["DOGE/USD"], paper_trading=True)
        bot = AlpacaCryptoBot(config)
        
        # Get account info
        account = bot.get_account()
        print(f"\n[ACCOUNT] Equity: ${account['equity']:.2f}")
        print(f"[ACCOUNT] Cash: ${account['cash']:.2f}")
        print(f"[ACCOUNT] Buying Power: ${account['buying_power']:.2f}")
        
        # Try multiple symbols to find which one has data
        symbols_to_try = ["BTC/USD", "DOGE/USD", "ETH/USD"]
        df = None
        working_symbol = None
        
        for symbol in symbols_to_try:
            print(f"\n[INFO] Trying to fetch data for {symbol}...")
            df = bot.get_historical_bars(symbol, 50)
            
            if df is not None and not df.empty:
                print(f"[OK] Got {len(df)} bars for {symbol}")
                working_symbol = symbol
                break
            else:
                print(f"[SKIP] No data for {symbol}")
        
        if df is None or df.empty:
            print(f"\n[ERROR] Could not fetch data from any symbol")
            return
        
        print(f"\n[INFO] Using symbol: {working_symbol}")
        print(f"[INFO] DataFrame shape: {df.shape}")
        
        print(f"\n[INFO] Calculating indicators...")
        indicators = bot.calculate_indicators(df)
        
        if not indicators or 'close' not in indicators:
            print(f"[ERROR] Could not calculate indicators")
            return
        
        # Test position sizing
        price = indicators['close']
        qty = bot.get_position_size(working_symbol, price)
        order_value = qty * price
        
        # Calculate trading fees
        from trade import calculate_trading_fees, TAKER_FEE
        fee = calculate_trading_fees(order_value, "market")
        
        print(f"\n[POSITION] Calculated position size: {qty:.4f} units")
        print(f"[POSITION] Current price: ${price:.4f}")
        print(f"[POSITION] Order value: ${order_value:.2f}")
        print(f"[POSITION] Trading fee (taker 0.25%): ${fee:.4f}")
        print(f"[POSITION] Total cost (including fee): ${order_value + fee:.2f}")
        
        # Check signal
        signal = bot.generate_signal(working_symbol, indicators)
        print(f"\n[SIGNAL] Generated signal: {signal}")
        
        if signal == "BUY":
            print(f"\n[ORDER DETAILS]")
            print(f"  Symbol: {working_symbol}")
            print(f"  Quantity: {qty:.4f} units")
            print(f"  Current Price: ${price:.4f}")
            print(f"  Order Value: ${order_value:.2f}")
            print(f"  Taker Fee (0.25%): ${fee:.4f}")
            print(f"  Total Cost: ${order_value + fee:.2f}")
            
            # Ask for confirmation to place actual trade
            print(f"\n[WARNING] About to place ACTUAL buy order on {working_symbol}")
            print(f"[INFO] Trading mode: {'LIVE' if not bot.config.paper_trading else 'PAPER'}")
            confirmation = input("Continue with buy order? (yes/no): ").strip().lower()
            
            if confirmation == "yes":
                order_id = bot.place_buy_order(working_symbol, qty, price)
                if order_id:
                    print(f"\n[SUCCESS] ✓ BUY order placed successfully")
                    print(f"  Order ID: {order_id}")
                    print(f"  Amount: ${qty * price:.2f}")
                else:
                    print(f"\n[ERROR] Failed to place BUY order")
            else:
                print(f"\n[CANCELLED] Buy order cancelled")
        
        elif signal == "SELL":
            print(f"\n[TEST] Would execute SELL order:")
            print(f"  Symbol: {working_symbol}")
            print(f"  Status: Need to have open position first")
        
        else:
            print(f"\n[TEST] No signal generated - holding")
        
        print(f"\n[RESULT] ✓ Test completed successfully")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        print("\n[TRACEBACK]")
        traceback.print_exc()

# ============================================================================
# TEST: Indicators Calculation
# ============================================================================

def test_indicators():
    """Test indicator calculations"""
    print("\n" + "="*80)
    print("TEST: Technical Indicators Calculation")
    print("="*80)
    
    set_credentials(is_live=False)
    
    try:
        config = TradingConfig(strategy="Bollinger", symbols=["BTC/USD"], paper_trading=True)
        bot = AlpacaCryptoBot(config)
        
        # Get data
        df = bot.get_historical_bars("BTC/USD", 50)
        indicators = bot.calculate_indicators(df)
        
        print(f"\n[INDICATORS] Calculated for BTC/USD (last 50 bars)")
        print(f"  Price: ${indicators.get('close', 0):.2f}")
        print(f"  SMA(10): {indicators.get('sma_fast', 0):.2f}")
        print(f"  SMA(30): {indicators.get('sma_slow', 0):.2f}")
        print(f"  EMA(12): {indicators.get('ema_fast', 0):.2f}")
        print(f"  EMA(26): {indicators.get('ema_slow', 0):.2f}")
        print(f"  RSI(14): {indicators.get('rsi', 0):.2f}")
        print(f"  MACD: {indicators.get('macd', 0):.6f}")
        print(f"  MACD Signal: {indicators.get('macd_signal', 0):.6f}")
        print(f"  BB Upper: {indicators.get('bb_upper', 0):.2f}")
        print(f"  BB Middle: {indicators.get('bb_middle', 0):.2f}")
        print(f"  BB Lower: {indicators.get('bb_lower', 0):.2f}")
        print(f"  Stochastic: {indicators.get('stochastic', 0):.2f}")
        print(f"  ATR: {indicators.get('atr', 0):.4f}")
        print(f"  Momentum: {indicators.get('momentum', 0):.4f}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

# ============================================================================
# TEST: Load Backtest Results
# ============================================================================

def test_load_backtest_results():
    """Test loading and parsing backtest results"""
    print("\n" + "="*80)
    print("TEST: Load Backtest Results from JSON")
    print("="*80)
    
    results_file = "backtest_results.json"
    
    if not os.path.exists(results_file):
        print(f"[ERROR] File not found: {results_file}")
        print("[INFO] Run backtest.py first to generate results")
        return
    
    try:
        with open(results_file, "r") as f:
            results = json.load(f)
        
        print(f"\n[LOADED] {len(results)} backtest results")
        
        # Show top 5
        print("\n[TOP 5] Best strategies by combo score:")
        for i, result in enumerate(results[:5], 1):
            combo_score = result.get('combo_score', 'N/A')
            win_rate = result.get('win_rate', 'N/A')
            profit_factor = result.get('profit_factor', 'N/A')
            total_return = result.get('total_return', 'N/A')
            
            print(f"\n  {i}. {result['strategy_name']} ({result['timeframe']}) on {result['symbol']}")
            
            try:
                if isinstance(combo_score, (int, float)):
                    print(f"     Combo Score: {combo_score:.1f}")
                else:
                    print(f"     Combo Score: {combo_score}")
            except:
                print(f"     Combo Score: {combo_score}")
            
            try:
                if isinstance(win_rate, (int, float)):
                    print(f"     Win Rate: {win_rate:.1f}%")
                else:
                    print(f"     Win Rate: {win_rate}")
            except:
                print(f"     Win Rate: {win_rate}")
            
            try:
                if isinstance(profit_factor, (int, float)):
                    print(f"     Profit Factor: {profit_factor:.2f}")
                else:
                    print(f"     Profit Factor: {profit_factor}")
            except:
                print(f"     Profit Factor: {profit_factor}")
            
            try:
                if isinstance(total_return, (int, float)):
                    print(f"     Return: {total_return:.1f}%")
                else:
                    print(f"     Return: {total_return}")
            except:
                print(f"     Return: {total_return}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

# ============================================================================
# TEST: Dashboard State
# ============================================================================

def test_dashboard_state():
    """Test dashboard state initialization"""
    print("\n" + "="*80)
    print("TEST: Dashboard State")
    print("="*80)
    
    print(f"\n[DASHBOARD STATE]")
    print(f"  Strategy: {dashboard_state.strategy}")
    print(f"  Symbol: {dashboard_state.symbol}")
    print(f"  Timeframe: {dashboard_state.timeframe}")
    print(f"  Price: ${dashboard_state.price:.4f}")
    print(f"  Equity: ${dashboard_state.equity:.2f}")
    print(f"  Cash: ${dashboard_state.cash:.2f}")
    print(f"  Buying Power: ${dashboard_state.buying_power:.2f}")
    print(f"  Recent Trades: {len(dashboard_state.recent_trades)} trades")
    print(f"  Last Signal: {dashboard_state.last_signal}")

# ============================================================================
# TEST: Strategy Specific Tests
# ============================================================================

def test_strategy_specific(strategy_name):
    """Test specific strategy in detail"""
    print("\n" + "="*80)
    print(f"TEST: {strategy_name} Strategy Details")
    print("="*80)
    
    set_credentials(is_live=False)
    
    try:
        config = TradingConfig(strategy=strategy_name, symbols=["BTC/USD"], paper_trading=True)
        bot = AlpacaCryptoBot(config)
        
        # Get data
        df = bot.get_historical_bars("BTC/USD", 100)
        indicators = bot.calculate_indicators(df)
        
        print(f"\n[{strategy_name.upper()}]")
        
        # Show strategy-specific indicators
        if "SMA" in strategy_name:
            print(f"  SMA(10): {indicators.get('sma_fast', 0):.2f}")
            print(f"  SMA(30): {indicators.get('sma_slow', 0):.2f}")
            print(f"  Price: {indicators.get('close', 0):.2f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "RSI" in strategy_name:
            print(f"  RSI(14): {indicators.get('rsi', 0):.2f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "Bollinger" in strategy_name:
            print(f"  BB Upper: {indicators.get('bb_upper', 0):.2f}")
            print(f"  BB Middle: {indicators.get('bb_middle', 0):.2f}")
            print(f"  BB Lower: {indicators.get('bb_lower', 0):.2f}")
            print(f"  Price: {indicators.get('close', 0):.2f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "EMA" in strategy_name:
            print(f"  EMA(12): {indicators.get('ema_fast', 0):.2f}")
            print(f"  EMA(26): {indicators.get('ema_slow', 0):.2f}")
            print(f"  Price: {indicators.get('close', 0):.2f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "MACD" in strategy_name:
            print(f"  MACD: {indicators.get('macd', 0):.6f}")
            print(f"  Signal: {indicators.get('macd_signal', 0):.6f}")
            print(f"  Histogram: {(indicators.get('macd', 0) - indicators.get('macd_signal', 0)):.6f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "Stochastic" in strategy_name:
            print(f"  Stochastic: {indicators.get('stochastic', 0):.2f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "ATR" in strategy_name:
            print(f"  ATR(14): {indicators.get('atr', 0):.4f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "Volume" in strategy_name:
            print(f"  Volume SMA: {indicators.get('volume_sma', 0):.0f}")
            print(f"  Current Volume: {df['volume'].iloc[-1]:.0f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        elif "Momentum" in strategy_name:
            print(f"  Momentum: {indicators.get('momentum', 0):.4f}")
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
        else:
            print(f"  Signal: {bot.generate_signal('BTC/USD', indicators)}")
        
    except Exception as e:
        print(f"[ERROR] {e}")

# ============================================================================
# TEST: Direct Buy Trade (No Signal Required)
# ============================================================================

def test_direct_buy_trade():
    """Execute a direct buy trade without needing a signal"""
    print("\n" + "="*80)
    print("TEST: Execute Direct Buy Trade")
    print("="*80)
    
    set_credentials(is_live=False)
    
    try:
        config = TradingConfig(strategy="Bollinger", symbols=["BTC/USD"], paper_trading=True)
        bot = AlpacaCryptoBot(config)
        
        # Get account info
        account = bot.get_account()
        print(f"\n[ACCOUNT] Equity: ${account['equity']:.2f}")
        print(f"[ACCOUNT] Cash: ${account['cash']:.2f}")
        print(f"[ACCOUNT] Buying Power: ${account['buying_power']:.2f}")
        
        # Get latest data
        print(f"\n[INFO] Fetching latest market data for BTC/USD...")
        df = bot.get_historical_bars("BTC/USD", 50)
        
        if df is None or df.empty:
            print(f"[ERROR] Could not fetch market data")
            return
        
        indicators = bot.calculate_indicators(df)
        
        if not indicators or 'close' not in indicators:
            print(f"[ERROR] Could not calculate indicators")
            return
        
        # Get position size
        price = indicators['close']
        qty = bot.get_position_size("BTC/USD", price)
        order_value = qty * price
        
        # Calculate trading fees
        from trade import calculate_trading_fees, TAKER_FEE
        fee = calculate_trading_fees(order_value, "market")
        
        print(f"\n[MARKET DATA]")
        print(f"  Symbol: BTC/USD")
        print(f"  Current Price: ${price:.4f}")
        print(f"  Position Size: {qty:.4f} units")
        print(f"  Order Value: ${order_value:.2f}")
        print(f"  Taker Fee (0.25%): ${fee:.4f}")
        print(f"  Total Cost (with fee): ${order_value + fee:.2f}")
        print(f"  Trading Mode: {'LIVE' if not bot.config.paper_trading else 'PAPER'}")
        
        # Ask for confirmation
        print(f"\n{'⚠️  WARNING' if not bot.config.paper_trading else '[PAPER TRADING]'} About to place BUY order")
        print(f"[INFO] This will:")
        print(f"  1. Buy {qty:.4f} BTC/USD @ ${price:.4f}")
        print(f"  2. Pay order value: ${order_value:.2f}")
        print(f"  3. Pay taker fee: ${fee:.4f}")
        print(f"  4. Total cost: ${order_value + fee:.2f}")
        print(f"  5. Open a position on your account")
        
        confirmation = input("\nConfirm buy order? (yes/no): ").strip().lower()
        
        if confirmation == "yes":
            print(f"\n[EXECUTING] Placing buy order...")
            order_id = bot.place_buy_order("BTC/USD", qty, price)
            
            if order_id:
                print(f"\n[SUCCESS] ✓ BUY order placed successfully!")
                print(f"  Order ID: {order_id}")
                print(f"  Symbol: BTC/USD")
                print(f"  Quantity: {qty:.4f} units")
                print(f"  Price: ${price:.4f}")
                print(f"  Amount: ${qty * price:.2f}")
                
                # Show updated account
                print(f"\n[INFO] Getting updated account info...")
                import time
                time.sleep(1)  # Wait for order to process
                
                updated_account = bot.get_account()
                print(f"  New Equity: ${updated_account['equity']:.2f}")
                print(f"  New Cash: ${updated_account['cash']:.2f}")
            else:
                print(f"\n[ERROR] Failed to place buy order")
                print(f"[INFO] Check logs for details")
        else:
            print(f"\n[CANCELLED] Buy order cancelled by user")
        
    except Exception as e:
        import traceback
        print(f"[ERROR] {e}")
        traceback.print_exc()

# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Main test menu"""
    while True:
        print("\n" + "="*80)
        print("TRADING BOT TEST SUITE")
        print("="*80)
        print("\n1. Test All Strategies (signals only)")
        print("2. Test Buy/Sell Orders (with confirmation)")
        print("3. Test Indicators Calculation")
        print("4. Load Backtest Results")
        print("5. Test Dashboard State")
        print("6. Test Specific Strategy (detailed)")
        print("7. Execute Direct Buy Trade")
        print("8. Exit")
        
        choice = input("\nSelect test (1-8): ").strip()
        
        if choice == "1":
            test_all_strategies()
        elif choice == "2":
            test_buy_sell_orders()
        elif choice == "3":
            test_indicators()
        elif choice == "4":
            test_load_backtest_results()
        elif choice == "5":
            test_dashboard_state()
        elif choice == "6":
            print("\nAvailable strategies:")
            strategies = [
                "SMA", "EMA", "MACD", "RSI", "Bollinger", 
                "Stochastic", "ATR", "Volume",
                "Short_SMA", "Short_RSI", "Short_Bollinger", 
                "Short_Momentum", "Short_Downtrend", "Short_EMA", "Short_Stochastic"
            ]
            for i, s in enumerate(strategies, 1):
                print(f"  {i:2d}. {s}")
            
            strategy_choice = input("\nSelect strategy (1-15): ").strip()
            try:
                idx = int(strategy_choice) - 1
                if 0 <= idx < len(strategies):
                    test_strategy_specific(strategies[idx])
                else:
                    print("[ERROR] Invalid choice")
            except:
                print("[ERROR] Invalid input")
        
        elif choice == "7":
            test_direct_buy_trade()
        
        elif choice == "8":
            print("\n[EXIT] Goodbye!")
            break
        else:
            print("[ERROR] Invalid choice")

if __name__ == "__main__":
    main()
