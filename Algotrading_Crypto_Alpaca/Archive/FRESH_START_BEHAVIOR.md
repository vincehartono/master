# Bot Behavior: Starting Fresh with Existing Positions

## ✅ Changes Made

### 1. **No Force-Close on Startup**
- Bot **does NOT** close existing positions when it starts
- Any positions from previous runs are preserved

### 2. **Position Detection at Startup**
When the bot starts, it now logs:
```
[INFO] Found 1 existing position(s):
  • DOGEUSD: +720.42 @ $0.1251 (will monitor P&L)
```

### 3. **Fill Price Reading**
- Bot reads `avg_entry_price` from each existing position
- Example: Position filled at $0.1251 for 720 DOGE

### 4. **Continuous P&L Monitoring**
The bot then monitors each position for:
- **Profit Target**: If P&L >= $2.00 (default) → SELL
- **Stop Loss**: If P&L <= -$5.00 (default) → SELL

### 5. **How It Works**

**Example Scenario:**
```
1. Bot starts with existing DOGEUSD position
   Entry Price: $0.1251
   Quantity: 720.42 DOGE

2. Current market price: $0.1276
   P&L = (0.1276 - 0.1251) * 720.42 = $18.03

3. If P&L >= $2.00 (profit target) → Bot SELLS
   If P&L <= -$5.00 (stop loss) → Bot SELLS

4. If neither condition hit → Bot continues monitoring
   and generates new signals for other symbols
```

## 🔍 Key Code Locations

**Startup Position Logging** (Lines 928-936):
```python
existing_positions = self.trading_client.get_all_positions()
if existing_positions:
    for pos in existing_positions:
        qty = float(pos.qty)
        avg_fill = float(pos.avg_entry_price) if pos.avg_entry_price else None
        log_message(f"  • {pos.symbol}: {qty:+.2f} @ ${avg_fill:.4f}")
```

**P&L Check Loop** (Lines 1097-1118):
```python
position = self.trading_client.get_open_position(symbol)
if position and float(position.qty) != 0:
    entry_price = float(position.avg_entry_price) if position.avg_entry_price else None
    current_price = indicators['close']
    position_pnl = (current_price - entry_price) * abs(float(position.qty))
    
    if position_pnl >= profit_target:
        # SELL - Profit target hit
    elif position_pnl <= -stop_loss:
        # SELL - Stop loss hit
```

## 📊 Current Status

**Existing Position:**
- Symbol: DOGEUSD (720.42 shares)
- Entry Price: $0.1251
- Market Value: $90.21
- Status: Will be monitored by bot on next start

## ✅ Verification

Run this to check for existing positions:
```bash
python test_existing_position.py
```

This will show:
- Number of open positions
- Entry price for each
- Market value
- Confirmation that bot will monitor them
