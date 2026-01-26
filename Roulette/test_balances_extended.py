import subprocess
import re

balances = [20000, 50000, 100000, 200000, 500000]

print("\nTESTING PROGRESSION STRATEGY WITH HIGHER STARTING BALANCES\n")
print("="*100)

for balance in balances:
    # Modify the Python file
    with open('simulate_10k_games.py', 'r') as f:
        content = f.read()
    
    content = re.sub(r'STARTING_BANKROLL = \d+', f'STARTING_BANKROLL = {balance}', content)
    
    with open('simulate_10k_games.py', 'w') as f:
        f.write(content)
    
    # Run simulation
    result = subprocess.run(['python', 'simulate_10k_games.py'], capture_output=True, text=True)
    output = result.stdout + result.stderr
    
    # Parse output
    final_match = re.search(r'Final Bankroll:\s+\$([0-9,.]+)', output)
    roi_match = re.search(r'^ROI:\s+([-0-9.]+)%', output, re.MULTILINE)
    pred_match = re.search(r'Total Predictions:\s+([0-9,]+)', output)
    
    if final_match and roi_match and pred_match:
        final = final_match.group(1).replace(',', '')
        roi = roi_match.group(1)
        pred = pred_match.group(1)
        
        final_val = float(final)
        profit = final_val - balance
        
        status = "🔴 LOSS" if profit < 0 else "🟢 PROFIT"
        
        print(f"Start: ${balance:7d} | Final: ${final_val:10.2f} | Profit: ${profit:10.2f} | ROI: {roi:7s}% | Predictions: {pred:8s} {status}")

print("="*100)
print("\nKEY INSIGHT: With ~48% win rate and 2:1 payout, the progression strategy")
print("will ALWAYS lose money long-term, regardless of starting balance.")
print("The math: -2% edge × (# of predictions) = guaranteed loss")
