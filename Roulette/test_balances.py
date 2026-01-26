import subprocess
import re

balances = [500, 1000, 2000, 5000, 10000, 20000]

print("\nTESTING PROGRESSION STRATEGY WITH DIFFERENT STARTING BALANCES\n")
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
        
        print(f"Start: ${balance:6d} | Final: ${final_val:9.2f} | Profit: ${profit:9.2f} | ROI: {roi:7s}% | Predictions: {pred}")

print("="*100)
