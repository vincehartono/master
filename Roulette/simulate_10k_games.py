import random
from datetime import datetime
from collections import defaultdict

# Color mapping
RED_NUMBERS = {1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36}
BLACK_NUMBERS = {2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35}

def get_priority_score(category, value, draws):
    """Calculate priority score - lower means higher priority"""
    if not value or value == "Either":
        return 999
    
    if category == "color":
        red_count = sum(1 for d in draws if d in RED_NUMBERS)
        black_count = sum(1 for d in draws if d in BLACK_NUMBERS)
        return red_count if value == "RED" else black_count
    elif category == "parity":
        odd_count = sum(1 for d in draws if d % 2 == 1)
        even_count = sum(1 for d in draws if d % 2 == 0)
        return odd_count if value == "ODD" else even_count
    elif category == "size":
        small_count = sum(1 for d in draws if 1 <= d <= 18)
        big_count = sum(1 for d in draws if 19 <= d <= 36)
        return small_count if value == "SMALL" else big_count
    return 0

def analyze_color(draws):
    if not draws:
        return (0, 0), "Either"
    
    red_count = sum(1 for d in draws if d in RED_NUMBERS)
    black_count = sum(1 for d in draws if d in BLACK_NUMBERS)
    
    drawn_red = any(d in RED_NUMBERS for d in draws)
    drawn_black = any(d in BLACK_NUMBERS for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_red = sum(1 for d in recent_draws if d in RED_NUMBERS)
    recent_black = sum(1 for d in recent_draws if d in BLACK_NUMBERS)
    
    if not drawn_red:
        rec = "RED"
    elif not drawn_black:
        rec = "BLACK"
    elif red_count < black_count:
        rec = "RED"
    elif black_count < red_count:
        rec = "BLACK"
    else:
        if recent_red > recent_black:
            rec = "RED"
        elif recent_black > recent_red:
            rec = "BLACK"
        else:
            rec = "Either"
    
    return (red_count, black_count), rec

def analyze_parity(draws):
    if not draws:
        return (0, 0), "Either"
    
    odd_count = sum(1 for d in draws if d % 2 == 1)
    even_count = sum(1 for d in draws if d % 2 == 0)
    
    drawn_odd = any(d % 2 == 1 for d in draws)
    drawn_even = any(d % 2 == 0 for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_odd = sum(1 for d in recent_draws if d % 2 == 1)
    recent_even = sum(1 for d in recent_draws if d % 2 == 0)
    
    if not drawn_odd:
        rec = "ODD"
    elif not drawn_even:
        rec = "EVEN"
    elif odd_count < even_count:
        rec = "ODD"
    elif even_count < odd_count:
        rec = "EVEN"
    else:
        if recent_odd > recent_even:
            rec = "ODD"
        elif recent_even > recent_odd:
            rec = "EVEN"
        else:
            rec = "Either"
    
    return (odd_count, even_count), rec

def analyze_size(draws):
    if not draws:
        return (0, 0), "Either"
    
    small_count = sum(1 for d in draws if 1 <= d <= 18)
    big_count = sum(1 for d in draws if 19 <= d <= 36)
    
    drawn_small = any(1 <= d <= 18 for d in draws)
    drawn_big = any(19 <= d <= 36 for d in draws)
    
    recent_draws = draws[-4:] if len(draws) >= 4 else draws
    recent_small = sum(1 for d in recent_draws if 1 <= d <= 18)
    recent_big = sum(1 for d in recent_draws if 19 <= d <= 36)
    
    if not drawn_small:
        rec = "SMALL"
    elif not drawn_big:
        rec = "BIG"
    elif small_count < big_count:
        rec = "SMALL"
    elif big_count < small_count:
        rec = "BIG"
    else:
        if recent_small > recent_big:
            rec = "SMALL"
        elif recent_big > recent_small:
            rec = "BIG"
        else:
            rec = "Either"
    
    return (small_count, big_count), rec

def get_top_recommendation(draws):
    """Get the highest priority recommendation"""
    if len(draws) < 1:
        return None
    
    recommendations = []
    
    color_counts, color_rec = analyze_color(draws)
    if color_rec and color_rec != "Either":
        recommendations.append(("Color", color_rec, get_priority_score("color", color_rec, draws)))
    
    parity_counts, parity_rec = analyze_parity(draws)
    if parity_rec and parity_rec != "Either":
        recommendations.append(("Parity", parity_rec, get_priority_score("parity", parity_rec, draws)))
    
    size_counts, size_rec = analyze_size(draws)
    if size_rec and size_rec != "Either":
        recommendations.append(("Size", size_rec, get_priority_score("size", size_rec, draws)))
    
    if not recommendations:
        return None
    
    recommendations.sort(key=lambda x: x[2])
    return recommendations[0][1]  # Return top recommendation

def check_prediction(prediction, actual_number):
    """Check if prediction matches the actual number"""
    # 0 is neither red/black, odd/even, nor big/small in roulette
    if actual_number == 0:
        return False
    
    if prediction == "RED":
        return actual_number in RED_NUMBERS
    elif prediction == "BLACK":
        return actual_number in BLACK_NUMBERS
    elif prediction == "ODD":
        return actual_number % 2 == 1
    elif prediction == "EVEN":
        return actual_number % 2 == 0
    elif prediction == "SMALL":
        return 1 <= actual_number <= 18
    elif prediction == "BIG":
        return 19 <= actual_number <= 36
    return False

def run_simulation(num_games=10000, sequence_length=6, starting_bankroll=10000, base_bet=100, strategy="martingale", debug=False):
    """
    Simulate roulette games with different betting strategies
    
    Args:
        num_games: Number of games to simulate
        sequence_length: How many prior draws to keep in history
        starting_bankroll: Starting amount of money
        base_bet: Base bet amount (flat or martingale starting amount)
        strategy: "flat" or "martingale"
        debug: Print each prediction and result
    """
    print(f"Starting simulation of {num_games:,} games...")
    print(f"Strategy: {strategy.upper()}")
    print(f"Sequence length: {sequence_length}")
    print(f"Starting Bankroll: ${starting_bankroll:,.2f}")
    print(f"Base Bet: ${base_bet:,.2f}")
    print(f"Roulette Numbers: 0-36 (0 is neither red nor black)")
    print("-" * 80)
    
    all_numbers = list(range(0, 37))  # 0-36 (0 is neither red nor black)
    
    bankroll = starting_bankroll
    bankroll_history = [starting_bankroll]
    
    stats = {
        "strategy": strategy,
        "total_predictions": 0,
        "correct_predictions": 0,
        "incorrect_predictions": 0,
        "predictions_by_type": defaultdict(lambda: {"correct": 0, "total": 0, "profit": 0}),
        "win_rate": 0,
        "starting_bankroll": starting_bankroll,
        "final_bankroll": 0,
        "total_profit": 0,
        "roi": 0,
        "max_bankroll": starting_bankroll,
        "min_bankroll": starting_bankroll,
        "max_drawdown": 0,
        "max_drawdown_pct": 0,
        "losing_streak": 0,
        "max_losing_streak": 0,
        "base_bet": base_bet,
    }
    
    losing_streak = 0
    current_bet = base_bet
    consecutive_losses = 0
    initial_prediction = None  # Persist across games for continuous martingale
    draws = []  # Keep draws across games for prediction consistency
    
    # Progression sequence: 2, 4, 6, 9, 13, 18, 24, 31, 39, ...
    # Each bet is: base_bet * progression_multiplier[consecutive_losses]
    progression_sequence = [2, 4, 6, 9, 13, 18, 24, 31, 39, 49, 60, 72, 85, 99, 114]
    
    for game_num in range(num_games):
        
        for round_num in range(sequence_length + 1):  # Build history then 1 prediction per game
            if len(draws) >= sequence_length:
                # Get prediction
                if strategy.lower() == "martingale":
                    # Stick with initial prediction, double on loss
                    if initial_prediction is None:
                        initial_prediction = get_top_recommendation(draws)
                    prediction = initial_prediction
                elif strategy.lower() == "adaptive":
                    # Always use best prediction, but double bet on loss
                    prediction = get_top_recommendation(draws)
                elif strategy.lower() == "switch_after_3":
                    # Keep prediction for up to 3 losses, then switch and reset bet
                    if initial_prediction is None or consecutive_losses >= 3:
                        initial_prediction = get_top_recommendation(draws)
                        current_bet = base_bet  # Reset bet when switching
                        consecutive_losses = 0  # Reset counter when switching
                    prediction = initial_prediction
                elif strategy.lower() == "progression":
                    # Stick with initial prediction, use progression sequence on loss
                    if initial_prediction is None:
                        initial_prediction = get_top_recommendation(draws)
                    prediction = initial_prediction
                else:
                    prediction = get_top_recommendation(draws)
                
                if prediction:
                    # Check if bankroll can cover the bet
                    if bankroll < current_bet:
                        if debug:
                            print(f"BANKRUPTCY: Game {game_num+1}, Round {round_num+1}: Bankroll ${bankroll:.2f} < Bet ${current_bet:.2f}")
                        break  # Stop this game, bankrupt
                    
                    # Store the bet amount BEFORE it changes
                    bet_placed = current_bet
                    actual = random.choice(all_numbers)
                    is_correct = check_prediction(prediction, actual)
                    
                    # Update stats
                    stats["total_predictions"] += 1
                    stats["predictions_by_type"][prediction]["total"] += 1
                    
                    if is_correct:
                        stats["correct_predictions"] += 1
                        stats["predictions_by_type"][prediction]["correct"] += 1
                        # Win
                        bankroll += current_bet
                        stats["predictions_by_type"][prediction]["profit"] += current_bet
                        consecutive_losses = 0
                        current_bet = base_bet  # Reset to base bet
                        initial_prediction = None  # Reset prediction for next sequence
                        losing_streak = 0
                    else:
                        stats["incorrect_predictions"] += 1
                        # Loss
                        bankroll -= current_bet
                        stats["predictions_by_type"][prediction]["profit"] -= current_bet
                        consecutive_losses += 1
                        losing_streak += 1
                        stats["max_losing_streak"] = max(stats["max_losing_streak"], losing_streak)
                        
                        # Update bet based on strategy
                        if strategy.lower() in ["martingale", "adaptive"]:
                            current_bet = base_bet * (2 ** consecutive_losses)
                        elif strategy.lower() == "switch_after_3":
                            # Double bet up to 3 losses, then reset when switching
                            if consecutive_losses < 3:
                                current_bet = base_bet * (2 ** consecutive_losses)
                            # If consecutive_losses >= 3, next prediction will trigger switch and reset
                        elif strategy.lower() == "progression":
                            # Use progression sequence: 2, 4, 6, 9, 13, 18, 24, 31, 39, ...
                            if consecutive_losses < len(progression_sequence):
                                current_bet = base_bet * progression_sequence[consecutive_losses]
                            else:
                                # If we exceed sequence length, use last value or cap it
                                current_bet = base_bet * progression_sequence[-1]
                        
                        # Debug print consecutive losses > 10
                        if consecutive_losses > 10:
                            print(f"WARNING: Game {game_num+1}, Round {round_num+1}: {consecutive_losses} consecutive losses! Bet=${current_bet:.2f}, Bankroll=${bankroll:.2f}")
                    
                    # Print debug info if enabled (AFTER bankroll update, using the bet that was placed)
                    if debug:
                        result = "WIN" if is_correct else "LOSS"
                        print(f"Game {game_num+1}, Round {round_num+1}: Prediction={prediction}, Actual={actual}, Bet=${bet_placed:.2f}, Result={result}, Bankroll=${bankroll:.2f}")
                    
                    # Track bankroll
                    bankroll_history.append(bankroll)
                    stats["max_bankroll"] = max(stats["max_bankroll"], bankroll)
                    stats["min_bankroll"] = min(stats["min_bankroll"], bankroll)
                    
                    # Add result to draws
                    draws.append(actual)
                    if len(draws) > sequence_length:
                        draws.pop(0)
                else:
                    draws.append(random.choice(all_numbers))
                    if len(draws) > sequence_length:
                        draws.pop(0)
            else:
                draws.append(random.choice(all_numbers))
        
        if (game_num + 1) % 1000 == 0:
            print(f"Completed {game_num + 1:,} games...")
    
    # Calculate metrics
    if stats["total_predictions"] > 0:
        stats["win_rate"] = (stats["correct_predictions"] / stats["total_predictions"]) * 100
    
    stats["final_bankroll"] = bankroll
    stats["total_profit"] = bankroll - starting_bankroll
    stats["roi"] = (stats["total_profit"] / starting_bankroll * 100) if starting_bankroll > 0 else 0
    
    # Calculate max drawdown
    peak = starting_bankroll
    for value in bankroll_history:
        if value > peak:
            peak = value
        drawdown = peak - value
        if drawdown > stats["max_drawdown"]:
            stats["max_drawdown"] = drawdown
            stats["max_drawdown_pct"] = (drawdown / peak * 100) if peak > 0 else 0
    
    return stats

def print_results(stats):
    """Print simulation results"""
    print("\n" + "=" * 100)
    print("SIMULATION RESULTS - MARTINGALE STRATEGY")
    print("=" * 100)
    print_strategy_results(stats)

def print_strategy_results(stats):
    """Print results for a single strategy"""
    print(f"Total Predictions: {stats['total_predictions']:,}")
    print(f"Correct: {stats['correct_predictions']:,}")
    print(f"Incorrect: {stats['incorrect_predictions']:,}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print("-" * 100)
    print("BANKROLL ANALYSIS")
    print("-" * 100)
    print(f"Starting Bankroll: ${stats['starting_bankroll']:,.2f}")
    print(f"Final Bankroll:    ${stats['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${stats['total_profit']:,.2f}")
    print(f"ROI:               {stats['roi']:.2f}%")
    print(f"Max Bankroll:      ${stats['max_bankroll']:,.2f}")
    print(f"Min Bankroll:      ${stats['min_bankroll']:,.2f}")
    print(f"Max Drawdown:      ${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.2f}%)")
    print(f"Max Losing Streak: {stats['max_losing_streak']} consecutive losses")
    print("-" * 100)
    print("Results by Prediction Type:")
    print("-" * 100)
    
    for prediction_type in sorted(stats['predictions_by_type'].keys()):
        data = stats['predictions_by_type'][prediction_type]
        total = data['total']
        correct = data['correct']
        profit = data['profit']
        win_rate = (correct / total * 100) if total > 0 else 0
        print(f"{prediction_type:10} | Total: {total:6,} | Correct: {correct:6,} | Win Rate: {win_rate:6.2f}% | Profit: ${profit:10,.2f}")
    
    print("=" * 100)

if __name__ == "__main__":
    STARTING_BANKROLL = 500000
    BASE_BET = 2
    NUM_GAMES = 10000
    DEBUG = False  # Set to True to show each game
    STRATEGY = "progression"  # "martingale", "adaptive", "switch_after_3", or "progression"
    
    print("=" * 100)
    print(f"ROULETTE {STRATEGY.upper()} STRATEGY SIMULATION")
    print("=" * 100 + "\n")
    
    # Run strategy
    results = run_simulation(
        num_games=NUM_GAMES,
        sequence_length=6,
        starting_bankroll=STARTING_BANKROLL,
        base_bet=BASE_BET,
        strategy=STRATEGY,
        debug=DEBUG
    )
    
    # Print results
    print_results(results)
