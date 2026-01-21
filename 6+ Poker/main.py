import itertools
import random
from collections import Counter
import tkinter as tk
from tkinter import ttk, messagebox

# 6+ Poker (Short-Deck): 36-card deck (6 through A)
RANKS = "6789TJQKA"
SUITS = "CDHS"

# 6+ Poker hand ranking order (flush > full house, trips > straight)
HAND_RANKS = [
    "High Card",
    "One Pair",
    "Two Pair",
    "Straight",
    "Trips",
    "Full House",
    "Flush",
    "Quads",
    "Straight Flush",
]

def rank(card):
    return RANKS.index(card[0])

def suit(card):
    return card[1]

def is_straight(ranks):
    ranks = sorted(set(ranks))
    # In 6+ poker, wheel straight is A-6-7-8-9
    # With RANKS = "6789TJQKA", indices are: 6=0,7=1,8=2,9=3,A=8
    if ranks == [0, 1, 2, 3, 8]:
        return True
    return max(ranks) - min(ranks) == 4 and len(ranks) == 5

def evaluate_hand(cards):
    ranks = [rank(c) for c in cards]
    suits = [suit(c) for c in cards]
    count = Counter(ranks)
    values = sorted(count.values(), reverse=True)
    flush = len(set(suits)) == 1
    straight = is_straight(ranks)

    if straight and flush:
        return "Straight"
    if 4 in values:
        return "Quads"
    if values == [3, 2]:
        return "Full House"
    if flush:
        return "Flush"
    if straight:
        return "Straight"
    if 3 in values:
        return "Trips"
    if values == [2, 2, 1]:
        return "Two Pair"
    if 2 in values:
        return "One Pair"
    return "High Card"

def best_hand(cards):
    return max(
        (evaluate_hand(combo) for combo in itertools.combinations(cards, 5)),
        key=lambda h: HAND_RANKS.index(h)
    )

def calculate_probabilities(player_cards, board_cards):
    deck = [r + s for r in RANKS for s in SUITS]
    used = set(player_cards + board_cards)
    deck = [c for c in deck if c not in used]

    outcomes = Counter()
    remaining = 5 - len(board_cards)

    for runout in itertools.combinations(deck, remaining):
        full_board = board_cards + list(runout)
        final_cards = player_cards + full_board
        outcome = best_hand(final_cards)
        outcomes[outcome] += 1

    total = sum(outcomes.values())
    probs = {k: round(v / total * 100, 2) for k, v in outcomes.items()}
    return probs


def _hand_value_5(cards):
    """Return a comparable hand value for 5 cards: (category_index, kicker ranks...)."""
    rs = sorted([rank(c) for c in cards], reverse=True)
    suits = [suit(c) for c in cards]
    count = Counter(rs)
    # sort ranks by (count, rank) to build kickers
    by_count = sorted(count.items(), key=lambda x: (x[1], x[0]), reverse=True)
    counts = sorted(count.values(), reverse=True)
    is_flush = len(set(suits)) == 1
    is_str = is_straight(rs)

    # handle wheel straight A-6-7-8-9 in 6+ poker
    straight_high = max(rs)
    if is_str and set(rs) == {8, 0, 1, 2, 3}:
        straight_high = 3

    if is_str and is_flush:
        cat = HAND_RANKS.index("Straight Flush")
        return (cat, straight_high)
    if counts == [4, 1]:
        cat = HAND_RANKS.index("Quads")
        four = by_count[0][0]
        kicker = by_count[1][0]
        return (cat, four, kicker)
    if counts == [3, 2]:
        cat = HAND_RANKS.index("Full House")
        trip = by_count[0][0]
        pair = by_count[1][0]
        return (cat, trip, pair)
    if is_flush:
        cat = HAND_RANKS.index("Flush")
        kickers = [r for r, _ in by_count for _ in range(count[r])]
        return (cat, *kickers)
    if is_str:
        cat = HAND_RANKS.index("Straight")
        return (cat, straight_high)
    if counts == [3, 1, 1]:
        cat = HAND_RANKS.index("Trips")
        trip = by_count[0][0]
        kickers = [r for r, c in by_count[1:] for _ in range(c)]
        return (cat, trip, *kickers)
    if counts == [2, 2, 1]:
        cat = HAND_RANKS.index("Two Pair")
        pair1, pair2 = by_count[0][0], by_count[1][0]
        kicker = by_count[2][0]
        return (cat, pair1, pair2, kicker)
    if counts == [2, 1, 1, 1]:
        cat = HAND_RANKS.index("One Pair")
        pair = by_count[0][0]
        kickers = [r for r, c in by_count[1:] for _ in range(c)]
        return (cat, pair, *kickers)
    cat = HAND_RANKS.index("High Card")
    kickers = [r for r, c in by_count for _ in range(c)]
    return (cat, *kickers)


def best_hand_value(cards):
    """Return best 5-card hand value from up to 7 cards."""
    best = None
    for combo in itertools.combinations(cards, 5):
        hv = _hand_value_5(combo)
        if best is None or hv > best:
            best = hv
    return best


def calculate_headsup_equity(player_cards, board_cards, samples=5000):
    """
    Monte Carlo heads-up equity vs a random 2-card opponent.
    Returns dict with win/tie/lose percentages.
    """
    deck = [r + s for r in RANKS for s in SUITS]
    used = set(player_cards + board_cards)
    deck = [c for c in deck if c not in used]

    remaining_board = 5 - len(board_cards)
    if remaining_board < 0:
        raise ValueError("Board has more than 5 cards")
    if len(deck) < 2 + remaining_board:
        raise ValueError("Not enough cards remaining in deck")

    wins = ties = losses = 0

    for _ in range(samples):
        # sample dealer + remaining board without replacement
        sample = random.sample(deck, 2 + remaining_board)
        dealer_cards = sample[:2]
        runout = sample[2:]
        full_board = board_cards + list(runout)

        hero_full = player_cards + full_board
        dealer_full = dealer_cards + full_board

        hero_val = best_hand_value(hero_full)
        dealer_val = best_hand_value(dealer_full)

        if hero_val > dealer_val:
            wins += 1
        elif hero_val < dealer_val:
            losses += 1
        else:
            ties += 1

    total = wins + losses + ties
    if total == 0:
        return {"win": 0.0, "tie": 0.0, "lose": 0.0}
    return {
        "win": round(wins / total * 100, 2),
        "tie": round(ties / total * 100, 2),
        "lose": round(losses / total * 100, 2),
    }


def calculate_winning_hand_distribution(player_cards, board_cards, samples=5000):
    """
    Monte Carlo distribution of winning hand types (you and dealer) vs a random 2-card dealer.
    Returns two dicts:
      hero_dist   - percentages of your WINS by hand type
      dealer_dist - percentages of dealer WINS by hand type
    """
    deck = [r + s for r in RANKS for s in SUITS]
    used = set(player_cards + board_cards)
    deck = [c for c in deck if c not in used]

    remaining_board = 5 - len(board_cards)
    if remaining_board < 0:
        raise ValueError("Board has more than 5 cards")
    if len(deck) < 2 + remaining_board:
        raise ValueError("Not enough cards remaining in deck")

    hero_win_counts: Counter[str] = Counter()
    dealer_win_counts: Counter[str] = Counter()

    for _ in range(samples):
        sample = random.sample(deck, 2 + remaining_board)
        dealer_cards = sample[:2]
        runout = sample[2:]
        full_board = board_cards + list(runout)

        hero_full = player_cards + full_board
        dealer_full = dealer_cards + full_board

        hero_val = best_hand_value(hero_full)
        dealer_val = best_hand_value(dealer_full)

        if hero_val > dealer_val:
            cat_idx = hero_val[0]
            cat_name = HAND_RANKS[cat_idx]
            hero_win_counts[cat_name] += 1
        elif dealer_val > hero_val:
            cat_idx = dealer_val[0]
            cat_name = HAND_RANKS[cat_idx]
            dealer_win_counts[cat_name] += 1

    hero_total_wins = sum(hero_win_counts.values())
    dealer_total_wins = sum(dealer_win_counts.values())

    hero_dist = {}
    dealer_dist = {}
    combined_dist = {}
    combined_total_wins = hero_total_wins + dealer_total_wins

    for name in HAND_RANKS:
        h = hero_win_counts.get(name, 0)
        d = dealer_win_counts.get(name, 0)

        if hero_total_wins > 0:
            hero_dist[name] = round(h / hero_total_wins * 100, 2)
        else:
            hero_dist[name] = 0.0

        if dealer_total_wins > 0:
            dealer_dist[name] = round(d / dealer_total_wins * 100, 2)
        else:
            dealer_dist[name] = 0.0

        if combined_total_wins > 0:
            combined_dist[name] = round((h + d) / combined_total_wins * 100, 2)
        else:
            combined_dist[name] = 0.0

    return hero_dist, dealer_dist, combined_dist


def run_cli():
    """
    Simple CLI to evaluate 6+ poker hand probabilities.
    Asks for 2 hole cards and optional board cards, then
    prints approximate probabilities by the river.
    """
    print("6+ Poker Hand Evaluator")
    print("Enter cards as e.g. Ah, Kd, 9c etc. (A=Ace, K=King, Q=Queen, J=Jack, T=Ten)")
    print("Suits: C=Clubs, D=Diamonds, H=Hearts, S=Spades")
    print()

    # Read player hole cards
    player_str = input("Enter your 2 hole cards (space-separated, e.g. Ah Kh): ").strip()
    player_cards = player_str.upper().split()
    if len(player_cards) != 2:
        print("Please enter exactly 2 cards.")
        return

    # Read board cards (0–5)
    board_str = input("Enter board cards if any (0–5, space-separated, e.g. 7h 8h 9h): ").strip()
    board_cards = board_str.upper().split() if board_str else []

    if len(board_cards) > 5:
        print("Board can have at most 5 cards.")
        return

    try:
        probs = calculate_probabilities(player_cards, board_cards)
    except Exception as e:
        print(f"Error calculating probabilities: {e}")
        return

    print("\nProbabilities by river:")
    for hand_name in HAND_RANKS:
        if hand_name in probs:
            print(f"{hand_name:14}: {probs[hand_name]:5.2f}%")


def run_gui():
    selected_player: list[str] = []
    selected_board: list[str] = []

    def format_card(card: str) -> str:
        r = card[0]
        s = card[1]
        return f"{rank_labels.get(r, r)}{suit_symbols.get(s, '')}"

    def update_labels():
        lbl_player_var.set("Hole: " + " ".join(format_card(c) for c in selected_player))
        lbl_board_var.set("Board: " + " ".join(format_card(c) for c in selected_board))

    def on_card_click(card: str):
        # Toggle off if already selected
        if card in selected_player:
            selected_player.remove(card)
        elif card in selected_board:
            selected_board.remove(card)
        else:
            # Add to player (max 2), then board (max 5)
            if len(selected_player) < 2:
                selected_player.append(card)
            elif len(selected_board) < 5:
                selected_board.append(card)
            else:
                messagebox.showinfo("Selection full", "You already have 2 hole cards and 5 board cards selected.")
                return
        update_labels()

    def on_clear():
        selected_player.clear()
        selected_board.clear()
        update_labels()
        text_result.config(state="normal")
        text_result.delete("1.0", tk.END)
        text_result.config(state="disabled")

    def on_calculate():
        if len(selected_player) != 2:
            messagebox.showerror("Input error", "Please select exactly 2 hole cards by clicking on the card buttons.")
            return

        try:
            equity = calculate_headsup_equity(selected_player, selected_board, samples=3000)
            hero_dist, dealer_dist, combined_dist = calculate_winning_hand_distribution(selected_player, selected_board, samples=3000)
        except Exception as e:
            messagebox.showerror("Error", f"Error calculating probabilities/equity:\n{e}")
            return

        text_result.config(state="normal")
        text_result.delete("1.0", tk.END)
        text_result.insert(tk.END, "Player:\n")
        hero_sorted = sorted(hero_dist.items(), key=lambda x: x[1], reverse=True)
        for hand_name, pct in hero_sorted[:2]:
            text_result.insert(tk.END, f"{hand_name:14}: {pct:5.2f}%\n")

        text_result.insert(tk.END, "\nDealer:\n")
        dealer_sorted = sorted(dealer_dist.items(), key=lambda x: x[1], reverse=True)
        for hand_name, pct in dealer_sorted[:2]:
            text_result.insert(tk.END, f"{hand_name:14}: {pct:5.2f}%\n")

        text_result.insert(tk.END, "\nCombined (player + dealer):\n")
        combined_sorted = sorted(combined_dist.items(), key=lambda x: x[1], reverse=True)
        for hand_name, pct in combined_sorted[:2]:
            text_result.insert(tk.END, f"{hand_name:14}: {pct:5.2f}%\n")

        text_result.insert(tk.END, "\nHeads-up win/tie/lose vs random 2-card dealer by river:\n")
        text_result.insert(tk.END, f"Win : {equity['win']:5.2f}%\n")
        text_result.insert(tk.END, f"Tie : {equity['tie']:5.2f}%\n")
        text_result.insert(tk.END, f"Lose: {equity['lose']:5.2f}%\n")
        text_result.config(state="disabled")

    root = tk.Tk()
    root.title("6+ Poker Hand Evaluator")
    root.geometry("800x700")

    default_font = ("Segoe UI", 14)

    main_frame = ttk.Frame(root, padding=20)
    main_frame.grid(row=0, column=0, sticky="nsew")

    # Selection labels
    lbl_player_var = tk.StringVar(value="Hole: ")
    lbl_board_var = tk.StringVar(value="Board: ")
    ttk.Label(main_frame, textvariable=lbl_player_var, font=default_font).grid(row=0, column=0, columnspan=4, sticky="w")
    ttk.Label(main_frame, textvariable=lbl_board_var, font=default_font).grid(row=1, column=0, columnspan=4, sticky="w")

    # Card buttons grid
    cards_frame = ttk.LabelFrame(main_frame, text="Click cards to select (2 hole, up to 5 board)", padding=5)
    cards_frame.grid(row=2, column=0, columnspan=4, pady=(15, 15), sticky="nsew")

    row = 0
    suit_colors = {
        "C": "#000000",  # clubs - black
        "S": "#000000",  # spades - black
        "D": "#B22222",  # diamonds - red
        "H": "#B22222",  # hearts - red
    }
    suit_symbols = {
        "C": "♣",
        "D": "♦",
        "H": "♥",
        "S": "♠",
    }
    rank_labels = {r: r for r in RANKS}
    rank_labels["T"] = "10"

    # Display order: group black suits together, then red suits
    display_suits = ["C", "S", "D", "H"]

    for r in RANKS:
        col = 0
        for s in display_suits:
            card = (r + s).upper()
            color = suit_colors.get(s, "#CCCCCC")
            # Use white text on both black and red backgrounds for clarity
            fg_color = "white"
            btn = tk.Button(
                cards_frame,
                text=f"{rank_labels.get(r, r)}{suit_symbols.get(s, '')}",
                width=6,
                height=2,
                bg=color,
                fg=fg_color,
                activebackground=color,
                activeforeground=fg_color,
                command=lambda c=card: on_card_click(c),
            )
            btn.grid(row=row, column=col, padx=1, pady=1)
            col += 1
        row += 1

    # Control buttons
    btn_calc = ttk.Button(main_frame, text="Calculate", command=on_calculate)
    btn_calc.grid(row=3, column=0, pady=(10, 10), sticky="w")

    btn_clear = ttk.Button(main_frame, text="Clear", command=on_clear)
    btn_clear.grid(row=3, column=1, pady=(10, 10), sticky="w")

    text_result = tk.Text(main_frame, width=60, height=18, font=default_font, state="disabled")
    text_result.grid(row=4, column=0, columnspan=4, sticky="nsew")

    main_frame.columnconfigure(0, weight=1)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(4, weight=1)

    root.mainloop()


def main():
    # Default to GUI; comment this line to use CLI instead
    run_gui()
    # run_cli()


if __name__ == "__main__":
    main()
