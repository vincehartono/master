import random
from collections import Counter
from typing import List, Dict, Tuple


RANKS = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "0"]  # 0 = ten/J/Q/K (value 0)


def rank_value(rank: str) -> int:
    """Return baccarat point value for a single rank."""
    if rank == "A":
        return 1
    if rank in {"2", "3", "4", "5", "6", "7", "8", "9"}:
        return int(rank)
    return 0  # 0 = 10/J/Q/K


def hand_total(cards: List[str]) -> int:
    """Baccarat hand total (mod 10)."""
    return sum(rank_value(c) for c in cards) % 10


def make_shoe(decks: int = 8) -> Dict[str, int]:
    """
    Create a shoe as counts of ranks.
    We collapse 10/J/Q/K into a single '0' rank since suits don't matter.
    """
    counts: Dict[str, int] = Counter()
    per_deck = {
        "A": 4,
        "2": 4,
        "3": 4,
        "4": 4,
        "5": 4,
        "6": 4,
        "7": 4,
        "8": 4,
        "9": 4,
        "0": 16,  # 10 + J + Q + K
    }
    for _ in range(decks):
        for r, n in per_deck.items():
            counts[r] += n
    return counts


def remove_known(cards: List[str], counts: Dict[str, int]) -> None:
    """Remove known cards from the shoe counts."""
    for c in cards:
        if c not in counts or counts[c] <= 0:
            raise ValueError(f"Card {c} not available in shoe for conditioning")
        counts[c] -= 1


def draw_card(counts: Dict[str, int]) -> str:
    """Draw a random rank from shoe counts (without replacement)."""
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError("Shoe is empty")
    r = random.randint(1, total)
    cum = 0
    for rank, n in counts.items():
        if n <= 0:
            continue
        cum += n
        if r <= cum:
            counts[rank] -= 1
            return rank
    # Should not get here
    raise RuntimeError("draw_card failed")


def deal_hand_with_known(
    counts: Dict[str, int],
    known_player: List[str],
    known_banker: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Build initial 2-card hands given known starting cards.

    Interpretation:
    - known_player and known_banker are cards that are already visible and
      guaranteed to be part of the player's / banker's initial 2-card hands.
    - We fill any missing slots by drawing uniformly from the remaining shoe.
    """
    player = list(known_player)
    banker = list(known_banker)

    # Remove known from shoe
    remove_known(player + banker, counts)

    # Fill remaining starting slots: order P, B, P, B doesn't matter for totals.
    while len(player) < 2:
        player.append(draw_card(counts))
    while len(banker) < 2:
        banker.append(draw_card(counts))

    return player, banker


def play_baccarat_round(
    counts: Dict[str, int],
    known_player: List[str],
    known_banker: List[str],
) -> str:
    """
    Simulate a full baccarat round with standard drawing rules given that
    some starting cards are already known.

    Returns 'P' (player win), 'B' (banker win), or 'T' (tie).
    """
    player, banker = deal_hand_with_known(counts, known_player, known_banker)

    p_total = hand_total(player)
    b_total = hand_total(banker)

    # Natural 8 or 9: both stand.
    if p_total >= 8 or b_total >= 8:
        pass
    else:
        # Player third-card rule
        player_third: str | None = None
        if p_total <= 5:
            player_third = draw_card(counts)
            player.append(player_third)
            p_total = hand_total(player)

        # Banker rules depend on whether player drew a third card
        if player_third is None:
            # Player stood on 6 or 7
            if b_total <= 5:
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
        else:
            pt_val = rank_value(player_third)
            if b_total <= 2:
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
            elif b_total == 3 and pt_val != 8:
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
            elif b_total == 4 and 2 <= pt_val <= 7:
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
            elif b_total == 5 and 4 <= pt_val <= 7:
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
            elif b_total == 6 and pt_val in (6, 7):
                banker.append(draw_card(counts))
                b_total = hand_total(banker)
            # else banker stands

    if p_total > b_total:
        return "P"
    if b_total > p_total:
        return "B"
    return "T"


def simulate_probabilities(
    known_player: List[str],
    known_banker: List[str],
    decks: int = 8,
    n_sims: int = 50000,
) -> Tuple[float, float, float]:
    """
    Monte Carlo estimate of P/B/T probabilities given some known starting cards.

    known_player / known_banker:
      - list of ranks using A,2-9,0 (0 = any 10/J/Q/K).
      - Example for step 1: known_player=['9'], known_banker=[]
      - Example for step 2: known_player=['9'], known_banker=['5']
    """
    wins_p = wins_b = ties = 0
    for _ in range(n_sims):
        counts = make_shoe(decks)
        outcome = play_baccarat_round(counts, known_player, known_banker)
        if outcome == "P":
            wins_p += 1
        elif outcome == "B":
            wins_b += 1
        else:
            ties += 1

    total = wins_p + wins_b + ties
    return wins_p / total, wins_b / total, ties / total


def parse_card_list(text: str) -> List[str]:
    """
    Parse a comma-separated list of ranks into our internal format.
    Accepted tokens (case-insensitive):
      A,2,3,4,5,6,7,8,9,10,J,Q,K
    We map 10/J/Q/K -> '0'.
    """
    text = text.strip()
    if not text:
        return []
    parts = [p.strip().upper() for p in text.replace(" ", "").split(",") if p.strip()]
    out: List[str] = []
    for p in parts:
        if p == "A":
            out.append("A")
        elif p in {"2", "3", "4", "5", "6", "7", "8", "9"}:
            out.append(p)
        elif p in {"10", "T", "J", "Q", "K"}:
            out.append("0")
        else:
            raise ValueError(f"Unrecognized rank: {p}")
    return out


def main() -> None:
    print("Baccarat probability simulator (Monte Carlo)")
    print("Input ranks as A,2-9,10,J,Q,K (comma-separated). 10/J/Q/K are treated the same.")
    print("Example step 1 (only one player card known): '9'")
    print("Example step 2 (player 9, banker 5): '9' and '5'")
    print()
    print("For odds, enter NET payout per 1 unit bet (win profit only).")
    print("  Examples: Player 1.0, Banker 0.95, Tie 8.0")

    try:
        p_txt = input("Known PLAYER cards (e.g. 9 or 9,4): ").strip()
        b_txt = input("Known BANKER cards (e.g. 5 or empty): ").strip()
    except (EOFError, KeyboardInterrupt):
        return

    try:
        known_p = parse_card_list(p_txt)
        known_b = parse_card_list(b_txt)
    except ValueError as e:
        print("Error:", e)
        return

    # Casino odds (net profit per 1 unit bet on that outcome)
    def _parse_odds(txt: str, default: float) -> float:
        """
        Accept either:
          - decimal net odds (e.g. 1.0, 0.95, 8.0)
          - American odds like -112, +150, etc.
        Returns net profit per 1 unit staked.
        """
        if not txt:
            return default
        txt = txt.strip()
        try:
            val = float(txt)
        except ValueError:
            return default

        # If magnitude is relatively large, interpret as American odds.
        if abs(val) >= 1 and abs(val) >= 10:
            if val > 0:
                # +150 -> win 1.5 per 1 staked
                return val / 100.0
            else:
                # -112 -> win 100/112 ≈ 0.8929 per 1 staked
                return 100.0 / abs(val)

        # Otherwise treat as already net odds (e.g. 0.95, 8.0)
        return val

    def _get_float(prompt: str, default: float) -> float:
        try:
            txt = input(f"{prompt} [{default}]: ").strip()
        except (EOFError, KeyboardInterrupt):
            return default
        val = _parse_odds(txt, default)
        if txt and val == default:
            print("  Could not parse, using default.")
        return val

    print()
    player_odds = _get_float("Casino net odds for PLAYER", 1.0)
    banker_odds = _get_float("Casino net odds for BANKER", 0.95)
    tie_odds = _get_float("Casino net odds for TIE", 8.0)

    decks = 8
    n_sims = 50000
    print(f"\nSimulating with {decks} decks, {n_sims} rounds...")
    p_prob, b_prob, t_prob = simulate_probabilities(
        known_p, known_b, decks=decks, n_sims=n_sims
    )

    print("\nEstimated probabilities (given the known cards):")
    print(f"  Player win: {p_prob*100:6.2f}%")
    print(f"  Banker win: {b_prob*100:6.2f}%")
    print(f"  Tie:        {t_prob*100:6.2f}%")

    # Expected value of a 1-unit bet on each option given casino odds
    def ev(prob: float, net_odds: float) -> float:
        # win: +net_odds, lose: -1
        return prob * net_odds - (1.0 - prob)

    ev_p = ev(p_prob, player_odds)
    ev_b = ev(b_prob, banker_odds)
    ev_t = ev(t_prob, tie_odds)

    print("\nExpected value per 1-unit bet (given odds):")
    print(f"  Player EV: {ev_p:+.4f}  ({'better' if ev_p>0 else 'worse'} than fair)")
    print(f"  Banker EV: {ev_b:+.4f}  ({'better' if ev_b>0 else 'worse'} than fair)")
    print(f"  Tie EV:    {ev_t:+.4f}  ({'better' if ev_t>0 else 'worse'} than fair)")


if __name__ == "__main__":
    main()
