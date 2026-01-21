"""
Simple text UI to simulate Texas Hold'em decision spots and collect data
for training the poker model.

Phase 1 goal:
- Randomly generate preflop situations.
- Show hero hole cards and basic context.
- Ask the user for an action (fold / call / raise).
- Show the current model's recommendation.
- Log state + chosen action to a CSV file for later training.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List

from ai_poker import Action, PokerState, SimpleEquityBasedAgent, Street


RANKS = "23456789TJQKA"
SUITS = "cdhs"  # clubs, diamonds, hearts, spades


def make_deck() -> List[str]:
    return [r + s for r in RANKS for s in SUITS]


def deal_hole_cards(rng: random.Random) -> List[str]:
    deck = make_deck()
    rng.shuffle(deck)
    return deck[:2]


def prompt_user_action() -> Action:
    while True:
        raw = input("Your action [f=fold, c=call, r=raise, q=quit]: ").strip().lower()
        if raw in {"q", "quit"}:
            raise KeyboardInterrupt
        if raw in {"f", "fold"}:
            return Action.FOLD
        if raw in {"c", "call"}:
            return Action.CALL
        if raw in {"r", "raise"}:
            return Action.RAISE
        print("Invalid input. Please enter f, c, r, or q.")


def append_training_row(csv_path: Path, state: PokerState, user_action: Action, model_action: Action, confidence: float) -> None:
    header = [
        "hole_1",
        "hole_2",
        "board",
        "pot",
        "to_call",
        "effective_stack",
        "street",
        "players_total",
        "players_active",
        "user_action",
        "model_action",
        "model_confidence",
    ]

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(
            {
                "hole_1": state.hero_hole[0],
                "hole_2": state.hero_hole[1],
                "board": " ".join(state.board),
                "pot": state.pot,
                "to_call": state.to_call,
                "effective_stack": state.effective_stack,
                "street": state.street.name,
                "players_total": state.players_total,
                "players_active": state.players_active,
                "user_action": user_action.value,
                "model_action": model_action.value,
                "model_confidence": confidence,
            }
        )


def main() -> None:
    rng = random.Random(42)
    agent = SimpleEquityBasedAgent()
    out_path = Path("training_data_preflop.csv")

    print("=== AI Poker Preflop Simulator ===")
    print("This will generate random preflop spots.")
    print("The model will show a recommendation; you choose an action.")
    print("Data is appended to", out_path.resolve())
    print("Press q at the action prompt to exit.\n")

    hand_counter = 0
    try:
        while True:
            hand_counter += 1
            hero_hole = deal_hole_cards(rng)

            # Simple random context
            players_total = rng.randint(2, 9)
            players_active = rng.randint(2, players_total)
            pot = 1.5  # blinds 0.5/1 for example
            to_call = random.choice([0.0, 0.5, 1.0, 3.0])
            effective_stack = 100.0

            state = PokerState(
                hero_hole=hero_hole,
                board=[],
                pot=pot,
                to_call=to_call,
                effective_stack=effective_stack,
                street=Street.PREFLOP,
                players_total=players_total,
                players_active=players_active,
            )

            rec = agent.recommend(state)

            print(f"\n--- Hand {hand_counter} ---")
            print(f"Hero hole cards: {hero_hole[0]} {hero_hole[1]}")
            print(f"Players: {players_active}/{players_total}")
            print(f"Pot: {pot}   To call: {to_call}   Stack: {effective_stack}")
            print(f"Model suggests: {rec.action.value.upper()} (confidence={rec.confidence:.2f})")
            if rec.comment:
                print("Model comment:", rec.comment)

            try:
                user_action = prompt_user_action()
            except KeyboardInterrupt:
                print("\nExiting.")
                break

            append_training_row(out_path, state, user_action, rec.action, rec.confidence)
            print("Saved decision to training data.")

    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()

