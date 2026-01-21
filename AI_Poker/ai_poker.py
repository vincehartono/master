"""
AI Poker project scaffold.

Goal (phase 1):
- Use a Texas Hold'em simulation library to generate hands and game states.
- Train/evaluate a simple decision model that recommends actions
  (fold / call / raise) given hole cards, board cards, and basic context.

Later (phase 2):
- Replace the simulated environment with screen capture + state parsing
  from a live poker client, then reuse the decision model.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional


class Street(Enum):
    PREFLOP = auto()
    FLOP = auto()
    TURN = auto()
    RIVER = auto()


class Action(Enum):
    FOLD = "fold"
    CALL = "call"
    RAISE = "raise"


@dataclass
class PokerState:
    """
    Minimal game state description for a single decision.
    This is what the AI will see.
    """

    hero_hole: List[str]  # e.g. ["As", "Kd"]
    board: List[str]  # community cards, 0–5
    pot: float
    to_call: float
    effective_stack: float
    street: Street
    players_total: int
    players_active: int


@dataclass
class Recommendation:
    action: Action
    confidence: float
    comment: str = ""


class SimpleEquityBasedAgent:
    """
    Placeholder AI: uses approximate hand strength (equity) to recommend actions.

    Phase 1: plug into a Texas Hold'em library (e.g. eval7, treys) to estimate
    win probability vs random ranges in simulation.
    For now, we stub an interface so the rest of the project can be wired.
    """

    def recommend(self, state: PokerState) -> Recommendation:
        # TODO: replace this heuristic with real equity calculation.
        strength = self._heuristic_strength(state)

        # Very crude thresholds for demonstration
        if strength < 0.3:
            action = Action.FOLD if state.to_call > 0 else Action.CALL
            comment = "Weak hand: folding to pressure." if state.to_call > 0 else "Check with weak hand."
        elif strength < 0.6:
            action = Action.CALL
            comment = "Medium-strength hand: calling."
        else:
            action = Action.RAISE
            comment = "Strong hand: play aggressively."

        return Recommendation(action=action, confidence=strength, comment=comment)

    def _heuristic_strength(self, state: PokerState) -> float:
        """
        Temporary heuristic: assigns a fake 'strength' based on hole cards only.
        Replace with Monte Carlo equity vs ranges using a poker library.
        """
        ranks = "".join(card[0] for card in state.hero_hole)
        suited = state.hero_hole[0][1] == state.hero_hole[1][1] if len(state.hero_hole) == 2 else False

        # Simple ranking of some strong starting hands
        premium = {"AA", "KK", "QQ", "JJ", "AK", "AQ"}
        if ranks in premium or ranks[::-1] in premium:
            base = 0.8
        else:
            base = 0.4

        if suited:
            base += 0.05

        # Clamp between 0 and 1
        return max(0.0, min(1.0, base))


def example_usage() -> None:
    """
    Small demo using a fake state.
    Run this file directly to see a sample recommendation.
    """
    agent = SimpleEquityBasedAgent()

    state = PokerState(
        hero_hole=["As", "Kd"],
        board=[],
        pot=1.5,
        to_call=0.5,
        effective_stack=100.0,
        street=Street.PREFLOP,
        players_total=6,
        players_active=3,
    )

    rec = agent.recommend(state)
    print("Hero hole:", state.hero_hole)
    print("Street:", state.street.name)
    print("Pot:", state.pot, "To call:", state.to_call)
    print("Recommendation:", rec.action.value, f"(confidence={rec.confidence:.2f})")
    if rec.comment:
        print("Comment:", rec.comment)


if __name__ == "__main__":
    example_usage()

