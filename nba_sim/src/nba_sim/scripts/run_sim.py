from __future__ import annotations

"""
Thin wrapper so `nba_sim.scripts.run_sim:main` resolves correctly.
It delegates to the top-level `scripts.run_sim` module (under src/scripts).
"""

from scripts.run_sim import main  # type: ignore F401


if __name__ == "__main__":
    main()

