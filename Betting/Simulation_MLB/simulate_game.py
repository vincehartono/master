import pandas as pd
import random
import os
from mlb_constants import team_abbr
import unicodedata

# ---- Player and Pitcher Classes ----
class Player:
    def __init__(self, name, stats):
        self.name = name
        self.stats = stats
        self.hits = 0
        self.rbi = 0

class Pitcher:
    def __init__(self, name, stats, max_pitches=100, hand='R'):
        self.name = name
        self.stats = stats
        self.pitches_thrown = 0
        self.max_pitches = max_pitches
        self.earned_runs = 0
        self.outs_recorded = 0
        self.hand = hand

    def is_fatigued(self):
        return self.pitches_thrown > self.max_pitches or self.outs_recorded >= 18

    def adjusted_stats(self):
        # Progressive fatigue curve based on % of max_pitches
        pct_used = min(1.0, self.pitches_thrown / self.max_pitches)
        fatigue_factor = 1 + 0.5 * pct_used  # up to +50% effect at full fatigue

        if self.pitches_thrown > 90:
            fatigue_factor += (self.pitches_thrown - 90) / 30  # steeper drop-off after 90
        elif self.pitches_thrown > 70:
            fatigue_factor += (self.pitches_thrown - 70) / 60  # slower degradation between 70–90

        return {
            "K%": self.stats['K%'] / fatigue_factor,
            "BB%": self.stats['BB%'] * fatigue_factor,
            "HR/9": self.stats['HR/9'] * fatigue_factor
        }

# ---- Team Class ----
class Team:
    def __init__(self, name, batters, starter, bullpen):
        self.name = name
        self.lineup = batters
        self.pitchers = [starter] + bullpen
        self.current_pitcher_index = 0
        self.score = 0
        self.lineup_index = 0

    def next_batter(self):
        batter = self.lineup[self.lineup_index]
        self.lineup_index = (self.lineup_index + 1) % len(self.lineup)
        return batter

    def current_pitcher(self):
        return self.pitchers[self.current_pitcher_index]

    def maybe_substitute_pitcher(self):
        if self.current_pitcher().is_fatigued() and self.current_pitcher_index < len(self.pitchers) - 1:
            self.current_pitcher_index += 1
            print(f"\n🔄 {self.name} brings in a new pitcher: {self.current_pitcher().name}\n")

# ---- Game Simulation ----
class Game:
    def __init__(self, team1, team2):
        self.inning = 1
        self.half = 'top'
        self.teams = {'top': team1, 'bottom': team2}
        self.bases = [None, None, None]
        self.allow_mid_inning_pitch_change = False

    def simulate_game(self, allow_mid_inning_pitch_change=False):
        self.allow_mid_inning_pitch_change = allow_mid_inning_pitch_change
        while self.inning <= 9 or self.teams['top'].score == self.teams['bottom'].score:
            print(f"\n--- Inning {self.inning} ({self.half}) ---")
            self.simulate_half_inning()
            if self.half == 'bottom':
                self.inning += 1
            self.half = 'bottom' if self.half == 'top' else 'top'

        print("\nFINAL SCORE:")
        for side, team in self.teams.items():
            print(f"{team.name}: {team.score}")

    def simulate_half_inning(self):
        outs = 0
        self.bases = [None, None, None]
        batting_team = self.teams[self.half]
        pitching_team = self.teams['bottom' if self.half == 'top' else 'top']

        while outs < 3:
            if self.allow_mid_inning_pitch_change:
                pitcher = pitching_team.current_pitcher()
                if pitcher.is_fatigued() and pitching_team.current_pitcher_index < len(pitching_team.pitchers) - 1:
                    pitching_team.current_pitcher_index += 1
                    pitcher = pitching_team.current_pitcher()
                    print(f"\n⚠️ Mid-inning pitching change! {pitching_team.name} brings in {pitcher.name}")
            else:
                if pitching_team.current_pitcher().is_fatigued():
                    pitching_team.maybe_substitute_pitcher()
                pitcher = pitching_team.current_pitcher()

            batter = batting_team.next_batter()
            outcome = simulate_at_bat(batter, pitcher)
            print(f"{batter.name} vs {pitcher.name}: {outcome.replace('_', ' ')}!")

            if pitcher.is_fatigued():
                print(f"⚠️  {pitcher.name} is fatigued! Performance is degrading.")

            if outcome in ["strikeout", "out"]:
                outs += 1
                pitcher.outs_recorded += 1
            elif outcome in ["walk", "single"]:
                self.advance_runners(batting_team, 1, batter)
                if outcome == "single":
                    batter.hits += 1
            elif outcome == "double":
                self.advance_runners(batting_team, 2, batter)
                batter.hits += 1
            elif outcome == "triple":
                self.advance_runners(batting_team, 3, batter)
                batter.hits += 1
            elif outcome == "home_run":
                self.advance_runners(batting_team, 4, batter)
                batter.hits += 1
                batter.rbi += 1
                pitcher.earned_runs += 1

    def advance_runners(self, team, bases_advanced, batter):
        runs_scored = 0
        new_bases = [None, None, None]

        for i in reversed(range(3)):
            runner = self.bases[i]
            if runner:
                if i + bases_advanced >= 3:
                    runs_scored += 1
                    print(f"{runner.name} scores!")
                else:
                    new_bases[i + bases_advanced] = runner

        if bases_advanced >= 4:
            runs_scored += 1
            print(f"{batter.name} scores on a home run!")
        else:
            new_bases[bases_advanced - 1] = batter

        self.bases = new_bases
        team.score += runs_scored

    def get_player_stats(self):
        stats = []
        for side in ['top', 'bottom']:
            team = self.teams[side]
            for player in team.lineup:
                stats.append({
                    'Team': team.name,
                    'Player': player.name,
                    'Hits': player.hits,
                    'RBI': player.rbi,
                    'Pitcher': False
                })
            for pitcher in team.pitchers:
                stats.append({
                    'Team': team.name,
                    'Player': pitcher.name,
                    'Hits': None,
                    'RBI': None,
                    'Pitcher': True,
                    'IP': round(pitcher.outs_recorded / 3, 2),
                    'ER': pitcher.earned_runs
                })
        return stats

# ---- Simulation Logic ----
def simulate_at_bat(batter, pitcher):
    pitcher.pitches_thrown += 4
    p = pitcher.adjusted_stats()
    b = batter.stats

    k = min(b.get("strikeout", 0.2) * (1 + p['K%']), 0.5)
    bb = min(b.get("walk", 0.1) * (1 + p['BB%']), 0.3)
    hr = max(b.get("home_run", 0.04) * (1 + p['HR/9'] / 9), 0.01)

    single = b.get("single", 0.15)
    double = b.get("double", 0.05)
    triple = b.get("triple", 0.01)

    # ✅ Platoon logic
    batter_side = b.get("side", "R")
    pitcher_hand = getattr(pitcher, "hand", "R")
    if batter_side in ["L", "R"]:
        if batter_side == pitcher_hand:
            k *= 1.05
            bb *= 0.95
            hr *= 0.95
        else:
            k *= 0.95
            bb *= 1.05
            hr *= 1.05

    out = max(0.0, min(1.0 - (k + bb + hr + single + double + triple), 1.0))
    outcomes = ["strikeout", "walk", "home_run", "single", "double", "triple", "out"]
    weights = [k, bb, hr, single, double, triple, out]
    return random.choices(outcomes, weights=weights, k=1)[0]

# ---- Build Teams ----
def estimate_batter_prob(row):
    pa = row['PA']
    if pa < 50: return None
    bb = row['BB%'] / 100
    k = row['K%'] / 100
    hr = row['HR'] / pa
    avg = row['AVG']
    single = max(avg - hr - 0.05, 0.05)
    return {
        "walk": round(bb, 3),
        "strikeout": round(k, 3),
        "home_run": round(hr, 3),
        "single": round(single, 3),
        "double": 0.05,
        "triple": 0.01,
        "side": row.get("Side", "R")  # ✅ NEW
    }

def estimate_pitcher_stats(row):
    return {
        "K%": row.get('K%', 20.0) / 100,
        "BB%": row.get('BB%', 8.0) / 100,
        "HR/9": row.get('HR/9', 1.1)
    }

def normalize_name(name):
    return unicodedata.normalize('NFKD', name).encode('ASCII', 'ignore').decode().lower()

def build_team(batting_df, pitching_df, team_code, name, starter_name=None):
    # Select top 9 batters by PA
    batters = batting_df[batting_df['Team'] == team_code].sort_values('PA', ascending=False).head(9)
    lineup = [
        Player(row['Name'], estimate_batter_prob(row))
        for _, row in batters.iterrows()
        if estimate_batter_prob(row)
    ]

    team_pitchers = pitching_df[pitching_df['Team'] == team_code]

    # Find and normalize the starter
    if starter_name and isinstance(starter_name, str):
        starter_row = team_pitchers[
            team_pitchers['Name'].apply(lambda x: normalize_name(x)) == normalize_name(starter_name)
        ]
        if starter_row.empty:
            raise ValueError(f"Starter {starter_name} not found for team {team_code}")
        starter = Pitcher(
            starter_row.iloc[0]['Name'],
            estimate_pitcher_stats(starter_row.iloc[0]),
            max_pitches=100,
            hand=starter_row.iloc[0].get("Throws", "R")  # ✅ NEW
        )
        remaining = team_pitchers[team_pitchers['Name'] != starter_row.iloc[0]['Name']]
    else:
        starters = team_pitchers[team_pitchers['GS'] > 0].sort_values('IP', ascending=False).head(1)
        if starters.empty:
            raise ValueError(f"No starters found for team {team_code}")
        starter = Pitcher(
            starter_row.iloc[0]['Name'],
            estimate_pitcher_stats(starter_row.iloc[0]),
            max_pitches=100,
            hand=starter_row.iloc[0].get("Throws", "R")  # ✅ NEW
        )
        remaining = team_pitchers[team_pitchers['Name'] != starters.iloc[0]['Name']]

    # Build bullpen with top 3 relievers by IP
    relievers = remaining[remaining['GS'] == 0].sort_values('IP', ascending=False).head(3)
    bullpen = [
        Pitcher(
            row['Name'],
            estimate_pitcher_stats(row),
            max_pitches=30,
            hand=row.get("Throws", "R")
        )
        for _, row in relievers.iterrows()
    ]

    return Team(name, lineup, starter, bullpen)

# ---- Main Test ----
if __name__ == "__main__":
    data_dir = r"C:\Users\Vince\master\Betting\Simulation_MLB"
    print("Loading batting and pitching data from .parquet files...")
    batting_df = pd.read_parquet(os.path.join(data_dir, "batting_2023_2025.parquet"))
    pitching_df = pd.read_parquet(os.path.join(data_dir, "pitching_2023_2025.parquet"))

    yankees = build_team(batting_df, pitching_df, "NYY", "Yankees")
    dodgers = build_team(batting_df, pitching_df, "LAD", "Dodgers")

    game = Game(yankees, dodgers)
    game.simulate_game(allow_mid_inning_pitch_change=True)
