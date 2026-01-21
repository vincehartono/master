import csv
from collections import defaultdict
from datetime import datetime
import re

# Read cleaned picks
picks_by_matchup = defaultdict(lambda: defaultdict(list))

with open('./results/super_cleaned_picks.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['pick_type'] == 'spread':
            matchup = row['matchup']
            pick = row['pick']
            source = row['source']
            picks_by_matchup[matchup][pick].append(source)

# Find non-contradicting spreads
non_contradicting = {}

for matchup, picks in sorted(picks_by_matchup.items()):
    # Get all spread picks for this matchup
    spread_picks = list(picks.items())
    
    if len(spread_picks) == 1:
        # Only one spread for this matchup - no contradiction
        pick, sources = spread_picks[0]
        non_contradicting[matchup] = pick, sources
    elif len(spread_picks) == 2:
        pick1, sources1 = spread_picks[0]
        pick2, sources2 = spread_picks[1]
        
        # Extract team and value from pick
        # Format: "TEAM +/- VALUE"
        match1 = re.search(r'([A-Z]{2,3})\s+([+-]\d+\.?\d*)', pick1)
        match2 = re.search(r'([A-Z]{2,3})\s+([+-]\d+\.?\d*)', pick2)
        
        if match1 and match2:
            team1, value1 = match1.group(1), float(match1.group(2))
            team2, value2 = match2.group(1), float(match2.group(2))
            
            # Check if they are opposite sides (e.g., LAL -3.5 vs DEN +2.5 for LAL vs DEN)
            if team1 != team2:
                # This is expected (opposite sides of same matchup)
                continue
            
            # Same team but different values - contradiction
            if team1 == team2 and value1 != value2:
                continue
    
    # If multiple picks with different values for same team
    if len(spread_picks) > 2:
        # Check if all are same team
        teams = set()
        for pick, sources in spread_picks:
            match = re.search(r'([A-Z]{2,3})\s+([+-]\d+\.?\d*)', pick)
            if match:
                teams.add(match.group(1))
        
        if len(teams) == 1:
            # All same team - check for duplicates
            values = []
            for pick, sources in spread_picks:
                match = re.search(r'([A-Z]{2,3})\s+([+-]\d+\.?\d*)', pick)
                if match:
                    values.append(float(match.group(2)))
            
            # If all values same, it's not contradicting
            if len(set(values)) == 1:
                pick, sources = spread_picks[0]
                non_contradicting[matchup] = pick, sources

# Generate report
with open('./results/spreads_recommendations.txt', 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("NBA PICKS AGGREGATOR - NON-CONTRADICTING SPREADS\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 80 + "\n\n")
    f.write("Spread picks that do NOT contradict (same pick from same team across sources):\n")
    f.write("=" * 80 + "\n\n")
    
    if non_contradicting:
        for matchup in sorted(non_contradicting.keys()):
            pick, sources = non_contradicting[matchup]
            f.write(f"\n{matchup}\n")
            f.write("-" * 40 + "\n")
            f.write(f"  {pick}: {len(sources)} pick(s)\n")
            f.write(f"    Sources: {', '.join(sources)}\n")
    else:
        f.write("\nNo non-contradicting spreads found.\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("Summary of all spreads with contradictions:\n")
    f.write("=" * 80 + "\n\n")
    
    for matchup in sorted(picks_by_matchup.keys()):
        picks = picks_by_matchup[matchup]
        f.write(f"{matchup}\n")
        f.write("-" * 40 + "\n")
        for pick in sorted(picks.keys()):
            sources = picks[pick]
            f.write(f"  {pick}: {len(sources)} pick(s)\n")
            f.write(f"    Sources: {', '.join(sources)}\n")
        f.write("\n")

print("Report generated successfully!")
