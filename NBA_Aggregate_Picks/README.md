# NBA Picks Aggregator

A Python-based NBA picks aggregator that scrapes and collects expert picks from multiple sources:
- **OddsShark** - Computer picks for player props and game spreads
- **CBS Sports** - Expert spreads and over/unders
- **PicksWise** - Picks with team and player prop coverage
- **SBR** (Sports Book Review) - Coming soon
- **Sports Chat Place** - Coming soon

## Features

- Aggregates picks from multiple sources
- Separates game picks (spreads, O/U) from player props
- Normalizes team codes and bet formats
- Exports to CSV with detailed sourcing
- Generates reports with pick tallies
- Includes matchup information

## Output Files

Results are saved to `./results/` folder:
- `raw_picks_YYYYMMDD.csv` - All picks from all sources
- `cleaned_picks_YYYYMMDD.csv` - Normalized picks
- `super_cleaned_picks.csv` - Final cleaned picks
- `spreads_recommendations.txt` - Text report

## CSV Columns

- **pick** - The actual pick (e.g., "PHO -1.5", "J. Green O13.5")
- **pick_type** - Type of pick (spread, over_under, etc.)
- **matchup** - Game matchup information
- **source** - Source website
- **expert** - Expert or player name
- **confidence** - Confidence level (0-1)
- **source_url** - Link to source

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python nba_picks_all_in_one.py
```

## Requirements

- Python 3.8+
- requests
- beautifulsoup4
- pandas

See `requirements.txt` for full dependencies.

## Author

Created for NBA picks aggregation and analysis.
