import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time

# Input and output paths
input_file = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\mlb_atbat_data_filtered.parquet"
output_dir = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\simulation_results"

# Create output directory
Path(output_dir).mkdir(parents=True, exist_ok=True)

class MLBGameSimulation:
    def __init__(self, data_path):
        print("Initializing MLB Game Simulation...")
        self.load_data(data_path)
        self.setup_probability_models()
    
    def load_data(self, data_path):
        """Load and process the MLB at-bat data"""
        print("Loading at-bat data...")
        self.df = pd.read_parquet(data_path)
        print(f"Loaded {len(self.df):,} at-bats")
        
        # Store teams
        if 'batter_team' in self.df.columns:
            self.teams = sorted(self.df['batter_team'].unique())
            print(f"Found {len(self.teams)} teams")
        else:
            # Create placeholder teams if team column doesn't exist
            self.teams = ["Team A", "Team B"]
            print("Warning: No team data found, using placeholder teams")
    
    def setup_probability_models(self):
        """Create statistical models from the data"""
        print("Building probability models...")
        
        # Identify relevant columns
        self.has_teams = 'batter_team' in self.df.columns and 'pitcher_team' in self.df.columns
        self.has_players = 'batter_id' in self.df.columns and 'pitcher_id' in self.df.columns
        
        # Base outcome probabilities
        if 'event' in self.df.columns:
            self.outcomes = self.df['event'].value_counts(normalize=True).to_dict()
        else:
            # Default outcomes if event column doesn't exist
            self.outcomes = {
                'single': 0.15,
                'double': 0.05,
                'triple': 0.01,
                'home_run': 0.03,
                'walk': 0.08,
                'strikeout': 0.22,
                'field_out': 0.40,
                'other_out': 0.06
            }
        
        # Team batting stats (if available)
        if self.has_teams:
            self.team_batting = {}
            for team in self.teams:
                team_abs = self.df[self.df['batter_team'] == team]
                
                # Calculate team batting outcomes
                if 'event' in self.df.columns:
                    self.team_batting[team] = team_abs['event'].value_counts(normalize=True).to_dict()
                # Calculate team batting average if hit column exists
                elif 'hit' in self.df.columns:
                    hit_rate = team_abs['hit'].mean()
                    self.team_batting[team] = {
                        'hit': hit_rate,
                        'out': 1 - hit_rate
                    }
        
        # Runner advancement probabilities
        self.runner_adv = {
            'single': {'1b_to_2b': 0.0, '1b_to_3b': 0.0, '1b_to_home': 0.0, 
                      '2b_to_3b': 0.0, '2b_to_home': 0.93, 
                      '3b_to_home': 0.98},
            'double': {'1b_to_3b': 0.0, '1b_to_home': 0.78, 
                      '2b_to_home': 0.93, 
                      '3b_to_home': 0.98},
            'triple': {'1b_to_home': 0.98, 
                      '2b_to_home': 0.99, 
                      '3b_to_home': 1.0},
            'field_out': {'1b_to_2b': 0.25, '1b_to_3b': 0.08, '1b_to_home': 0.02,
                         '2b_to_3b': 0.33, '2b_to_home': 0.14,
                         '3b_to_home': 0.51}
        }
        
        # Extract inning-specific models if column exists
        if 'inning' in self.df.columns and 'event' in self.df.columns:
            self.inning_models = {}
            for inning in sorted(self.df['inning'].unique()):
                inning_data = self.df[self.df['inning'] == inning]
                self.inning_models[inning] = inning_data['event'].value_counts(normalize=True).to_dict()
                
        print("Probability models built successfully")
    
    def simulate_game(self, home_team, away_team, num_innings=9):
        """Simulate a complete game between two teams"""
        print(f"\nSimulating: {away_team} @ {home_team}")
        
        # Game state
        score = {home_team: 0, away_team: 0}
        inning_scores = {home_team: [0] * num_innings, away_team: [0] * num_innings}
        game_log = []
        
        # Create boxscore 
        boxscore = {
            'team': [],
            '1': [], '2': [], '3': [], '4': [], '5': [], 
            '6': [], '7': [], '8': [], '9': [],
            'R': [], 'H': [], 'E': []
        }
        
        # Initialize hit counters
        hits = {home_team: 0, away_team: 0}
        errors = {home_team: 0, away_team: 0} # For realism, random errors
        
        # Simulate each inning
        for inning in range(1, num_innings + 1):
            # Top of inning (away team batting)
            batting_team, fielding_team = away_team, home_team
            game_log.append(f"\n{'='*60}")
            game_log.append(f"TOP OF INNING {inning}: {batting_team} batting")
            game_log.append(f"{'-'*60}")
            
            runs, inning_log, inning_hits = self.simulate_half_inning(inning, batting_team, fielding_team, is_home=False)
            score[batting_team] += runs
            hits[batting_team] += inning_hits
            
            # Random error chance (1 in 10 innings)
            if random.random() < 0.1:
                errors[fielding_team] += 1
                
            if inning <= len(inning_scores[batting_team]):
                inning_scores[batting_team][inning-1] = runs
            game_log.extend(inning_log)
            
            # Update boxscore
            if inning == 1:  # Add team name only on first inning
                boxscore['team'].append(batting_team)
            boxscore[str(inning)].append(runs)
            
            # Bottom of inning (home team batting)
            batting_team, fielding_team = home_team, away_team
            # Skip bottom of 9th if home team is ahead
            if inning == num_innings and score[home_team] > score[away_team]:
                game_log.append(f"\n{'='*60}")
                game_log.append(f"BOTTOM OF INNING {inning}: Home team already ahead, game over!")
                if inning == 1:  # Add team name only on first inning
                    boxscore['team'].append(batting_team)
                boxscore[str(inning)].append("X")  # X indicates bottom not played
                break
                
            game_log.append(f"\n{'='*60}")
            game_log.append(f"BOTTOM OF INNING {inning}: {batting_team} batting")
            game_log.append(f"{'-'*60}")
            
            runs, inning_log, inning_hits = self.simulate_half_inning(inning, batting_team, fielding_team, is_home=True)
            score[batting_team] += runs
            hits[batting_team] += inning_hits
            
            # Random error chance (1 in 10 innings)
            if random.random() < 0.1:
                errors[fielding_team] += 1
                
            if inning <= len(inning_scores[batting_team]):
                inning_scores[batting_team][inning-1] = runs
            game_log.extend(inning_log)
            
            # Update boxscore
            if inning == 1:  # Add team name only on first inning
                boxscore['team'].append(batting_team)
            boxscore[str(inning)].append(runs)
            
            # Check if game ended due to walk-off
            if inning == num_innings and score[home_team] > score[away_team]:
                game_log.append("\nWALK-OFF WIN for the home team!")
            
            # Extra innings if tied after regulation
            if inning == num_innings and score[home_team] == score[away_team]:
                num_innings += 1
                game_log.append("\nGame tied after 9 innings. Going to extra innings!")
                for team in [home_team, away_team]:
                    inning_scores[team].append(0)
                # Add extra inning column to boxscore
                boxscore[str(inning+1)] = []
        
        # Complete boxscore with R/H/E
        boxscore['R'] = [score[away_team], score[home_team]]
        boxscore['H'] = [hits[away_team], hits[home_team]]
        boxscore['E'] = [errors[away_team], errors[home_team]]
        
        # Game results
        winner = home_team if score[home_team] > score[away_team] else away_team
        
        # Final score summary
        game_log.append(f"\n{'='*60}")
        game_log.append(f"FINAL SCORE: {away_team} {score[away_team]}, {home_team} {score[home_team]}")
        game_log.append(f"WINNER: {winner}")
        game_log.append(f"{'='*60}")
        
        # Print nicely formatted boxscore
        boxscore_text = self.format_boxscore(boxscore)
        game_log.append("\nBOXSCORE:")
        game_log.extend(boxscore_text)
        
        game_result = {
            'home_team': home_team,
            'away_team': away_team,
            'score': score,
            'inning_scores': inning_scores,
            'winner': winner,
            'game_log': game_log,
            'boxscore': boxscore
        }
        
        # Save game log to file
        self.save_game_log(game_result)
        
        print(f"Final Score: {away_team} {score[away_team]}, {home_team} {score[home_team]}")
        return game_result
    
    def simulate_half_inning(self, inning, batting_team, fielding_team, is_home):
        """Simulate a half inning"""
        outs = 0
        runs = 0
        hits = 0
        bases = [0, 0, 0]  # 0 = empty, 1 = runner on base
        inning_log = []
        
        # Continue until 3 outs
        at_bat_num = 1
        while outs < 3:
            # Simulate at-bat
            result, outcome_detail = self.simulate_at_bat(batting_team, fielding_team, inning, bases, outs)
            
            # Log the result
            at_bat_log = f"At-bat #{at_bat_num}: {outcome_detail}"
            inning_log.append(at_bat_log)
            at_bat_num += 1
            
            # Process result
            if result == 'single':
                hits += 1
                # Advance runners
                new_runs = self.advance_runners(bases, 1, self.runner_adv['single'])
                runs += new_runs
                # Batter to first
                bases[0] = 1
                if new_runs > 0:
                    inning_log.append(f"  → {new_runs} run(s) scored on the single!")
                else:
                    inning_log.append(f"  → Runner on first")
                
            elif result == 'double':
                hits += 1
                # Advance runners
                new_runs = self.advance_runners(bases, 2, self.runner_adv['double'])
                runs += new_runs
                # Batter to second
                bases[1] = 1
                if new_runs > 0:
                    inning_log.append(f"  → {new_runs} run(s) scored on the double!")
                else:
                    inning_log.append(f"  → Runner on second")
                
            elif result == 'triple':
                hits += 1
                # Advance runners
                new_runs = self.advance_runners(bases, 3, self.runner_adv['triple'])
                runs += new_runs
                # Batter to third
                bases[2] = 1
                if new_runs > 0:
                    inning_log.append(f"  → {new_runs} run(s) scored on the triple!")
                else:
                    inning_log.append(f"  → Runner on third")
                
            elif result == 'home_run':
                hits += 1
                # Count runners on base
                runners = sum(bases)
                # All runners score including batter
                runs += runners + 1
                # Clear bases
                bases = [0, 0, 0]
                if runners > 0:
                    inning_log.append(f"  → {runners + 1} run HOME RUN! (That's a {runners+1}-run homer)")
                else:
                    inning_log.append(f"  → SOLO HOME RUN!")
                
            elif result == 'walk' or result == 'hit_by_pitch':
                # Force advancing - only if needed
                if bases[0] == 1:  # Runner on first
                    if bases[1] == 1:  # Runner on second
                        if bases[2] == 1:  # Runner on third
                            # Bases loaded - runner from third scores
                            runs += 1
                            inning_log.append(f"  → Bases loaded walk! 1 run scores!")
                            # Keep first and second full
                        else:
                            # First and second full - runner to third
                            bases[2] = 1
                            inning_log.append(f"  → Walk loads the bases")
                    else:
                        # First base full, second empty - runner to second
                        bases[1] = 1
                        inning_log.append(f"  → Walk, runners advance")
                else:
                    inning_log.append(f"  → Walk, runner on first")
                # Batter to first
                bases[0] = 1
                
            elif result == 'strikeout':
                outs += 1
                inning_log.append(f"  → Strikeout, {outs} out(s)")
                
            elif result == 'field_out' or result == 'other_out':
                outs += 1
                
                # Potential advancing on outs
                if result == 'field_out' and outs < 3:  # No advance on third out
                    new_runs = self.advance_runners_on_out(bases, self.runner_adv['field_out'])
                    runs += new_runs
                    
                    if new_runs > 0:
                        inning_log.append(f"  → {outs} out(s), sacrifice! {new_runs} run(s) scored")
                    else:
                        inning_log.append(f"  → {outs} out(s)")
                else:
                    inning_log.append(f"  → {outs} out(s)")
            
            # Log current situation after every at-bat
            situation = self.describe_situation(bases, outs, runs)
            inning_log.append(f"  → Current situation: {situation}\n")
        
        # End of inning summary
        inning_log.append(f"\n{'-'*40}")
        inning_log.append(f"End of {'Bottom' if is_home else 'Top'} {inning}: {runs} runs, {hits} hits, 3 outs")
        return runs, inning_log, hits
    
    def simulate_at_bat(self, batting_team, fielding_team, inning, bases, outs):
        """Simulate a single at-bat and return the result"""
        # Use team-specific models if available
        if self.has_teams and batting_team in self.team_batting:
            outcome_probs = self.team_batting[batting_team]
        else:
            outcome_probs = self.outcomes
            
        # Adjust for inning if model available
        if hasattr(self, 'inning_models') and inning in self.inning_models:
            # Blend base and inning-specific models
            combined_probs = {}
            for outcome in set(list(outcome_probs.keys()) + list(self.inning_models[inning].keys())):
                base_prob = outcome_probs.get(outcome, 0)
                inning_prob = self.inning_models[inning].get(outcome, 0)
                combined_probs[outcome] = 0.7 * base_prob + 0.3 * inning_prob
            outcome_probs = combined_probs
        
        # Select outcome based on probabilities
        outcomes = list(outcome_probs.keys())
        probabilities = list(outcome_probs.values())
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p/total_prob for p in probabilities]
        
        result = random.choices(outcomes, weights=probabilities, k=1)[0]
        
        # Map to simplified result categories if needed
        simplified_result = self.simplify_outcome(result)
        
        # Create detailed outcome string
        detail = f"{result}"
        
        return simplified_result, detail
    
    def simplify_outcome(self, result):
        """Map detailed outcomes to simplified categories"""
        # Hit outcomes
        if result in ['single', 'double', 'triple', 'home_run']:
            return result
            
        # Walks
        if result in ['walk', 'intent_walk', 'hit_by_pitch']:
            return 'walk'
            
        # Strikeouts
        if 'strikeout' in result.lower():
            return 'strikeout'
            
        # Field outs
        if any(term in result.lower() for term in ['groundout', 'flyout', 'lineout', 'popout', 'forceout', 'double_play']):
            return 'field_out'
            
        # Other outs
        return 'other_out'
    
    def advance_runners(self, bases, hit_bases, advancement_probs):
        """Advance runners based on the hit type and return runs scored"""
        runs = 0
        new_bases = [0, 0, 0]
        
        # Process from 3rd to 1st to avoid overwriting
        # Runner on 3rd
        if bases[2] == 1:
            # Third base runner always scores on hits
            runs += 1
        
        # Runner on 2nd
        if bases[1] == 1:
            if hit_bases == 1:  # Single
                # Check if runner advances home
                if random.random() < advancement_probs.get('2b_to_home', 0.6):
                    runs += 1
                else:
                    new_bases[2] = 1  # Advance to third
            else:  # Extra base hit - runner scores
                runs += 1
        
        # Runner on 1st
        if bases[0] == 1:
            if hit_bases == 1:  # Single
                # Check if runner advances to third (rare)
                if random.random() < advancement_probs.get('1b_to_3b', 0.2):
                    new_bases[2] = 1
                else:
                    new_bases[1] = 1  # Advance to second
            elif hit_bases == 2:  # Double
                # Check if runner scores from first
                if random.random() < advancement_probs.get('1b_to_home', 0.45):
                    runs += 1
                else:
                    new_bases[2] = 1  # Advance to third
            else:  # Triple or HR - runner scores
                runs += 1
        
        # Place batter-runner
        if hit_bases < 4:  # Not a HR
            new_bases[hit_bases-1] = 1
        
        # Update bases - copy new bases state
        for i in range(3):
            bases[i] = new_bases[i]
            
        return runs
    
    def advance_runners_on_out(self, bases, advancement_probs):
        """Handle runner advancement on outs (sacrifice flies, etc.)"""
        runs = 0
        new_bases = [0, 0, 0]
        
        # Runner on 3rd
        if bases[2] == 1:
            # Check for sacrifice fly or productive out
            if random.random() < advancement_probs.get('3b_to_home', 0.3):
                runs += 1
            else:
                new_bases[2] = 1  # Stay at third
        
        # Runner on 2nd
        if bases[1] == 1:
            # Check for advancement to third
            if random.random() < advancement_probs.get('2b_to_3b', 0.4):
                new_bases[2] = 1  # Move to third
            else:
                new_bases[1] = 1  # Stay at second
        
        # Runner on 1st
        if bases[0] == 1:
            # Check for advancement to second
            if random.random() < advancement_probs.get('1b_to_2b', 0.2):
                new_bases[1] = 1  # Move to second
            else:
                new_bases[0] = 1  # Stay at first
        
        # Update bases - copy new bases state
        for i in range(3):
            bases[i] = new_bases[i]
            
        return runs
    
    def describe_situation(self, bases, outs, runs_this_inning):
        """Create a description of the current game situation"""
        base_desc = ""
        if sum(bases) == 0:
            base_desc = "Bases empty"
        else:
            base_positions = []
            if bases[0] == 1:
                base_positions.append("1st")
            if bases[1] == 1:
                base_positions.append("2nd")
            if bases[2] == 1:
                base_positions.append("3rd")
            base_desc = "Runner(s) on " + ", ".join(base_positions)
        
        return f"{base_desc}, {outs} out(s), {runs_this_inning} run(s) this inning"
    
    def format_boxscore(self, boxscore):
        """Format a nice text boxscore"""
        lines = []
        
        # Get all columns that represent innings
        inning_cols = [col for col in boxscore.keys() if col.isdigit()]
        inning_cols.sort(key=int)  # Ensure proper order
        
        # Calculate column widths
        team_width = max(len(team) for team in boxscore['team'])
        team_width = max(team_width, 5)  # Minimum 5 chars for "TEAM"
        
        # Header row
        header = "TEAM".ljust(team_width)
        for inning in inning_cols:
            header += " | " + inning.center(3)
        header += " | " + "R".center(3) + " | " + "H".center(3) + " | " + "E".center(3)
        lines.append(header)
        
        # Separator
        separator = "-" * team_width
        for _ in range(len(inning_cols) + 3):  # +3 for R, H, E
            separator += "-+-" + "-" * 3
        lines.append(separator)
        
        # Team rows
        for i in range(len(boxscore['team'])):
            row = boxscore['team'][i].ljust(team_width)
            for inning in inning_cols:
                if inning in boxscore and i < len(boxscore[inning]):
                    # Center the inning score in 3 chars, handling 'X' case
                    if boxscore[inning][i] == "X":
                        value = " X "
                    else:
                        value = str(boxscore[inning][i]).center(3)
                    row += " | " + value
                else:
                    row += " | " + " " * 3  # Empty cell for missing innings
            
            # Add R, H, E
            for stat in ["R", "H", "E"]:
                if i < len(boxscore[stat]):
                    row += " | " + str(boxscore[stat][i]).center(3)
                else:
                    row += " | " + " " * 3
                    
            lines.append(row)
        
        return lines
    
    def save_game_log(self, game_result):
        """Save the game log to a file"""
        output_path = Path(output_dir) / "game_log.txt"
        
        with open(output_path, "w") as f:
            f.write(f"BASEBALL GAME SIMULATION\n")
            f.write(f"==============================================\n\n")
            f.write(f"{game_result['away_team']} @ {game_result['home_team']}\n\n")
            
            for line in game_result['game_log']:
                f.write(line + "\n")
        
        print(f"Game log saved to {output_path}")
        
        # Create boxscore visual with matplotlib
        self.visualize_boxscore(game_result)
    
    def visualize_boxscore(self, game_result):
        """Create a visual boxscore using matplotlib"""
        boxscore = game_result['boxscore']
        away_team = game_result['away_team']
        home_team = game_result['home_team']
        
        # Get innings played
        inning_cols = [col for col in boxscore.keys() if col.isdigit()]
        inning_cols.sort(key=int)
        
        # Create figure
        plt.figure(figsize=(max(10, len(inning_cols)*1.2), 4))
        
        # Set background color
        plt.gca().set_facecolor('#f8f8f8')
        
        # Title
        final_score = f"{away_team} {boxscore['R'][0]}, {home_team} {boxscore['R'][1]}"
        plt.title(f"Baseball Game Simulation\n{final_score}", fontsize=14)
        
        # Create table data
        table_data = []
        # Header row
        header = ['Team'] + inning_cols + ['R', 'H', 'E']
        table_data.append(header)
        
        # Team rows
        for i in range(len(boxscore['team'])):
            row = [boxscore['team'][i]]
            for inning in inning_cols:
                if inning in boxscore and i < len(boxscore[inning]):
                    row.append(boxscore[inning][i])
                else:
                    row.append('')
            # Add R, H, E
            for stat in ["R", "H", "E"]:
                if i < len(boxscore[stat]):
                    row.append(boxscore[stat][i])
                else:
                    row.append('')
            table_data.append(row)
        
        # Create table
        table = plt.table(cellText=table_data[1:],
                         colLabels=table_data[0],
                         loc='center',
                         cellLoc='center',
                         colLoc='center')
        
        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        
        # Highlight winner
        winner_idx = 0 if boxscore['R'][0] > boxscore['R'][1] else 1
        for j in range(len(header)):
            cell = table[(winner_idx+1, j)]
            cell.set_facecolor('#e6ffe6')  # Light green
        
        # Hide axes
        plt.axis('off')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(Path(output_dir) / "boxscore.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Boxscore visualization saved to {Path(output_dir) / 'boxscore.png'}")

# Main execution
def main():
    try:
        # Initialize simulation
        simulation = MLBGameSimulation(input_file)
        
        # Select teams
        if len(simulation.teams) >= 2:
            # Random team selection
            home_team, away_team = random.sample(simulation.teams, 2)
        else:
            # Use placeholder teams
            home_team, away_team = "Home Team", "Away Team"
        
        # Display available teams and let user choose
        print("\nAvailable teams:")
        for i, team in enumerate(simulation.teams):
            print(f"{i+1}. {team}")
        
        print(f"\nRandomly selected: {away_team} @ {home_team}")
        print("Using these teams for simulation. To choose specific teams, modify the code.")
        
        # Simulate the game
        game_result = simulation.simulate_game(home_team, away_team)
        
        print("\nSimulation complete!")
        print(f"Game log and boxscore saved to: {output_dir}")
        
    except Exception as e:
        print(f"Error in simulation: {e}")

if __name__ == "__main__":
    main()