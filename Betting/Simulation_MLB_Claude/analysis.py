import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Input file path
input_file = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\mlb_atbat_data_filtered.parquet"
output_dir = r"C:\Users\Vince\master\Betting\Simulation_MLB_Claude\analysis_results"

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Function to save figures
def save_fig(fig, filename):
    filepath = Path(output_dir) / filename
    fig.savefig(filepath, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"Saved figure: {filepath}")

try:
    print("Loading Parquet file...")
    df = pd.read_parquet(input_file)
    
    # 1. DATA STRUCTURE EXAMINATION
    print("\n==== DATA STRUCTURE ====")
    print(f"Number of at-bats: {df.shape[0]:,}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Store column information
    with open(Path(output_dir) / "column_info.txt", "w") as f:
        f.write("COLUMN INFORMATION:\n")
        f.write(f"Total records: {df.shape[0]:,}\n")
        f.write(f"Total columns: {df.shape[1]}\n\n")
        f.write("Column names and data types:\n")
        f.write(str(df.dtypes))
        
        # Sample data
        f.write("\n\nSample data (first 10 rows):\n")
        f.write(df.head(10).to_string())
    
    print("Column information saved to column_info.txt")
    
    # 2. STATISTICAL ANALYSIS
    print("\n==== STATISTICAL ANALYSIS ====")
    
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    
    print(f"Numeric columns: {len(numeric_cols)}")
    print(f"Categorical columns: {len(categorical_cols)}")
    
    # Basic statistics
    stats_file = Path(output_dir) / "basic_statistics.csv"
    df[numeric_cols].describe().to_csv(stats_file)
    print(f"Basic statistics saved to {stats_file}")
    
    # Assuming baseball-specific columns (modify as needed)
    if 'batter_id' in df.columns and 'hit' in df.columns:
        # Batting average by player (top 20)
        batting_avg = df.groupby('batter_id')['hit'].agg(['count', 'sum'])
        batting_avg['avg'] = batting_avg['sum'] / batting_avg['count']
        batting_avg = batting_avg.sort_values('avg', ascending=False)
        batting_avg = batting_avg[batting_avg['count'] >= 100]  # Min 100 at-bats
        
        top_batters = batting_avg.head(20)
        batting_file = Path(output_dir) / "top_batters.csv"
        top_batters.to_csv(batting_file)
        print(f"Top batters saved to {batting_file}")
        
        # Visualization of top batters
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x=top_batters.index, y=top_batters['avg'], ax=ax)
        ax.set_title('Top 20 Batters by Average (min 100 at-bats)')
        ax.set_xlabel('Batter ID')
        ax.set_ylabel('Batting Average')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
        save_fig(fig, "top_batters.png")
    
    # Team statistics if team columns exist
    if 'batter_team' in df.columns:
        # Team performance
        team_stats = df.groupby('batter_team').agg({
            'hit': ['count', 'sum', 'mean'],
            'home_run': ['sum'] if 'home_run' in df.columns else [],
            'runs_scored': ['sum'] if 'runs_scored' in df.columns else []
        })
        
        team_file = Path(output_dir) / "team_stats.csv"
        team_stats.to_csv(team_file)
        print(f"Team statistics saved to {team_file}")
        
        # Team performance visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        team_batting = team_stats[('hit', 'mean')].sort_values(ascending=False)
        sns.barplot(x=team_batting.index, y=team_batting.values, ax=ax)
        ax.set_title('Team Batting Averages')
        ax.set_xlabel('Team')
        ax.set_ylabel('Batting Average')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        save_fig(fig, "team_batting_avg.png")
    
    # 3. ADVANCED VISUALIZATIONS
    print("\n==== CREATING VISUALIZATIONS ====")
    
    # Correlation heatmap for numeric variables
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(14, 12))
        correlation = df[numeric_cols].corr()
        mask = np.triu(correlation)
        sns.heatmap(correlation, mask=mask, annot=True, cmap='coolwarm', 
                   linewidths=0.5, ax=ax)
        ax.set_title('Correlation between Numeric Variables')
        save_fig(fig, "correlation_heatmap.png")
    
    # Distribution plots for key metrics
    key_metrics = [col for col in ['hit', 'home_run', 'runs_scored', 'rbi'] 
                  if col in df.columns]
    
    if key_metrics:
        fig, axes = plt.subplots(len(key_metrics), 1, figsize=(12, 4*len(key_metrics)))
        if len(key_metrics) == 1:
            axes = [axes]
            
        for i, metric in enumerate(key_metrics):
            sns.histplot(df[metric], kde=True, ax=axes[i])
            axes[i].set_title(f'Distribution of {metric}')
            
        plt.tight_layout()
        save_fig(fig, "key_metrics_distribution.png")
    
    # 4. SPECIFIC INSIGHTS
    print("\n==== EXTRACTING INSIGHTS ====")
    
    insights_file = Path(output_dir) / "mlb_insights.txt"
    with open(insights_file, "w") as f:
        # Count total records
        f.write(f"Total at-bats analyzed: {df.shape[0]:,}\n\n")
        
        # Most common outcomes if outcome column exists
        if 'event' in df.columns:
            f.write("MOST COMMON AT-BAT OUTCOMES:\n")
            outcomes = df['event'].value_counts()
            for outcome, count in outcomes.head(10).items():
                f.write(f"- {outcome}: {count:,} ({count/len(df)*100:.2f}%)\n")
            f.write("\n")
        
        # Home vs Away performance
        if 'is_home_team' in df.columns and 'hit' in df.columns:
            home = df[df['is_home_team'] == True]['hit'].mean()
            away = df[df['is_home_team'] == False]['hit'].mean()
            f.write(f"HOME vs AWAY BATTING AVERAGE:\n")
            f.write(f"- Home: {home:.3f}\n")
            f.write(f"- Away: {away:.3f}\n")
            f.write(f"- Difference: {(home-away):.3f}\n\n")
        
        # Pitcher analysis if pitcher columns exist
        if 'pitcher_id' in df.columns and 'hit' in df.columns:
            pitchers = df.groupby('pitcher_id')['hit'].agg(['count', 'mean'])
            pitchers = pitchers[pitchers['count'] >= 100]  # Min 100 batters faced
            pitchers = pitchers.sort_values('mean')
            
            f.write("TOP 10 PITCHERS (lowest batting average against, min 100 batters):\n")
            for idx, row in pitchers.head(10).iterrows():
                f.write(f"- Pitcher {idx}: {row['mean']:.3f} avg against ({row['count']} batters faced)\n")
            f.write("\n")
        
        # Game situation analysis
        if 'inning' in df.columns and 'hit' in df.columns:
            inning_stats = df.groupby('inning')['hit'].mean()
            f.write("BATTING AVERAGE BY INNING:\n")
            for inning, avg in inning_stats.items():
                f.write(f"- Inning {inning}: {avg:.3f}\n")
            f.write("\n")
            
        if 'count_balls' in df.columns and 'count_strikes' in df.columns and 'hit' in df.columns:
            # Create count combinations
            df['count'] = df['count_balls'].astype(str) + '-' + df['count_strikes'].astype(str)
            count_stats = df.groupby('count')['hit'].agg(['count', 'mean'])
            count_stats = count_stats.sort_values('count', ascending=False)
            count_stats = count_stats[count_stats['count'] >= 100]
            
            f.write("BATTING AVERAGE BY COUNT (min 100 at-bats):\n")
            for count, row in count_stats.iterrows():
                f.write(f"- {count} count: {row['mean']:.3f} ({row['count']:,} at-bats)\n")
    
    print(f"Insights saved to {insights_file}")
    print("\nAnalysis complete! Results saved to: {output_dir}")

except Exception as e:
    print(f"Error during analysis: {e}")