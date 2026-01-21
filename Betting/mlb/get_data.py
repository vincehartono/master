import statsapi
import pandas as pd

sched = statsapi.schedule(start_date='03/20/2024',end_date='09/29/2024')
# print(sched)

df = pd.json_normalize(sched)

df.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\df.csv")

df = df[['game_date', 'status', 'away_name', 'home_name', 'home_probable_pitcher', 'away_probable_pitcher', 'away_score', 'home_score', 'venue_id']]

df.to_csv(r"C:\Users\Vince\Downloads\Python\Betting\mlb\data.csv")