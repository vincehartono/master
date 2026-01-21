import datetime
import pandas as pd
import os
import statsapi

def get_today():
    return datetime.datetime.now().date()

def get_today_games():
    today = get_today()
    today_str = today.strftime("%Y-%m-%d")
    schedule = statsapi.schedule(start_date=today_str, end_date=today_str)

    games = []
    for game in schedule:
        home = game['home_name']
        away = game['away_name']
        game_time = game['game_datetime'].split('T')[-1][:5]
        home_pitcher = game.get('home_probable_pitcher', 'TBD')
        away_pitcher = game.get('away_probable_pitcher', 'TBD')

        games.append({
            'Home': home,
            'Away': away,
            'Time': game_time,
            'Home SP': home_pitcher,
            'Away SP': away_pitcher
        })

    games_df = pd.DataFrame(games)
    if games_df.empty:
        return games_df

    games_df = games_df.sort_values(by='Time')
    return games_df

if __name__ == "__main__":
    games_df = get_today_games()
    if games_df.empty:
        print("No games found today.")
    else:
        print("Today's Games:")
        print(games_df)

        # Save today's schedule to parquet
        save_path = r"C:\\Users\\Vince\\master\\Betting\\Simulation_MLB\\todays_schedule.parquet"
        games_df.to_parquet(save_path, index=False)
        print(f"\nSaved today's schedule to {save_path}")
