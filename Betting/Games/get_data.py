import requests
import json
import pandas as pd


def fetch_and_save_nba_data(api_key, host, season, output_path):
    url = "https://api-nba-v1.p.rapidapi.com/games"
    querystring = {"season": season}

    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": host
    }

    response = requests.get(url, headers=headers, params=querystring)

    if response.status_code == 200:
        data = response.json()
        df = pd.json_normalize(data['response'])

        # Select desired columns
        df = df[['date.start', 'status.long', 'arena.city', 'teams.visitors.code', 'teams.home.code',
                 'scores.visitors.points', 'scores.home.points']]

        # Save the DataFrame to CSV
        df.to_csv(output_path, index=False)
        print(f"Data saved to {output_path}")
    else:
        print(f"Failed to fetch data. Status code: {response.status_code}")


# Example usage
api_key = "a22eff4319msh7f19aefa5ca6675p1153b3jsn08075099bd47"
host = "api-nba-v1.p.rapidapi.com"
season = "2024"
output_path = r"C:\Users\Vince\Downloads\Python\Betting\Games\data.csv"