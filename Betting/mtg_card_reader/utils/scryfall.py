import requests

def get_card_data(card_name):
    url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None