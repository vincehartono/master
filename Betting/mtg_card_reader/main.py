import os
from utils.ocr import extract_text_from_image
from utils.scryfall import get_card_data

IMAGE_PATH = "card_samples/sample1.jpg"  # Replace with your image

print("🔍 Extracting text...")
card_name = extract_text_from_image(IMAGE_PATH)
print(f"🧾 Detected Card Name: {card_name}")

print("🌐 Querying Scryfall...")
card_data = get_card_data(card_name)

if card_data:
    print(f"\n🎴 {card_data['name']}")
    print(f"Type: {card_data['type_line']}")
    print(f"Mana Cost: {card_data.get('mana_cost', 'N/A')}")
    print(f"Oracle Text: {card_data.get('oracle_text', 'N/A')}")
    print(f"Set: {card_data['set_name']}")
    print(f"Image: {card_data['image_uris']['normal']}")
else:
    print("❌ Card not found.")