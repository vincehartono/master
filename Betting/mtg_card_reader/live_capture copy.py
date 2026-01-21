import cv2
import pytesseract
import requests
import pandas as pd
import pickle
from playsound import playsound
import numpy as np
from difflib import get_close_matches
import re
import os
import csv
from datetime import datetime
from tkinter import Tk, Button, Label, StringVar, Radiobutton, IntVar, DISABLED
from PIL import Image, ImageTk

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
CSV_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\scanned_cards.csv"
TCG_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\tcgplayer_cards.csv"
TCG_SOURCE_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\TCGplayer__Pricing_Custom_Export_20250513_055701.csv"
CACHE_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\tcg_cache.pkl"
foil_locked = False
foil_is_selected = None

# --- LOAD LOCAL TCG CSV FOR LOOKUP (cached) ---
def load_tcgplayer_lookup():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            print("✅ Loaded cached TCGplayer data")
            return pickle.load(f)

    print("🔄 Building TCGplayer lookup from CSV...")
    df = pd.read_csv(TCG_SOURCE_FILE, dtype=str, encoding="utf-8-sig", on_bad_lines='skip')
    df.fillna("", inplace=True)
    df.columns = df.columns.str.strip()
    df["Set Name"] = df["Set Name"].str.strip().str.lower()
    df["Product Name"] = df["Product Name"].str.strip().str.lower()
    df["Condition"] = df["Condition"].str.strip().str.lower()
    lookup = {(row["Set Name"], row["Product Name"], row["Condition"]): row for _, row in df.iterrows()}

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(lookup, f)
    print("✅ TCGplayer lookup cached")
    return lookup

TCG_LOOKUP = load_tcgplayer_lookup()

# --- SCRYFALL ---
def get_all_card_names():
    url = "https://api.scryfall.com/catalog/card-names"
    response = requests.get(url)
    return response.json()["data"]

CARD_NAME_LIST = get_all_card_names()

def get_card_data(card_name):
    exact_url = f"https://api.scryfall.com/cards/named?exact={card_name}"
    r = requests.get(exact_url)
    if r.status_code == 200:
        return r.json()
    fuzzy_url = f"https://api.scryfall.com/cards/named?fuzzy={card_name}"
    r = requests.get(fuzzy_url)
    return r.json() if r.status_code == 200 else None

def try_fuzzy_match(text):
    matches = get_close_matches(text, CARD_NAME_LIST, n=1, cutoff=0.5)
    return matches[0] if matches else None

def beep_success():
    try:
        playsound('C:\\Windows\\Media\\tada.wav')
    except:
        print("🔇 Beep failed")

def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contrast = cv2.convertScaleAbs(gray, alpha=1.8, beta=-30)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(contrast, -1, kernel)
    return sharp

# --- SAVE FUNCTION ---
def save_card_to_csv(card, is_foil):
    condition_key = "near mint foil" if is_foil else "near mint"
    key = (card['set_name'].strip().lower(), card['name'].strip().lower(), condition_key)
    tcg_row = TCG_LOOKUP.get(key, {})

    tcg_id = tcg_row.get("TCGplayer Id", "")
    collector_number = tcg_row.get("Number", card.get("collector_number", ""))
    rarity = tcg_row.get("Rarity", card.get("rarity", "")).capitalize()
    price = tcg_row.get("TCG Market Price") or card.get("prices", {}).get("usd_foil" if is_foil else "usd", "") or "0.01"

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "name", "set", "type", "mana_cost", "text"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "name": card['name'],
            "set": card['set_name'],
            "type": card['type_line'],
            "mana_cost": card.get('mana_cost', ''),
            "text": card.get('oracle_text', '')
        })

    write_tcg_header = not os.path.exists(TCG_FILE)
    with open(TCG_FILE, "a", newline="", encoding="utf-8") as f2:
        fieldnames = [
            "TCGplayer Id", "Product Line", "Set Name", "Product Name", "Title", "Number", "Rarity",
            "Condition", "Language", "TCG Market Price", "TCG Direct Low",
            "TCG Low Price With Shipping", "TCG Low Price", "Total Quantity",
            "Add to Quantity", "TCG Marketplace Price"
        ]
        writer = csv.DictWriter(f2, fieldnames=fieldnames)
        if write_tcg_header:
            writer.writeheader()
        writer.writerow({
            "TCGplayer Id": tcg_id,
            "Product Line": "Magic",
            "Set Name": card['set_name'],
            "Product Name": card['name'],
            "Title": card['name'],
            "Number": collector_number,
            "Rarity": rarity,
            "Condition": "Near Mint Foil" if is_foil else "Near Mint",
            "Language": "English",
            "TCG Market Price": price,
            "TCG Direct Low": "",
            "TCG Low Price With Shipping": "",
            "TCG Low Price": "",
            "Total Quantity": 0,
            "Add to Quantity": 1,
            "TCG Marketplace Price": price
        })

    print(f"💾 Saved: {card['name']} | ID: {tcg_id or 'N/A'} | ${price or 'N/A'}")

# --- SCAN CARD ---
def scan_card():
    global foil_locked, foil_is_selected

    ret, frame = cap.read()
    if not ret:
        result_var.set("❌ Camera error")
        return

    if not foil_locked:
        foil_is_selected = (foil_var.get() == 1)
        foil_locked = True
        foil_radio_1.config(state=DISABLED)
        foil_radio_2.config(state=DISABLED)
        print(f"🔒 Foil selection locked: {'Foil' if foil_is_selected else 'Non-Foil'}")

    processed = preprocess_image(frame)
    full_text = pytesseract.image_to_string(processed, config='--psm 6').strip()
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    top_line = lines[0] if lines else ""
    top_line = re.sub(r"[^a-zA-Z0-9\s]", "", top_line)
    top_line = re.sub(r"\s+", " ", top_line).strip()
    print(f"🧾 OCR Top Line Cleaned: '{top_line}'")

    card = get_card_data(top_line)
    if not card:
        fuzzy = try_fuzzy_match(top_line)
        if fuzzy:
            print(f"🔍 Fuzzy match: {fuzzy}")
            card = get_card_data(fuzzy)

    if card:
        beep_success()
        result_var.set(f"🎴 {card['name']} - {card['set_name']}")
        save_card_to_csv(card, foil_is_selected)

        try:
            img_data = requests.get(card['image_uris']['normal'], stream=True).content
            npimg = np.frombuffer(img_data, np.uint8)
            card_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            cv2.imshow("Detected Card", card_img)
            cv2.waitKey(3000)
            cv2.destroyWindow("Detected Card")
        except:
            print("⚠️ Could not show card image.")
    else:
        result_var.set("❌ Card not found")

# --- GUI ---
root = Tk()
root.title("MTG Card Scanner")

result_var = StringVar()
result_var.set("🔍 Ready to scan")

Label(root, textvariable=result_var, font=("Arial", 14)).pack(pady=10)

Label(root, text="🪙 Foil Status (locked after first scan):", font=("Arial", 12)).pack()
foil_var = IntVar(value=0)
foil_radio_1 = Radiobutton(root, text="Non-Foil", variable=foil_var, value=0, font=("Arial", 11))
foil_radio_2 = Radiobutton(root, text="Foil", variable=foil_var, value=1, font=("Arial", 11))
foil_radio_1.pack()
foil_radio_2.pack()

Button(root, text="📸 Scan Card", font=("Arial", 12), command=scan_card).pack(pady=5)
Button(root, text="❌ Quit", font=("Arial", 12), command=lambda: root.quit()).pack(pady=5)

cap = cv2.VideoCapture(0)

def update_camera():
    ret, frame = cap.read()
    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb).resize((480, 360))
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(20, update_camera)

camera_label = Label(root)
camera_label.pack(pady=10)
update_camera()

root.mainloop()
cap.release()
cv2.destroyAllWindows()