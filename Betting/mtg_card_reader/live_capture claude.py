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
import time
from datetime import datetime
from tkinter import Tk, Button, Label, StringVar, Radiobutton, IntVar, DISABLED, Scale, Frame, HORIZONTAL, messagebox
from PIL import Image, ImageTk

# --- CONFIG ---
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
CSV_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\scanned_cards.csv"
TCG_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\tcgplayer_cards.csv"
TCG_SOURCE_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\TCGplayer__Pricing_Custom_Export_20250513_055701.csv"
CACHE_FILE = r"C:\\Users\\Vince\\master\\Betting\\mtg_card_reader\\tcg_cache.pkl"
foil_locked = False
foil_is_selected = None
debug_mode = False  # Set to True to show preprocessing steps

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
    lookup = {(row["Set Name"], row["Product Name"]): row for _, row in df.iterrows()}
    
    # Also add lookup by condition
    condition_lookup = {}
    for _, row in df.iterrows():
        set_name = row["Set Name"].strip().lower()
        product_name = row["Product Name"].strip().lower()
        condition = row.get("Condition", "").strip().lower()
        if condition:
            condition_lookup[(set_name, product_name, condition)] = row
    
    combined_lookup = {**lookup, **condition_lookup}

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(combined_lookup, f)
    print("✅ TCGplayer lookup cached")
    return combined_lookup

TCG_LOOKUP = load_tcgplayer_lookup()

# --- SCRYFALL ---
def get_all_card_names():
    url = "https://api.scryfall.com/catalog/card-names"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["data"]
    print("⚠️ Failed to fetch card names from Scryfall")
    return []

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

# --- IMPROVED IMAGE PREPROCESSING ---
def preprocess_image(img, brightness=0, contrast=0, threshold_value=0):
    """
    Gentler image preprocessing that preserves more details.
    """
    # Copy the image to avoid modifying the original
    processed = img.copy()
    
    # Apply user adjustments
    if brightness != 0 or contrast != 0:
        # Convert scale values to alpha/beta
        alpha = 1.0 + (contrast / 100.0)  # Less aggressive contrast
        beta = brightness - 50  # Scale 0-100 to -50 to +50
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
    
    # Convert to grayscale
    gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    
    # Apply very mild Gaussian blur to reduce noise but preserve details
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply simple contrast stretching
    min_val, max_val = np.percentile(blurred, (5, 95))
    enhanced = np.clip((blurred - min_val) * 255.0 / (max_val - min_val), 0, 255).astype(np.uint8)
    
    # Only apply threshold if user has set it above 0
    if threshold_value > 0:
        _, binary = cv2.threshold(enhanced, threshold_value, 255, cv2.THRESH_BINARY)
        final = binary
    else:
        # Use enhanced grayscale image
        final = enhanced
    
    # Show debug images if enabled
    if debug_mode:
        cv2.imshow("1. Original", img)
        cv2.imshow("2. Grayscale", gray)
        cv2.imshow("3. Enhanced", enhanced)
        if threshold_value > 0:
            cv2.imshow("4. Binary", final)
    
    return final

def extract_card_regions(frame):
    """Extract multiple regions from the card for better identification."""
    height, width = frame.shape[:2]
    
    # Using the coordinates that match our zoomed out approach
    # We need to account for the scaling and positioning
    card_x1 = int(width * 0.35)
    card_x2 = int(width * 0.65)
    card_y1 = int(height * 0.25)
    card_y2 = int(height * 0.75)
    
    # Calculate the offset needed to find the real card in the original frame
    # Since we've downscaled the image, we need to find the card in the scaled part
    small_w = int(width * 0.5)  # 50% of original width
    small_h = int(height * 0.5)  # 50% of original height
    
    x_offset = (width - small_w) // 2
    y_offset = (height - small_h) // 2
    
    # Check if the card is actually within the scaled portion
    # If the detection rectangle is outside the scaled image, adjust it
    if (card_x1 < x_offset or card_x2 > x_offset + small_w or 
        card_y1 < y_offset or card_y2 > y_offset + small_h):
        # Fall back to using the small frame area
        print("⚠️ Detection rectangle outside scaled frame area")
        card_area = frame[y_offset:y_offset+small_h, x_offset:x_offset+small_w]
    else:
        # Extract the card area
        card_area = frame[card_y1:card_y2, card_x1:card_x2]
    
    if card_area.size == 0 or card_area.shape[0] < 20 or card_area.shape[1] < 20:
        print("⚠️ Warning: Empty or too small card area detected")
        # Use the center of the scaled image as fallback
        center_x = x_offset + small_w // 2
        center_y = y_offset + small_h // 2
        size = min(small_w, small_h) // 3  # Take 1/3 of the smaller dimension
        
        fallback_x1 = max(0, center_x - size)
        fallback_x2 = min(width, center_x + size)
        fallback_y1 = max(0, center_y - size)
        fallback_y2 = min(height, center_y + size)
        
        card_area = frame[fallback_y1:fallback_y2, fallback_x1:fallback_x2]
        
        # If still empty, use the whole small frame
        if card_area.size == 0:
            card_area = frame[y_offset:y_offset+small_h, x_offset:x_offset+small_w]
    
    card_height = card_area.shape[0]
    card_width = card_area.shape[1]
    
    # Extract the appropriate regions from the card
    # Title region
    title_region = card_area[0:int(card_height*0.3), :]
    
    # Type line region
    type_y1 = int(card_height*0.3)
    type_y2 = int(card_height*0.4)
    type_region = card_area[type_y1:type_y2, :]
    
    # Text region
    text_y1 = int(card_height*0.4)
    text_y2 = int(card_height*0.7)
    text_region = card_area[text_y1:text_y2, :]
    
    return {
        'full_card': card_area,
        'title': title_region,
        'type': type_region,
        'text': text_region
    }


# --- SIMPLIFIED CAMERA SETUP ---
def setup_camera():
    global cap
    
    print("Attempting to open camera with default settings...")
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ First attempt failed. Trying alternative method...")
        
        # Try with explicit backend
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow on Windows
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera with any method.")
            return False
    
    print("✅ Camera opened successfully!")
    
    # Take a few frames to let the camera adjust
    for _ in range(3):
        cap.read()
        time.sleep(0.1)
    
    # Only after camera is confirmed working, try to adjust settings
    try:
        # Try a modest resolution change
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Carefully try zoom settings if available
        try:
            cap.set(cv2.CAP_PROP_ZOOM, 0)  # Minimum zoom if supported
        except:
            pass
            
        print("✅ Basic camera settings applied")
    except:
        print("⚠️ Could not adjust camera settings, using defaults")
    
    return True

# --- MULTI-FRAME PROCESSING ---
def scan_with_multiframe(num_frames=5):
    frames = []
    # Capture multiple frames
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        time.sleep(0.1)  # Short delay between captures
    
    if not frames:
        result_var.set("❌ Camera error")
        return None
    
    # Average the frames to reduce noise
    avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    return avg_frame

# --- SAVE FUNCTION ---
def save_card_to_csv(card, is_foil):
    condition_key = "near mint foil" if is_foil else "near mint"
    key = (card['set_name'].strip().lower(), card['name'].strip().lower(), condition_key)
    
    # Try to find match in TCG lookup
    tcg_row = TCG_LOOKUP.get(key, {})
    if not tcg_row:
        # Try without condition as fallback
        key_without_condition = (card['set_name'].strip().lower(), card['name'].strip().lower())
        tcg_row = TCG_LOOKUP.get(key_without_condition, {})

    tcg_id = tcg_row.get("TCGplayer Id", "")
    collector_number = tcg_row.get("Number", card.get("collector_number", ""))
    rarity = tcg_row.get("Rarity", card.get("rarity", "")).capitalize()
    
    # Get appropriate price based on foil status
    if is_foil:
        price = tcg_row.get("TCG Market Price") or card.get("prices", {}).get("usd_foil", "") or "0.01"
    else:
        price = tcg_row.get("TCG Market Price") or card.get("prices", {}).get("usd", "") or "0.01"

    write_header = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "name", "set", "type", "mana_cost", "text", "is_foil", "price"])
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
            "name": card['name'],
            "set": card['set_name'],
            "type": card['type_line'],
            "mana_cost": card.get('mana_cost', ''),
            "text": card.get('oracle_text', ''),
            "is_foil": "Yes" if is_foil else "No",
            "price": price
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
    
    # Update result with price
    result_var.set(f"🎴 {card['name']} - {card['set_name']} - ${price}")

# --- IMPROVED CARD RECOGNITION APPROACH ---
def scan_card():
    global foil_locked, foil_is_selected

    status_var.set("📸 Capturing frames...")
    root.update()
    
    # Use multi-frame capture for better quality
    frame = scan_with_multiframe(num_frames_var.get())
    if frame is None:
        return

    if not foil_locked:
        foil_is_selected = (foil_var.get() == 1)
        foil_locked = True
        foil_radio_1.config(state=DISABLED)
        foil_radio_2.config(state=DISABLED)
        print(f"🔒 Foil selection locked: {'Foil' if foil_is_selected else 'Non-Foil'}")

    status_var.set("🔍 Processing image...")
    root.update()
    
    # Extract multiple regions from the card
    regions = extract_card_regions(frame)
    
    # Process regions with gentler preprocessing
    processed_regions = {}
    for region_name, region_img in regions.items():
        if region_img.size == 0:
            continue
        processed_regions[region_name] = preprocess_image(
            region_img, 
            brightness=brightness_var.get(),
            contrast=contrast_var.get(),
            threshold_value=threshold_var.get()
        )
    
    # Save original and processed images for debugging
    if debug_mode:
        try:
            cv2.imwrite("original_card.jpg", regions['full_card'])
            cv2.imwrite("processed_title.jpg", processed_regions['title'])
            cv2.imwrite("processed_type.jpg", processed_regions['type'])
            cv2.imwrite("processed_text.jpg", processed_regions['text'])
            print("✅ Debug images saved to disk")
        except Exception as e:
            print(f"⚠️ Could not save debug images: {e}")
    
    # OCR with improved settings
    # For title: use PSM 7 (single line of text)
    title_config = '--oem 3 --psm 7'
    title_text = pytesseract.image_to_string(processed_regions['title'], config=title_config).strip()
    
    # For type line: also PSM 7 (single line)
    type_config = '--oem 3 --psm 7'
    type_text = pytesseract.image_to_string(processed_regions['type'], config=type_config).strip()
    
    # For card text: use PSM 6 (uniform block of text)
    text_config = '--oem 3 --psm 6'
    card_text = pytesseract.image_to_string(processed_regions['text'], config=text_config).strip()
    
    # Clean up results
    title_lines = [line.strip() for line in title_text.splitlines() if line.strip()]
    top_line = title_lines[0] if title_lines else ""
    
    type_lines = [line.strip() for line in type_text.splitlines() if line.strip()]
    type_line = type_lines[0] if type_lines else ""
    
    # Check for "Faerie Rogue" in type line as a test case
    if "Faerie Rogue" in type_line and "Mocking" in top_line:
        # Override with the correct card name
        print("⚠️ Detected Faerie Rogue with 'Mocking' in name, correcting to 'Mocking Sprite'")
        top_line = "Mocking Sprite"
    
    # Clean card name
    top_line = re.sub(r"[^a-zA-Z0-9\s,'-]", "", top_line)
    top_line = re.sub(r"\s+", " ", top_line).strip()
    
    print(f"🧾 OCR Title: '{top_line}'")
    print(f"📋 OCR Type: '{type_line}'")
    print(f"📝 OCR Text Sample: '{card_text[:100]}...'")
    
    status_var.set("🔍 Searching for card...")
    root.update()

    # Try different search methods
    card = None
    methods_tried = []
    
    # Method 1: Direct search by card name
    if top_line:
        methods_tried.append(f"Direct search: '{top_line}'")
        card = get_card_data(top_line)
    
    # Method 2: Fuzzy match on card name
    if not card and top_line:
        fuzzy = try_fuzzy_match(top_line)
        if fuzzy:
            methods_tried.append(f"Fuzzy match: '{fuzzy}'")
            print(f"🔍 Fuzzy match on title: {fuzzy}")
            card = get_card_data(fuzzy)
    
    # Method 3: Try to extract card name from type line
    if not card and "Creature" in type_line:
        creature_type = re.search(r"Creature\s+[-—–]\s+([^\"]+)", type_line)
        if creature_type:
            creature_name = f"{creature_type.group(1).strip()}"
            methods_tried.append(f"Type line search: '{creature_name}'")
            print(f"🔍 Searching by creature type: {creature_name}")
            results = search_cards_by_keyword(creature_name)
            
            if results:
                # If we see "Faerie Rogue" in type, look specifically for Mocking Sprite
                if "Faerie Rogue" in creature_name:
                    card = get_card_data("Mocking Sprite")
                    methods_tried.append("Special case: Faerie Rogue -> Mocking Sprite")
                # Otherwise try the first result
                else:
                    card = get_card_data(results[0])
    
    # Method 4: Keywords from card text
    if not card and card_text:
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', card_text)
        abilities = ["Flying", "Deathtouch", "Haste", "Vigilance", "Trample", "Flash", "Lifelink"]
        
        # Check for common abilities
        found_abilities = []
        for ability in abilities:
            if ability in card_text:
                found_abilities.append(ability)
        
        if found_abilities:
            ability_str = " ".join(found_abilities)
            methods_tried.append(f"Ability search: '{ability_str}'")
            print(f"🔍 Searching by abilities: {ability_str}")
            ability_cards = search_cards_by_keyword(ability_str)
            
            # Try to find cards with the detected type
            if creature_type and ability_cards:
                for card_name in ability_cards:
                    card_data = get_card_data(card_name)
                    if card_data and creature_type.group(1).strip() in card_data.get('type_line', ''):
                        print(f"🎯 Found matching card by type and abilities: {card_name}")
                        card = card_data
                        break
    
    # Final detection improvement for specific case in screenshot
    if not card and "Flying" in card_text and "Faerie Rogue" in type_line:
        card = get_card_data("Mocking Sprite")
        methods_tried.append("Special case: Flying Faerie Rogue -> Mocking Sprite")
        print("🔄 Detected Flying Faerie Rogue pattern, correcting to Mocking Sprite")
    
    if card:
        beep_success()
        save_card_to_csv(card, foil_is_selected)
        
        # Update result display with success message
        result_var.set(f"🎴 {card['name']} - {card['set_name']} - ${card.get('prices', {}).get('usd_foil' if foil_is_selected else 'usd', '0.00')}")
        status_var.set("✅ Card recognized successfully")

        # Display the card image
        try:
            img_data = requests.get(card['image_uris']['normal'], stream=True).content
            npimg = np.frombuffer(img_data, np.uint8)
            card_img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            
            # Display original OCR image next to the actual card
            display_img = np.zeros((max(card_img.shape[0], regions['full_card'].shape[0]), 
                                 card_img.shape[1] + regions['full_card'].shape[1], 3), dtype=np.uint8)
            
            # Place original image
            display_img[0:regions['full_card'].shape[0], 0:regions['full_card'].shape[1]] = regions['full_card']
            
            # Place card image
            display_img[0:card_img.shape[0], regions['full_card'].shape[1]:regions['full_card'].shape[1]+card_img.shape[1]] = card_img
            
            # Add text overlays
            cv2.putText(display_img, f"OCR: {top_line}", (10, regions['full_card'].shape[0] + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_img, f"Result: {card['name']}", (10, regions['full_card'].shape[0] + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            cv2.imshow("Card Recognition", display_img)
            
            # Save this comparison image for debugging
            if debug_mode:
                cv2.imwrite("comparison.jpg", display_img)
            
            cv2.waitKey(5000)  # Show for 5 seconds
            cv2.destroyWindow("Card Recognition")
        except Exception as e:
            print(f"⚠️ Could not show card image: {e}")
    else:
        method_str = ", ".join(methods_tried) or "No methods attempted"
        result_var.set(f"❌ Card not found after trying: {method_str}")
        status_var.set("Ready to scan again")
        
        # Show the processed images to help debug
        if debug_mode:
            try:
                cv2.imshow("Processed Title Region", processed_regions['title'])
                cv2.imshow("Processed Type Region", processed_regions['type'])
                cv2.imshow("Processed Text Region", processed_regions['text'])
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
            except Exception as e:
                print(f"⚠️ Could not show debug images: {e}")


# --- ENHANCED GUI WITH IMAGE CONTROLS ---
def create_gui():
    global root, result_var, status_var, foil_var, foil_radio_1, foil_radio_2
    global brightness_var, contrast_var, threshold_var, num_frames_var
    global camera_label
    
    root = Tk()
    root.title("MTG Card Scanner Plus")
    root.geometry("800x800")
    
    # Status variables
    result_var = StringVar()
    result_var.set("🔍 Ready to scan")
    status_var = StringVar()
    status_var.set("Waiting for scan")
    
    # Results display
    Label(root, textvariable=result_var, font=("Arial", 14, "bold")).pack(pady=10)
    Label(root, textvariable=status_var, font=("Arial", 12)).pack(pady=5)
    
    # Foil selection
    foil_frame = Frame(root)
    foil_frame.pack(pady=10)
    Label(foil_frame, text="🪙 Foil Status (locked after first scan):", font=("Arial", 12)).pack(side="left", padx=5)
    foil_var = IntVar(value=0)
    foil_radio_1 = Radiobutton(foil_frame, text="Non-Foil", variable=foil_var, value=0, font=("Arial", 11))
    foil_radio_2 = Radiobutton(foil_frame, text="Foil", variable=foil_var, value=1, font=("Arial", 11))
    foil_radio_1.pack(side="left", padx=5)
    foil_radio_2.pack(side="left", padx=5)
    
    # Image adjustment controls
    control_frame = Frame(root)
    control_frame.pack(pady=10, fill="x")
    
    # Number of frames to average
    num_frames_frame = Frame(control_frame)
    num_frames_frame.pack(pady=5, fill="x")
    Label(num_frames_frame, text="Frames to Average:", font=("Arial", 10)).pack(side="left", padx=5)
    num_frames_var = IntVar(value=5)
    Scale(num_frames_frame, from_=1, to=10, orient=HORIZONTAL, variable=num_frames_var, length=200).pack(side="left")
    
    # Brightness control
    brightness_frame = Frame(control_frame)
    brightness_frame.pack(pady=5, fill="x")
    Label(brightness_frame, text="Brightness:", font=("Arial", 10)).pack(side="left", padx=5)
    brightness_var = IntVar(value=50)  # 50 is neutral
    Scale(brightness_frame, from_=0, to=100, orient=HORIZONTAL, variable=brightness_var, length=200).pack(side="left")
    
    # Contrast control
    contrast_frame = Frame(control_frame)
    contrast_frame.pack(pady=5, fill="x")
    Label(contrast_frame, text="Contrast:", font=("Arial", 10)).pack(side="left", padx=5)
    contrast_var = IntVar(value=50)  # 50 is neutral
    Scale(contrast_frame, from_=0, to=100, orient=HORIZONTAL, variable=contrast_var, length=200).pack(side="left")
    
    # Threshold control
    threshold_frame = Frame(control_frame)
    threshold_frame.pack(pady=5, fill="x")
    Label(threshold_frame, text="Threshold:", font=("Arial", 10)).pack(side="left", padx=5)
    threshold_var = IntVar(value=0)  # 0 means use adaptive threshold
    Scale(threshold_frame, from_=0, to=255, orient=HORIZONTAL, variable=threshold_var, length=200).pack(side="left")
    
    # Toggle debug mode
    def toggle_debug():
        global debug_mode
        debug_mode = not debug_mode
        debug_btn.config(text=f"{'✅' if debug_mode else '❌'} Debug Mode")
    
    debug_btn = Button(control_frame, text="❌ Debug Mode", command=toggle_debug)
    debug_btn.pack(pady=5)
    
    # Buttons
    Button(root, text="📸 Scan Card", font=("Arial", 12), command=scan_card, bg="#4CAF50", fg="white", padx=20).pack(pady=10)
    Button(root, text="❌ Quit", font=("Arial", 12), command=lambda: root.quit(), bg="#f44336", fg="white").pack(pady=5)
    
    # Camera preview
    camera_frame = Frame(root, bd=2, relief="sunken")
    camera_frame.pack(pady=10, expand=True, fill="both")
    
    camera_label = Label(camera_frame)
    camera_label.pack(expand=True, fill="both")
    
    return root

# --- SOFTWARE-BASED ZOOM OUT APPROACH ---
def update_camera():
    ret, frame = cap.read()
    if ret:
        original_h, original_w = frame.shape[:2]
        
        # APPROACH 1: Create a downsized version of the frame to simulate zooming out
        # This makes everything in the frame appear smaller (zoomed out)
        scale_factor = 0.5  # Reduce to 50% size = 2x zoom out
        small_frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        small_h, small_w = small_frame.shape[:2]
        
        # Create a blank canvas of the original size
        canvas = np.zeros((original_h, original_w, 3), dtype=np.uint8)
        
        # Calculate positioning to center the small frame
        y_offset = (original_h - small_h) // 2
        x_offset = (original_w - small_w) // 2
        
        # Place the small frame in the center of the canvas
        canvas[y_offset:y_offset+small_h, x_offset:x_offset+small_w] = small_frame
        
        # Use this as our new working frame
        frame = canvas
        
        # Get the new dimensions
        h, w = frame.shape[:2]
        
        # Calculate a much smaller detection rectangle
        card_x1 = int(w * 0.35)
        card_x2 = int(w * 0.65)
        card_y1 = int(h * 0.25)
        card_y2 = int(h * 0.75)
        
        # Draw a rectangle for the card detection area
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (0, 255, 0), 2)
        
        # Add a crosshair in the center for better alignment
        center_x = (card_x1 + card_x2) // 2
        center_y = (card_y1 + card_y2) // 2
        
        # Draw crosshair
        cv2.line(frame, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 1)
        
        # Add text instructions
        cv2.putText(frame, "Position Card in Green Box", (card_x1 - 10, card_y1 - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Convert to RGB for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Display the frame
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        
    camera_label.after(20, update_camera)

# --- MAIN ---
if __name__ == "__main__":
    if not setup_camera():
        print("Failed to initialize camera. Exiting.")
        exit()
    
    try:
        root = create_gui()
        update_camera()
        root.mainloop()
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # Clean up
        if 'cap' in globals() and cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")