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
from tkinter import Tk, Button, Label, StringVar, Radiobutton, IntVar, DISABLED, Scale, Frame, HORIZONTAL
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

# --- ENHANCED IMAGE PROCESSING ---
def preprocess_image(img, brightness=0, contrast=0, threshold_value=0):
    # Copy the image to avoid modifying the original
    processed = img.copy()
    
    # Apply user adjustments
    if brightness != 0 or contrast != 0:
        # Convert scale values to alpha/beta
        alpha = 1.0 + (contrast / 50.0)  # Scale 0-100 to 1.0-3.0
        beta = brightness - 50  # Scale 0-100 to -50 to +50
        processed = cv2.convertScaleAbs(processed, alpha=alpha, beta=beta)
    
    # Apply Gaussian blur to reduce noise before other operations
    blur = cv2.GaussianBlur(processed, (3, 3), 0)
    
    # Convert to grayscale
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    
    # Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharp = cv2.filter2D(equalized, -1, kernel)
    
    # Apply binary threshold if threshold value is set
    if threshold_value > 0:
        _, binary = cv2.threshold(sharp, threshold_value, 255, cv2.THRESH_BINARY)
        final = binary
    else:
        # Use adaptive threshold
        binary = cv2.adaptiveThreshold(sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        final = binary
    
    # Show debug images if enabled
    if debug_mode:
        cv2.imshow("1. Original", img)
        cv2.imshow("2. Adjusted", processed)
        cv2.imshow("3. Grayscale", gray)
        cv2.imshow("4. Equalized", equalized)
        cv2.imshow("5. Sharpened", sharp)
        cv2.imshow("6. Final Binary", final)
    
    return final

def extract_card_regions(frame):
    """Extract multiple regions from the card for better identification."""
    height, width = frame.shape[:2]
    
    # Use the smaller central region for card detection (matching the zoomed out view)
    card_x1 = int(width * 0.3)
    card_x2 = int(width * 0.7)
    card_y1 = int(height * 0.2)
    card_y2 = int(height * 0.8)
    
    # Extract the card area from this smaller central region
    card_area = frame[card_y1:card_y2, card_x1:card_x2]
    
    if card_area.size == 0 or card_area.shape[0] < 20 or card_area.shape[1] < 20:
        print("⚠️ Warning: Empty or too small card area detected")
        return {
            'full_card': frame,  # Fall back to full frame
            'title': frame[0:int(height*0.3), :],
            'type': frame[int(height*0.3):int(height*0.5), :],
            'text': frame[int(height*0.5):int(height*0.8), :]
        }
    
    card_height = card_area.shape[0]
    card_width = card_area.shape[1]
    
    # Extract regions proportionally from the smaller card area
    # Title region (top portion, including the art)
    title_region = card_area[0:int(card_height*0.3), :]
    
    # Type line region (middle part where "Creature - Faerie Rogue" is)
    type_y1 = int(card_height*0.3)
    type_y2 = int(card_height*0.4)
    type_region = card_area[type_y1:type_y2, :]
    
    # Text region (where "Flying" ability text is)
    text_y1 = int(card_height*0.4)
    text_y2 = int(card_height*0.7)
    text_region = card_area[text_y1:text_y2, :]
    
    return {
        'full_card': card_area,
        'title': title_region,
        'type': type_region,
        'text': text_region
    }


# --- CAMERA SETUP WITH ZOOM OUT ATTEMPT ---
def setup_camera():
    global cap
    cap = cv2.VideoCapture(0)
    
    # Try to set a wider field of view with lower resolution
    # This effectively "zooms out" on many webcams
    try:
        # Try lower resolution first - this often gives a wider field of view
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Some cameras support zoom control
        cap.set(cv2.CAP_PROP_ZOOM, 0)  # Minimum zoom (widest view)
        
        # Some cameras support field of view control
        # This is not standard and may not work on all cameras
        cap.set(cv2.CAP_PROP_FOCUS, 0)  # Sometimes affects FOV
        
        print("✅ Attempted to set wider field of view")
    except:
        print("⚠️ Could not adjust camera FOV settings")
    
    # Check if camera opened successfully
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return False
    
    # Take a few frames to let the camera adjust
    for _ in range(3):
        cap.read()
        time.sleep(0.1)
    
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

# --- SCAN FUNCTION WITH FULL CARD ANALYSIS ---
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
    
    # Process regions
    processed_regions = {}
    for region_name, region_img in regions.items():
        processed_regions[region_name] = preprocess_image(
            region_img, 
            brightness=brightness_var.get(),
            contrast=contrast_var.get(),
            threshold_value=threshold_var.get()
        )
    
    # OCR on title region
    title_config = '--oem 3 --psm 6'  # Single line title
    title_text = pytesseract.image_to_string(processed_regions['title'], config=title_config).strip()
    
    # OCR on text region
    text_config = '--oem 3 --psm 6'  # Uniform text block
    card_text = pytesseract.image_to_string(processed_regions['text'], config=text_config).strip()
    
    # Clean up title
    title_lines = [line.strip() for line in title_text.splitlines() if line.strip()]
    top_line = title_lines[0] if title_lines else ""
    top_line = re.sub(r"[^a-zA-Z0-9\s]", "", top_line)
    top_line = re.sub(r"\s+", " ", top_line).strip()
    
    print(f"🧾 OCR Title: '{top_line}'")
    print(f"📝 OCR Text Sample: '{card_text[:100]}...'")
    
    status_var.set("🔍 Searching for card...")
    root.update()

    # First try exact match with title
    card = get_card_data(top_line)
    
    # If no match, try fuzzy match with title
    if not card:
        fuzzy = try_fuzzy_match(top_line)
        if fuzzy:
            print(f"🔍 Fuzzy match on title: {fuzzy}")
            card = get_card_data(fuzzy)
    
    # If still no match, try using keywords from the card text
    if not card and card_text:
        # Extract potential keywords from card text
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', card_text)  # Capitalized words, at least 4 chars
        if words:
            print(f"🔍 Trying text keywords: {', '.join(words[:5])}")
            # Try to find a card with these keywords
            for word in words[:5]:  # Try first 5 keywords
                potential_cards = search_cards_by_keyword(word)
                if potential_cards:
                    print(f"🔍 Found potential match via keyword '{word}': {potential_cards[0]}")
                    card = get_card_data(potential_cards[0])
                    if card:
                        break
    
    if card:
        beep_success()
        save_card_to_csv(card, foil_is_selected)

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
            
            # Add title text overlay
            cv2.putText(display_img, f"OCR: {top_line}", (10, regions['full_card'].shape[0] + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("Card Recognition", display_img)
            cv2.waitKey(5000)  # Show for 5 seconds
            cv2.destroyWindow("Card Recognition")
        except Exception as e:
            print(f"⚠️ Could not show card image: {e}")
            
        status_var.set("✅ Card recognized successfully")
    else:
        result_var.set("❌ Card not found")
        status_var.set("Ready to scan again")
        
        # Show the processed images to help debug
        if debug_mode:
            try:
                cv2.imshow("Processed Title Region", processed_regions['title'])
                cv2.imshow("Processed Text Region", processed_regions['text'])
                cv2.waitKey(3000)
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

# --- CAMERA PREVIEW WITH ZOOMED OUT VIEW ---
def update_camera():
    ret, frame = cap.read()
    if ret:
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Calculate a smaller central area (making everything appear "zoomed out")
        # Use a much smaller rectangle in the center for card detection
        card_x1 = int(w * 0.3)
        card_x2 = int(w * 0.7)
        card_y1 = int(h * 0.2)
        card_y2 = int(h * 0.8)
        
        # Draw a rectangle for the card detection area
        cv2.rectangle(frame, (card_x1, card_y1), (card_x2, card_y2), (0, 255, 0), 2)
        
        # Add a reticle in the center for better alignment
        center_x = (card_x1 + card_x2) // 2
        center_y = (card_y1 + card_y2) // 2
        
        # Draw crosshair
        cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 255, 0), 1)
        cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 255, 0), 1)
        
        # Convert to RGB for tkinter
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Simple visual "zoom out" effect by scaling the image smaller and embedding it in a black frame
        # This creates a visual perception of a wider field of view
        scale = 0.8  # Scale to 80% of original size
        small_frame = cv2.resize(frame_rgb, (0, 0), fx=scale, fy=scale)
        
        # Create a larger blank canvas
        border_color = (0, 0, 0)  # Black
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        canvas[:] = border_color
        
        # Place the smaller frame in the center of the canvas
        y_offset = (h - small_frame.shape[0]) // 2
        x_offset = (w - small_frame.shape[1]) // 2
        
        # Insert the smaller frame into the larger canvas
        canvas[y_offset:y_offset+small_frame.shape[0], x_offset:x_offset+small_frame.shape[1]] = small_frame
        
        # Convert to PIL Image
        img = Image.fromarray(frame_rgb)  # Using the original frame to avoid the artificial zoom effect
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        
    camera_label.after(20, update_camera)

# --- MAIN ---
if __name__ == "__main__":
    if not setup_camera():
        print("Failed to initialize camera. Exiting.")
        exit()
    
    root = create_gui()
    update_camera()
    
    try:
        root.mainloop()
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()