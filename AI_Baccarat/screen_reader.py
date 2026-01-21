import os
import re
import time
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

import mss
import numpy as np
import cv2
from PIL import Image
import pytesseract

Region = Tuple[int, int, int, int]  # (left, top, width, height)


# =========================
# 1) Screen capture
# =========================
def grab_region(region: Optional[Region] = None) -> Image.Image:
    """
    Grab a region of the screen and return it as a PIL Image.

    :param region: (left, top, width, height) in screen pixels.
                   If None, use full primary monitor.
    """
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]  # full primary monitor
        else:
            left, top, width, height = region
            monitor = {"left": left, "top": top, "width": width, "height": height}

        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def save_screenshot(img: Image.Image, prefix: str = "baccarat") -> str:
    """
    Save a screenshot image to ./screenshots/ with a timestamped filename.
    Returns the saved filepath.
    """
    os.makedirs("screenshots", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join("screenshots", f"{prefix}_{ts}.png")
    img.save(path)
    return path


# =========================
# 2) OCR preprocessing
# =========================
def preprocess_for_ocr(img: Image.Image) -> Image.Image:
    """
    Preprocess the image for better OCR:
      - grayscale
      - upscale
      - blur
      - adaptive threshold
    """
    arr = np.array(img)  # RGB
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    # Upscale improves OCR on small fonts
    gray = cv2.resize(gray, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # Reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold helps with varied backgrounds
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )

    return Image.fromarray(th)


def ocr_image(img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> str:
    """
    Run OCR on a PIL image.

    :param psm: Tesseract Page Segmentation Mode.
    :param whitelist: Optional whitelist string for allowed chars.
    """
    img = preprocess_for_ocr(img)

    config = f"--psm {psm}"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"

    return pytesseract.image_to_string(img, config=config)


# =========================
# 3) Parsers
# =========================
def _to_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def parse_baccarat_panel(text: str) -> Dict[str, Any]:
    """
    Parse key baccarat fields from the MAIN region OCR text.
    Returns only fields found.
    """
    data: Dict[str, Any] = {}

    # Balance $228.00
    m = re.search(r"Balance\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", text, re.IGNORECASE)
    if m:
        data["balance"] = _to_float(m.group(1))

    # Total Bet $0.00
    m = re.search(r"Total\s*Bet\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)", text, re.IGNORECASE)
    if m:
        data["total_bet"] = _to_float(m.group(1))

    # Table number like "#41"
    m = re.search(r"#\s*([0-9]{1,4})", text)
    if m:
        data["table_number"] = int(m.group(1))

    # Baccarat limits
    m = re.search(
        r"Baccarat\s+Limits\s*\$?\s*([0-9]+)\s*-\s*\$?\s*([0-9]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        data["min_bet"] = int(m.group(1))
        data["max_bet"] = int(m.group(2))

    # Which option words are visible (NOT the outcome, just presence)
    options = []
    for key in ["PLAYER", "BANKER", "TIE", "SMALL", "BIG"]:
        if re.search(rf"\b{key}\b", text, re.IGNORECASE):
            options.append(key)
    if options:
        data["options_present"] = options

    return data


def parse_history_from_text(text: str, max_len: int = 60) -> str:
    """
    Parse B/P/T sequence from HISTORY region OCR text.
    NOTE: This only works if the history area actually contains letters or
    OCR-able symbols that Tesseract converts into B/P/T.

    If your casino UI uses circles/icons (not letters), you'll need OpenCV
    template detection instead of OCR.
    """
    raw = text.upper()

    # Normalize common OCR confusions
    raw = raw.replace("8", "B")   # icons sometimes appear as 8
    raw = raw.replace("|", "I")
    raw = raw.replace("I", "T")   # thin 'T' sometimes read as 'I'
    raw = raw.replace("TI", "T")
    raw = raw.replace("TE", "T")

    seq = re.findall(r"[BPT]", raw)
    return "".join(seq)[-max_len:]


# =========================
# 3b) Image-based helpers (stubs)
# =========================
def extract_card_region(full_img: Image.Image) -> Image.Image:
    """
    Return a cropped area that contains only the player/banker cards.
    Coords are tuned for 1920x1080; adjust as needed.
    """
    w, h = full_img.size
    # Rough band across the felt where cards appear (just above betting buttons)
    left = int(w * 0.18)
    right = int(w * 0.82)
    top = int(h * 0.32)
    bottom = int(h * 0.58)
    return full_img.crop((left, top, right, bottom))


def extract_history_region(full_img: Image.Image) -> Image.Image:
    """
    Return a cropped area for the bead-road history grids at the bottom.
    Coords are tuned for 1920x1080; adjust as needed.
    """
    w, h = full_img.size
    # Lower band where the large bead-road grids live
    left = int(w * 0.03)
    right = int(w * 0.97)
    top = int(h * 0.63)
    bottom = int(h * 0.93)
    return full_img.crop((left, top, right, bottom))


def detect_cards_from_image(full_img: Image.Image) -> str:
    """
    Placeholder: image-based card recognition.
    Currently returns an empty string and does not save any debug files.

    Next steps: implement card segmentation + template matching so this
    function returns something like 'P:K♥,8♦|B:4♣,9♣' for each hand.
    """
    return ""


def detect_history_from_image(full_img: Image.Image) -> str:
    """
    Image-based history (B/P/T) recognition from bead-road grid on the right.

    This implementation is heuristic:
      - crops the bottom band containing the history boards
      - thresholds for red / blue / green circles in HSV
      - finds circle centroids and orders them in bead-road reading order
      - emits a sequence like 'BBBPB...' (ignoring empties)
    """
    hist_img = extract_history_region(full_img)
    os.makedirs("screenshots\\debug_history", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    hist_img.save(os.path.join("screenshots\\debug_history", f"hist_{ts}.png"))

    # Convert to OpenCV BGR / HSV
    bgr = cv2.cvtColor(np.array(hist_img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Color masks (tuned loosely for your UI)
    # Blue circles (Player)
    lower_blue = np.array([90, 80, 80])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    # Red circles (Banker) – wrap-around in HSV
    lower_red1 = np.array([0, 80, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 80, 80])
    upper_red2 = np.array([180, 255, 255])
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    # Green circles (Tie)
    lower_green = np.array([40, 60, 60])
    upper_green = np.array([85, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    masks = [("P", mask_blue), ("B", mask_red), ("T", mask_green)]

    detections = []
    for label, mask in masks:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            area = cv2.contourArea(c)
            if area < 20:  # skip noise
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            if radius < 3:
                continue
            detections.append((int(x), int(y), label))

    if not detections:
        return ""

    # Sort detections into bead-road order: column left->right, within column top->bottom
    # Estimate column width from overall width and typical grid size (~40 cols)
    h_w, h_h = hist_img.size
    approx_cols = 40
    col_width = h_w / approx_cols

    # Assign column index
    enriched = []
    for x, y, label in detections:
        col_idx = int(x // col_width)
        enriched.append((col_idx, y, label))

    enriched.sort(key=lambda t: (t[0], t[1]))

    seq = "".join(label for _, _, label in enriched)
    return seq

# =========================
# 4) Monitor loop (2 regions)
# =========================
def monitor_baccarat(
    main_region: Optional[Region] = None,
    history_region: Optional[Region] = None,
    interval_seconds: float = 0.75,
    screenshot_every_change: bool = True,
    screenshot_every_n_seconds: Optional[float] = None,
    debug_print_raw: bool = False,
) -> None:
    """
    OCR + parse baccarat using a MAIN region and optional HISTORY region.

    - MAIN region: balance, bet, limits, etc.
    - HISTORY region: bead road / past outcomes area (optional but recommended)

    Screenshots:
      - If screenshot_every_change True: saves screenshot when OCR text changes.
      - If screenshot_every_n_seconds provided: also saves periodic screenshots.

    Press Ctrl+C to stop.
    """
    print("Starting baccarat OCR monitor. Press Ctrl+C to stop.")
    last_main_text = ""
    last_main_parsed: Dict[str, Any] = {}
    last_hist_seq = ""

    last_shot_time = 0.0

    # Whitelist for main panel (loosely)
    main_whitelist = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz:.%$+-/()# "
    # History is usually sparse; whitelist only BPT + a bit
    hist_whitelist = "BPTbpt8|I "

    try:
        while True:
            now = time.time()

            # --- MAIN TEXT PANEL (for balance/limits/etc.) ---
            main_img = grab_region(main_region)
            main_text = ocr_image(main_img, psm=6, whitelist=main_whitelist).strip()

            # --- FULL FRAME FOR IMAGE-BASED SIGNALS (cards, bead-road) ---
            full_img = grab_region(None)
            cards_signal = detect_cards_from_image(full_img)
            hist_img_signal = detect_history_from_image(full_img)

            # --- HISTORY ---
            hist_seq = ""
            hist_text = ""
            if history_region is not None:
                hist_img = grab_region(history_region)
                # PSM 11 works well for sparse text; sometimes try 7 too
                hist_text = ocr_image(hist_img, psm=11, whitelist=hist_whitelist).strip()
                hist_seq = parse_history_from_text(hist_text)

            changed = (main_text != last_main_text) or (hist_seq != last_hist_seq)

            # Screenshots on change (save full frame for review)
            if changed and screenshot_every_change:
                path = save_screenshot(full_img, prefix="main")
                if history_region is not None:
                    _ = save_screenshot(grab_region(history_region), prefix="history")
                print(f"📸 Saved screenshots: {path} (+ history if enabled)")

            # Periodic screenshots
            if screenshot_every_n_seconds is not None and (now - last_shot_time) >= screenshot_every_n_seconds:
                path = save_screenshot(main_img, prefix="main_periodic")
                if history_region is not None:
                    _ = save_screenshot(grab_region(history_region), prefix="history_periodic")
                last_shot_time = now
                print(f"📸 Periodic screenshot saved: {path}")

            if changed and main_text:
                main_parsed = parse_baccarat_panel(main_text)

                if debug_print_raw:
                    print("-" * 40)
                    print(time.strftime("%H:%M:%S"), "RAW MAIN OCR:")
                    print(main_text)
                    if history_region is not None:
                        print("\nRAW HISTORY OCR:")
                        print(hist_text)

                # Print parsed updates
                print("-" * 40)
                print(time.strftime("%H:%M:%S"), "parsed:")

                if main_parsed:
                    for k, v in main_parsed.items():
                        print(f"{k}: {v}")
                else:
                    print("(no baccarat fields found)")

                if history_region is not None:
                    if hist_seq:
                        print(f"shoe_history: {hist_seq}")
                    else:
                        print("shoe_history: (none)")

                last_main_parsed = main_parsed
                last_main_text = main_text
                last_hist_seq = hist_seq

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\nStopped baccarat OCR monitor.")


# =========================
# 5) Optional helper: quick calibration window
# =========================
def calibrate_region(region: Region) -> None:
    """
    Show a single grabbed region in a window to help you adjust coordinates.
    Close window or press any key to exit.
    """
    with mss.mss() as sct:
        left, top, width, height = region
        frame = np.array(sct.grab({"left": left, "top": top, "width": width, "height": height}))
        cv2.imshow("calibration", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # =========================
    # IMPORTANT: SET YOUR REGIONS
    # =========================
    # You MUST change these coordinates to match your screen.
    #
    # MAIN_REGION should include:
    # - Balance
    # - Total Bet
    # - Bet buttons (PLAYER/BANKER/TIE etc.)
    #
    # HISTORY_REGION should be ONLY the bead road / history grid.
    #
    # MAIN_REGION: wide band covering cards + bottom history panels.
    # Tune these numbers with calibrate_region if needed.
    MAIN_REGION: Optional[Region] = (120, 220, 1680, 720)
    HISTORY_REGION: Optional[Region] = None   # e.g. (1100, 320, 280, 220)

    # If you want to tune regions visually, uncomment one at a time:
    # calibrate_region(MAIN_REGION)
    # calibrate_region(HISTORY_REGION)

    monitor_baccarat(
        main_region=MAIN_REGION,
        history_region=HISTORY_REGION,      # set to None if you only want main panel
        interval_seconds=0.75,
        screenshot_every_change=True,       # saves screenshots when OCR changes
        screenshot_every_n_seconds=None,    # e.g. 10.0 for periodic screenshots every 10s
        debug_print_raw=False,              # True prints raw OCR text (noisy)
    )
