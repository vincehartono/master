import os
import time
from glob import glob
from datetime import datetime
from typing import Optional, Tuple, Dict, List

import cv2
import mss
import numpy as np
from PIL import Image
import pytesseract

Region = Tuple[int, int, int, int]  # (left, top, width, height)


def grab_region(region: Optional[Region] = None) -> Image.Image:
    with mss.mss() as sct:
        if region is None:
            monitor = sct.monitors[1]
        else:
            left, top, width, height = region
            monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(monitor)
        return Image.frombytes("RGB", shot.size, shot.rgb)


def save_screenshot(img: Image.Image, prefix: str) -> str:
    base_dir = os.path.join("AI_Baccarat", "screenshots")
    os.makedirs(base_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    path = os.path.join(base_dir, f"{prefix}_{ts}.png")
    img.save(path)
    return path


def extract_card_region(full_img: Image.Image) -> Image.Image:
    """Band where cards appear (tuned for 1920x1080)."""
    w, h = full_img.size
    left = int(w * 0.18)
    right = int(w * 0.82)
    top = int(h * 0.32)
    bottom = int(h * 0.58)
    return full_img.crop((left, top, right, bottom))


def extract_history_region(full_img: Image.Image) -> Image.Image:
    """Band with bead-road grids at the bottom (tuned for 1920x1080).

    This is also the region we use for both card + history template
    matching, so that the runtime detector sees the same content as
    the saved debug_history PNGs.
    """
    w, h = full_img.size
    left = int(w * 0.03)
    right = int(w * 0.97)
    top = int(h * 0.63)
    bottom = int(h * 0.93)
    return full_img.crop((left, top, right, bottom))


# -------- Template loading helpers (suits + history icons) --------


def _load_suit_templates() -> Dict[str, np.ndarray]:
    """
    Load suit templates (heart/diamond/club/spade) from
    AI_Baccarat/screenshots/suits.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots", "suits")
    pattern = os.path.join(base_dir, "*.png")
    files = glob(pattern)
    templates: Dict[str, np.ndarray] = {}

    for path in files:
        name = os.path.basename(path).lower()
        label: Optional[str] = None
        if "heart" in name or "_h" in name or "h." in name:
            label = "H"
        elif "diamond" in name or "_d" in name or "d." in name:
            label = "D"
        elif "club" in name or "_c" in name or "c." in name:
            label = "C"
        elif "spade" in name or "_s" in name or "s." in name:
            label = "S"

        if label is None:
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[label] = img

    return templates


def _load_history_templates() -> Dict[str, np.ndarray]:
    """
    Load banker/player/tie templates from AI_Baccarat/screenshots/suits.
    Filenames should contain 'player', 'banker', or 'tie'.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots", "suits")
    pattern = os.path.join(base_dir, "*.png")
    files = glob(pattern)
    templates: Dict[str, np.ndarray] = {}

    for path in files:
        name = os.path.basename(path).lower()
        label: Optional[str] = None
        if "player" in name:
            label = "P"
        elif "banker" in name:
            label = "B"
        elif "tie" in name:
            label = "T"

        if label is None:
            continue

        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        templates[label] = img

    return templates


def _ocr_card_rank(card_img: Image.Image) -> str:
    """OCR rank in top-left of a single card."""
    w, h = card_img.size
    corner = card_img.crop((0, 0, int(w * 0.4), int(h * 0.4)))
    corner = corner.resize((corner.size[0] * 2, corner.size[1] * 2), Image.LANCZOS)
    gray = cv2.cvtColor(np.array(corner), cv2.COLOR_RGB2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pil = Image.fromarray(th)
    txt = pytesseract.image_to_string(
        pil, config="--psm 10 -c tessedit_char_whitelist=A23456789TJQK"
    )
    txt = txt.strip().upper()
    for ch in txt:
        if ch in "A23456789TJQK":
            return ch
    return ""


def _detect_cards_from_history_band(band: Image.Image) -> str:
    """
    Card + suit detection tuned for the same history band used for debug_history
    screenshots. Returns strings like 'P:9H4D|B:5S?'.
    """
    bgr = cv2.cvtColor(np.array(band), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    card_regions: List[Tuple[int, int, int, int]] = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 1500:
            continue
        aspect = ch / float(cw + 1e-6)
        if aspect < 1.1 or aspect > 2.8:
            continue
        card_regions.append((x, y, cw, ch))

    if len(card_regions) < 2:
        return ""

    card_regions.sort(key=lambda r: r[0])

    suit_templates = _load_suit_templates()

    ranks: List[str] = []
    suits: List[str] = []
    for x, y, cw, ch in card_regions:
        card_img = band.crop((x, y, x + cw, y + ch))
        rank = _ocr_card_rank(card_img)
        ranks.append(rank or "?")

        cw_img, ch_img = card_img.size
        sx1 = int(cw_img * 0.4)
        sy1 = int(ch_img * 0.2)
        sx2 = int(cw_img * 0.9)
        sy2 = int(ch_img * 0.8)
        suit_crop = card_img.crop((sx1, sy1, sx2, sy2))

        suit_label = "?"
        if suit_templates:
            gray = cv2.cvtColor(np.array(suit_crop), cv2.COLOR_RGB2GRAY)
            best_label = None
            best_score = -1.0
            for label, tmpl in suit_templates.items():
                try:
                    th, tw = tmpl.shape
                    scaled = cv2.resize(tmpl, (gray.shape[1], gray.shape[0]))
                    res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = max_val
                        best_label = label
                except Exception:
                    continue
            # Slightly lower threshold so suits are more often detected
            if best_label is not None and best_score > 0.20:
                suit_label = best_label

        suits.append(suit_label)

    mid_x = w / 2.0
    player: List[str] = []
    banker: List[str] = []
    for (x, _, cw, _), rank, suit in zip(card_regions, ranks, suits):
        cx = x + cw / 2.0
        card_token = rank + suit
        if cx < mid_x:
            player.append(card_token)
        else:
            banker.append(card_token)

    if not player and not banker:
        return ""

    return f"P:{''.join(player)}|B:{''.join(banker)}"


def detect_cards(full_img: Image.Image) -> str:
    """
    Entry point used by the live monitor.

    We now delegate to the history-band detector so that runtime uses
    the same view and logic as the offline debug scripts.
    """
    band = extract_history_region(full_img)
    return _detect_cards_from_history_band(band)


def detect_history(full_img: Image.Image) -> str:
    """
    Return bead-road history as sequence of B/P/T using template
    matching against the banker / player / tie icons.
    """
    hist_img = extract_history_region(full_img)

    debug_dir = os.path.join("AI_Baccarat", "screenshots", "debug_history")
    os.makedirs(debug_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    hist_img.save(os.path.join(debug_dir, f"hist_{ts}.png"))

    board = hist_img
    w, h = board.size

    # Crop approximate bead-road board on the left side (same as debug helper)
    x1 = int(w * 0.02)
    x2 = int(w * 0.40)
    y1 = int(h * 0.10)
    y2 = int(h * 0.90)
    board = board.crop((x1, y1, x2, y2))

    board_gray = cv2.cvtColor(np.array(board), cv2.COLOR_RGB2GRAY)
    templates = _load_history_templates()
    if not templates:
        return ""

    detections: List[Tuple[float, float, str]] = []
    for label, tmpl in templates.items():
        try:
            th, tw = tmpl.shape
            if board_gray.shape[0] < th or board_gray.shape[1] < tw:
                continue
            res = cv2.matchTemplate(board_gray, tmpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= 0.6)
            for pt_y, pt_x in zip(loc[0], loc[1]):
                cx = pt_x + tw / 2.0
                cy = pt_y + th / 2.0
                detections.append((cx, cy, label))
        except Exception:
            continue

    if not detections:
        return ""

    xs = sorted({cx for cx, _, _ in detections})
    ys = sorted({cy for _, cy, _ in detections})

    def _cluster(vals: List[float], min_gap: float) -> List[float]:
        if not vals:
            return []
        centers = [vals[0]]
        for v in vals[1:]:
            if abs(v - centers[-1]) >= min_gap:
                centers.append(v)
        return centers

    bw, bh = board_gray.shape[1], board_gray.shape[0]
    col_centers = _cluster(xs, min_gap=bw * 0.015)
    row_centers = _cluster(ys, min_gap=bh * 0.08)

    def _nearest_center(v: float, centers: List[float]) -> int:
        return min(range(len(centers)), key=lambda i: abs(centers[i] - v))

    grid: Dict[Tuple[int, int], str] = {}
    for cx, cy, label in detections:
        col_idx = _nearest_center(cx, col_centers)
        row_idx = _nearest_center(cy, row_centers)
        key = (col_idx, row_idx)
        if key not in grid:
            grid[key] = label

    items = sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    return "".join(label for (_, _), label in items)


def _is_complete_hand(cards: str) -> bool:
    """
    Heuristic: only accept hands where each side has 2 or 3 cards.
    cards format: 'P:..|B:..'
    """
    if not cards or "|" not in cards:
        return False
    try:
        p_part, b_part = cards.split("|", 1)
        p_ranks = p_part.split(":", 1)[1] if ":" in p_part else ""
        b_ranks = b_part.split(":", 1)[1] if ":" in b_part else ""
    except Exception:
        return False
    return (2 <= len(p_ranks) <= 3) and (2 <= len(b_ranks) <= 3)


def monitor(
    interval_seconds: float = 0.75,
    log_path: str = "AI_Baccarat\\shoe_history_cards.csv",
    print_screenshot: bool = True,
) -> None:
    print("Starting card + history monitor. Ctrl+C to stop.")
    last_hist = ""
    last_cards = ""

    # Prepare CSV log for full shoe history
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("timestamp,shoe_history,cards\n")

    try:
        while True:
            full = grab_region(None)
            hist_seq = detect_history(full)
            cards = detect_cards(full)

            # Only log when we have a complete hand (2–3 cards each side)
            if not _is_complete_hand(cards):
                time.sleep(interval_seconds)
                continue

            if hist_seq != last_hist or cards != last_cards:
                path = save_screenshot(full, prefix="main")
                print("-" * 40)
                print(time.strftime("%H:%M:%S"), "update")
                print("shoe_history:", hist_seq or "(none)")
                print("cards:", cards or "(none)")
                if print_screenshot:
                    print("screenshot:", path)

                # Append to CSV log
                with open(log_path, "a", encoding="utf-8") as f:
                    ts = datetime.now().isoformat(timespec="milliseconds")
                    f.write(f"{ts},{hist_seq},{cards}\n")

                last_hist = hist_seq
                last_cards = cards

            time.sleep(interval_seconds)
    except KeyboardInterrupt:
        print("\nStopped monitor.")


if __name__ == "__main__":
    monitor()
