import os
from glob import glob

import cv2
import numpy as np
from PIL import Image

from screen_reader_cards import detect_cards, extract_card_region


def get_all_screenshots() -> list[str]:
    """
    Return all PNGs under AI_Baccarat/screenshots/debug_history, sorted by mtime.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots", "debug_history")
    pattern = os.path.join(base_dir, "*.png")
    candidates = glob(pattern)
    candidates.sort(key=os.path.getmtime)
    return candidates


def load_suit_templates() -> dict[str, np.ndarray]:
    """
    Load suit templates from AI_Baccarat/screenshots/suits.
    Filenames should contain indicators like heart/diamond/club/spade or H/D/C/S.
    Returns mapping label -> grayscale template.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots", "suits")
    pattern = os.path.join(base_dir, "*.png")
    files = glob(pattern)
    templates: dict[str, np.ndarray] = {}

    for path in files:
        name = os.path.basename(path).lower()
        label = None
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


def load_history_templates() -> dict[str, np.ndarray]:
    """
    Load banker/player/tie templates from AI_Baccarat/screenshots/suits.
    Filenames should contain 'player', 'banker', or 'tie'.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots", "suits")
    pattern = os.path.join(base_dir, "*.png")
    files = glob(pattern)
    templates: dict[str, np.ndarray] = {}

    for path in files:
        name = os.path.basename(path).lower()
        label = None
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


def history_from_crop(img: Image.Image) -> str:
    """
    Derive B/P/T history from a debug_history crop image using banker/player/tie templates.
    Assumes the main bead-road board is on the left side of the image.
    """
    w, h = img.size
    # Crop approximate bead-road board on the left (tuned for your PNGs)
    x1 = int(w * 0.02)
    x2 = int(w * 0.40)
    y1 = int(h * 0.10)
    y2 = int(h * 0.90)
    board = img.crop((x1, y1, x2, y2))

    board_gray = cv2.cvtColor(np.array(board), cv2.COLOR_RGB2GRAY)
    templates = load_history_templates()
    if not templates:
        return ""

    detections = []
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

    # Derive logical grid by clustering unique x/y positions instead of
    # assuming fixed spacing. This tends to be more robust to small
    # cropping differences.
    xs = sorted({cx for cx, _, _ in detections})
    ys = sorted({cy for _, cy, _ in detections})

    # Merge very close coordinates into the same bucket
    def _cluster(vals: list[float], min_gap: float) -> list[float]:
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

    def _nearest_center(v: float, centers: list[float]) -> int:
        return min(range(len(centers)), key=lambda i: abs(centers[i] - v))

    grid: dict[tuple[int, int], str] = {}
    for cx, cy, label in detections:
        col_idx = _nearest_center(cx, col_centers)
        row_idx = _nearest_center(cy, row_centers)
        key = (col_idx, row_idx)
        if key not in grid:
            grid[key] = label

    items = sorted(grid.items(), key=lambda kv: (kv[0][0], kv[0][1]))
    return "".join(label for (_, _), label in items)


def detect_cards_from_history_crop(img: Image.Image) -> str:
    """
    A variant of card detection tuned for the history-strip screenshots.
    These images already contain mostly the lower half of the table,
    so we run rectangle detection on the full image rather than
    applying the full-screen extract_card_region cropping.
    """
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 70, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    card_regions = []
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 1500:  # slightly looser than main detector
            continue
        aspect = ch / float(cw + 1e-6)
        if aspect < 1.1 or aspect > 2.8:
            continue
        card_regions.append((x, y, cw, ch))

    if len(card_regions) < 2:
        return ""

    card_regions.sort(key=lambda r: r[0])

    # Reuse _ocr_card_rank via screen_reader_cards.detect_cards by building a fake band
    # Instead of importing private helpers, we just crop here and OCR directly
    from screen_reader_cards import _ocr_card_rank  # type: ignore

    suit_templates = load_suit_templates()

    ranks = []
    suits = []
    for x, y, cw, ch in card_regions:
        card_img = img.crop((x, y, x + cw, y + ch))
        rank = _ocr_card_rank(card_img)
        ranks.append(rank or "?")

        # Suit crop: region to the right of the rank, roughly center-top
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
                    # Resize template to fit crop height
                    th, tw = tmpl.shape
                    scaled = cv2.resize(tmpl, (gray.shape[1], gray.shape[0]))
                    res = cv2.matchTemplate(gray, scaled, cv2.TM_CCOEFF_NORMED)
                    min_val, max_val, _, _ = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = max_val
                        best_label = label
                except Exception:
                    continue
            # Lower threshold a bit to be more permissive
            if best_label is not None and best_score > 0.2:
                suit_label = best_label

        suits.append(suit_label)

    mid_x = w / 2.0
    player = []
    banker = []
    for (x, _, cw, _), rank, suit in zip(card_regions, ranks, suits):
        cx = x + cw / 2.0
        if cx < mid_x:
            player.append(rank + suit)
        else:
            banker.append(rank + suit)

    if not player and not banker:
        return ""

    return f"P:{''.join(player)}|B:{''.join(banker)}"


def main() -> None:
    paths = get_all_screenshots()
    if not paths:
        print("No PNG screenshots found under AI_Baccarat\\screenshots\\debug_history")
        return

    print(f"Found {len(paths)} screenshots. Scanning all for cards and history...\n")

    any_found = False
    last_history = ""

    for path in paths:
        img = Image.open(path).convert("RGB")
        cards = detect_cards_from_history_crop(img)
        history_seq = history_from_crop(img)

        if cards:
            any_found = True
            print("FILE:", path)
            print("  cards:", cards)
            print("  history:", history_seq or "(none)")
            last_history = history_seq or last_history
        else:
            # Delete images where no cards were detected, per your request
            try:
                os.remove(path)
                print("Deleted (no cards):", path)
            except OSError as e:
                print("Could not delete", path, ":", e)

    if any_found:
        print("\nLatest non-empty history sequence:", last_history or "(none)")
    else:
        print("No cards detected in any debug_history PNGs.")




if __name__ == "__main__":
    main()
