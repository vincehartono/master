import os
from glob import glob

from PIL import Image

from screen_reader_cards import detect_cards, extract_card_region


def get_latest_main() -> str:
    """
    Return the most recent PNG under AI_Baccarat/screenshots.
    Prefer files named main_*.png.
    """
    base_dir = os.path.join("AI_Baccarat", "screenshots")
    patterns = [
        os.path.join(base_dir, "main_*.png"),
        os.path.join(base_dir, "*.png"),
    ]
    candidates = []
    for pat in patterns:
        candidates.extend(glob(pat))
    if not candidates:
        return ""
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def main() -> None:
    path = get_latest_main()
    if not path:
        print("No PNG screenshots found under AI_Baccarat\\screenshots")
        return

    print("Using full screenshot:", path)
    full = Image.open(path).convert("RGB")

    # Save the card band crop so you can inspect what detect_cards sees
    card_band = extract_card_region(full)
    band_path = os.path.join("AI_Baccarat", "screenshots", "debug_card_band.png")
    card_band.save(band_path)
    print("Saved card band crop to:", band_path)

    cards = detect_cards(full)
    if cards:
        print("Detected cards:", cards)
    else:
        print("No cards detected from this full screenshot.")


if __name__ == "__main__":
    main()

