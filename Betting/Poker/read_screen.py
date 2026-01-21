import cv2
import numpy as np
import pyautogui
import time
import os

# Path to the folder with card templates (e.g., "templates/ace_of_spades.png", etc.)
TEMPLATE_FOLDER = "C:\\Users\\Vince\\master\\Betting\\Poker\\Card_Templates_Stakes"

def load_templates():
    templates = {}
    for filename in os.listdir(TEMPLATE_FOLDER):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            path = os.path.join(TEMPLATE_FOLDER, filename)
            card_name = os.path.splitext(filename)[0]
            templates[card_name] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return templates

def capture_screen(region):
    screenshot = pyautogui.screenshot(region=region)
    frame = np.array(screenshot)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

def identify_card(screen_img, templates, threshold=0.8):
    for name, template in templates.items():
        for scale in np.linspace(0.8, 1.2, 10):  # Try 10 scales from 80% to 120%
            resized = cv2.resize(template, (0, 0), fx=scale, fy=scale)
            if resized.shape[0] > screen_img.shape[0] or resized.shape[1] > screen_img.shape[1]:
                continue
            res = cv2.matchTemplate(screen_img, resized, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(res)
            if max_val >= threshold:
                print(f"Matched {name} at scale {scale:.2f}, score {max_val:.3f}")
                return name
    return "Unknown"

def main():
    print("Loading card templates...")
    templates = load_templates()
    
    print("Capturing screen in 3 seconds...")
    time.sleep(3)

    region=(2000, 100, 4000, 1000)

    screen = capture_screen(region)
    card = identify_card(screen, templates)
    cv2.imwrite("debug_capture.png", screen)
    
    print(f"Identified card: {card}")

if __name__ == "__main__":
    main()
