# import pyautogui
# import time

# print("Move the mouse to the top-left of the area you want to capture.")
# time.sleep(2)  # Wait for you to position the cursor

# x, y = pyautogui.position()
# print(f"Top-left corner of the region: x = {x}, y = {y}")

import cv2
import numpy as np
import win32gui
import win32ui
import win32con

def grab_window(window_title):
    hwnd = win32gui.FindWindow(None, window_title)
    if not hwnd:
        print("Window not found!")
        return None

    left, top, right, bottom = win32gui.GetClientRect(hwnd)
    width = right - left
    height = bottom - top

    hwnd_dc = win32gui.GetWindowDC(hwnd)
    mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
    save_dc = mfc_dc.CreateCompatibleDC()

    save_bitmap = win32ui.CreateBitmap()
    save_bitmap.CreateCompatibleBitmap(mfc_dc, width, height)
    save_dc.SelectObject(save_bitmap)
    save_dc.BitBlt((0, 0), (width, height), mfc_dc, (0, 0), win32con.SRCCOPY)

    bmpinfo = save_bitmap.GetInfo()
    bmpstr = save_bitmap.GetBitmapBits(True)
    img = np.frombuffer(bmpstr, dtype='uint8')
    img.shape = (bmpinfo['bmHeight'], bmpinfo['bmWidth'], 4)

    # Clean up
    win32gui.DeleteObject(save_bitmap.GetHandle())
    save_dc.DeleteDC()
    mfc_dc.DeleteDC()
    win32gui.ReleaseDC(hwnd, hwnd_dc)

    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Example usage
img = grab_window("PokerStars")  # Use the exact title of the window
if img is not None:
    cv2.imshow("Captured Window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
