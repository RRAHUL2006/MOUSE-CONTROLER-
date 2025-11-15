"""
Mediapipe Hand -> Mouse Controller with window mapping + drag support

- Pointer: index fingertip
- Left click: quick pinch (index+thumb)
- Drag: pinch and hold (mouseDown while pinch persists)
- Right click: index+middle pinch
- Optional: map pointer to the Active Window's client area instead of full screen

Dependencies:
    pip install mediapipe opencv-python pyautogui numpy pygetwindow

Usage:
    python mediapipe_mouse_window.py

Keys:
    ESC - quit
    p   - pause / resume pointer control
    w   - toggle window-mapped mode (map to active window)
"""
import time
import math
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# optional helper to read active window bounds on Windows/macOS/Linux (may need pyobjc/pywin32 on some OS)
try:
    import pygetwindow as gw
except Exception:
    gw = None

# ---------------- CONFIG ----------------
CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

SMOOTHING_ALPHA = 0.25      # EMA factor for pointer smoothing (0..1)
POS_BUFFER_LEN = 4          # median buffer length

PINCH_THRESH = 0.06         # normalized distance threshold to detect pinch (tune)
PINCH_CONSEC_FRAMES = 5     # frames pinch must persist to register click
CLICK_COOLDOWN = 0.5        # seconds between click events

DRAG_HOLD_FRAMES = 8        # frames of sustained pinch to start dragging (hover then hold)
RIGHT_CLICK_FRAMES = 6      # frames for right-click registration (index+middle)

DRAW_LANDMARKS = True
SHOW_FPS = True

# Map to window instead of full screen?
WINDOW_MAPPED = True        # start in window-mapped mode if pygetwindow is available

# pyautogui settings
pyautogui.FAILSAFE = True
screen_w, screen_h = pyautogui.size()

# ---------------- STATE ----------------
smoothed_x = None
smoothed_y = None
pos_buffer = deque(maxlen=POS_BUFFER_LEN)

left_counter = 0
right_counter = 0
last_left_time = 0.0
last_right_time = 0.0

dragging = False
drag_start_time = 0.0

paused = False

# Mediapipe init
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.5)

# Camera
cap = cv2.VideoCapture(CAM_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

def norm_to_screen(x_norm, y_norm, win_bbox=None):
    """
    Convert mediapipe normalized coords (0..1) to screen coordinates.
    If win_bbox provided, map to that window's client area (left, top, width, height).
    """
    if win_bbox is None:
        # full-screen mapping (flip x for mirroring)
        sx = (1.0 - x_norm) * screen_w
        sy = y_norm * screen_h
    else:
        left, top, w, h = win_bbox
        sx = (1.0 - x_norm) * w + left
        sy = y_norm * h + top
    sx = max(0, min(screen_w - 1, int(sx)))
    sy = max(0, min(screen_h - 1, int(sy)))
    return sx, sy

def euclid(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])

def get_active_window_bbox():
    """
    Try to return the active window bounding box as (left, top, width, height).
    If unavailable, return None and we'll map to full screen.
    """
    if gw is None:
        return None
    try:
        w = gw.getActiveWindow()
        if w is None:
            return None
        try:
            left, top = w.topleft
            width, height = w.width, w.height
        except Exception:
            left, top = w.left, w.top
            width, height = w.width, w.height
        return (left, top, width, height)
    except Exception:
        return None

print("Mediapipe mouse controller (window-mapped + drag). ESC to quit, 'p' to pause, 'w' toggle window-mapped mode.")
prev_time = time.time()

# main loop
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror
    h, w_frame, _ = frame.shape

    # process frame
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    pointer_drawn = False
    action_text = ""
    # fps
    if SHOW_FPS:
        now = time.time()
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0.0
        prev_time = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # Determine mapping bbox if WINDOW_MAPPED
    win_bbox = None
    if WINDOW_MAPPED:
        win_bbox = get_active_window_bbox()
        if win_bbox is None:
            win_bbox = None

    if not paused and results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        lm = hand.landmark
        idx = lm[8]    # index tip
        thumb = lm[4]  # thumb tip
        mid = lm[12]   # middle tip

        # get screen coords according to mapping mode
        sx, sy = norm_to_screen(idx.x, idx.y, win_bbox=win_bbox)

        # smoothing: median buffer + EMA
        pos_buffer.append((sx, sy))
        bx = int(np.median([p[0] for p in pos_buffer]))
        by = int(np.median([p[1] for p in pos_buffer]))

        # NOTE: smoothed_x and smoothed_y are top-level variables; do NOT redeclare global here
        if smoothed_x is None:
            smoothed_x, smoothed_y = bx, by
        else:
            smoothed_x = SMOOTHING_ALPHA * bx + (1 - SMOOTHING_ALPHA) * smoothed_x
            smoothed_y = SMOOTHING_ALPHA * by + (1 - SMOOTHING_ALPHA) * smoothed_y

        # Move system mouse
        try:
            pyautogui.moveTo(int(smoothed_x), int(smoothed_y), duration=0.01)
        except Exception as e:
            print("pyautogui error:", e)

        pointer_drawn = True
        cv2.circle(frame, (int(idx.x * w_frame), int(idx.y * h)), 10, (255, 0, 0), -1)
        cv2.putText(frame, "Pointer", (int(idx.x * w_frame) + 12, int(idx.y * h) + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)

        # distances (normalized)
        d_thumb = euclid((idx.x, idx.y), (thumb.x, thumb.y))
        d_mid = euclid((idx.x, idx.y), (mid.x, mid.y))

        # LEFT click logic (index+thumb)
        if d_thumb < PINCH_THRESH:
            left_counter += 1
        else:
            left_counter = 0

        # start a drag if pinch held beyond DRAG_HOLD_FRAMES
        if left_counter >= DRAG_HOLD_FRAMES and not dragging:
            if (time.time() - last_left_time) > 0.15:
                try:
                    pyautogui.mouseDown()
                    dragging = True
                    drag_start_time = time.time()
                    action_text = "Drag started"
                    print("Drag started")
                except Exception as e:
                    print("mouseDown error:", e)

        # click if quick pinch (sustained for smaller PINCH_CONSEC_FRAMES but less than drag threshold)
        if left_counter >= PINCH_CONSEC_FRAMES and left_counter < DRAG_HOLD_FRAMES and (time.time() - last_left_time) > CLICK_COOLDOWN:
            try:
                pyautogui.click()
                last_left_time = time.time()
                action_text = "Left Click"
                print("Left click")
            except Exception as e:
                print("Left click error:", e)
            left_counter = 0

        # release drag when pinch ends
        if dragging and left_counter == 0:
            try:
                pyautogui.mouseUp()
                dragging = False
                action_text = "Drag released"
                print("Drag released")
                last_left_time = time.time()
            except Exception as e:
                print("mouseUp error:", e)

        # RIGHT click logic (index+middle)
        if d_mid < PINCH_THRESH:
            right_counter += 1
        else:
            right_counter = 0

        if right_counter >= RIGHT_CLICK_FRAMES and (time.time() - last_right_time) > CLICK_COOLDOWN:
            try:
                pyautogui.rightClick()
                last_right_time = time.time()
                action_text = "Right Click"
                print("Right click")
            except Exception as e:
                print("Right click error:", e)
            right_counter = 0

        # draw helper lines and distances
        thumb_px = (int(thumb.x * w_frame), int(thumb.y * h))
        mid_px = (int(mid.x * w_frame), int(mid.y * h))
        idx_px = (int(idx.x * w_frame), int(idx.y * h))
        cv2.line(frame, idx_px, thumb_px, (0,255,255), 2)
        cv2.line(frame, idx_px, mid_px, (0,255,255), 2)
        cv2.putText(frame, f"t={d_thumb:.3f} m={d_mid:.3f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        if DRAW_LANDMARKS:
            mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

    # overlay status
    mode_text = "PAUSED" if paused else ("Window-mapped" if WINDOW_MAPPED else "Full-screen")
    cv2.putText(frame, f"Mode: {mode_text}", (10, frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)
    cv2.putText(frame, f"Action: {action_text}", (10, frame.shape[0]-35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,0), 2)
    cv2.putText(frame, "ESC quit | p pause | w toggle window mode", (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Mediapipe Mouse Controller (Window Mapped)", frame)

    # key handling
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('p'):
        paused = not paused
        print("Paused" if paused else "Resumed")
    elif key == ord('w'):
        # toggle mapping mode (only if pygetwindow is available)
        if gw is None:
            WINDOW_MAPPED = False
            print("pygetwindow not available — can't enable window mapping.")
        else:
            WINDOW_MAPPED = not WINDOW_MAPPED
            print("Window mapping:", WINDOW_MAPPED)

# cleanup
if dragging:
    try:
        pyautogui.mouseUp()
    except Exception:
        pass
cap.release()
cv2.destroyAllWindows()
hands.close()
