import cv2
import numpy as np
import pyautogui
import time

cam = cv2.VideoCapture(0)

# HSV ranges (may need tuning)
lower_blue = np.array([90, 100, 100])
upper_blue = np.array([130, 255, 255])
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])
lower_green = np.array([35, 100, 100])
upper_green = np.array([85, 255, 255])

ROI_X, ROI_Y, ROI_W, ROI_H = 50, 50, 300, 300
MIN_AREA = 500
CLICK_COOLDOWN = 0.5  # seconds

screen_w, screen_h = pyautogui.size()
last_left_click = 0
last_right_click = 0

# Small kernel for morphology
kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = cam.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    image_smooth = cv2.GaussianBlur(frame, (7,7), 0)

    # ROI crop (work with crop to avoid offset confusion)
    roi = image_smooth[ROI_Y:ROI_Y+ROI_H, ROI_X:ROI_X+ROI_W]

    # Draw grid + ROI rectangle on full frame for visual feedback
    cv2.rectangle(frame, (ROI_X, ROI_Y), (ROI_X+ROI_W, ROI_Y+ROI_H), (0,255,255), 2)
    for i in range(4):
        cv2.line(frame, (ROI_X + i * 100, ROI_Y), (ROI_X + i * 100, ROI_Y + ROI_H), (0,255,255), 1)
        cv2.line(frame, (ROI_X, ROI_Y + i * 100), (ROI_X + ROI_W, ROI_Y + i * 100), (0,255,255), 1)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # BLUE - pointer (clean mask)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > MIN_AREA:
            M = cv2.moments(largest)
            if M['m00'] != 0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                # Draw on the full frame (apply ROI offset)
                cv2.circle(frame, (ROI_X + cx, ROI_Y + cy), 8, (255,0,0), -1)

                # Map ROI coords to screen coords
                screen_x = np.interp(cx, [0, ROI_W], [0, screen_w])
                screen_y = np.interp(cy, [0, ROI_H], [0, screen_h])

                # Move mouse to mapped screen location (small duration)
                pyautogui.moveTo(screen_x, screen_y, duration=0.02)

    # GREEN - right click
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(mask_green) > 2000 and time.time() - last_right_click > CLICK_COOLDOWN:
        pyautogui.rightClick()
        last_right_click = time.time()

    # RED - left click
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    if cv2.countNonZero(mask_red) > 2000 and time.time() - last_left_click > CLICK_COOLDOWN:
        pyautogui.click()
        last_left_click = time.time()

    cv2.imshow("frame", frame)
    # fast key poll (1 ms). ESC to quit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
