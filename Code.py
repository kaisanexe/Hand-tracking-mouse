import cv2
import mediapipe as mp
import pyautogui
import math
import numpy as np
import time

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

# Smoothing parameters
smooth_factor = 7  # higher = smoother but slower
cursor_x, cursor_y = 0, 0
target_x, target_y = 0, 0

# Click states
left_click_active = False
right_click_active = False

pTime = 0

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    h, w, c = img.shape

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            if len(lmList) >= 21:
                x_thumb, y_thumb = lmList[4]
                x_index, y_index = lmList[8]
                x_middle, y_middle = lmList[12]

                # Distance between fingers
                dist_index_thumb = math.hypot(x_index - x_thumb, y_index - y_thumb)
                dist_middle_thumb = math.hypot(x_middle - x_thumb, y_middle - y_thumb)

                # Map hand position (index tip) to screen
                mapped_x = np.interp(x_index, (80, w - 80), (0, screen_w))
                mapped_y = np.interp(y_index, (80, h - 80), (0, screen_h))

                # Exponential smoothing
                cursor_x = cursor_x + (mapped_x - cursor_x) / smooth_factor
                cursor_y = cursor_y + (mapped_y - cursor_y) / smooth_factor

                # Determine gesture
                if dist_index_thumb < 35 and dist_middle_thumb > 45:
                    # Left click / drag
                    cv2.circle(img, ((x_index + x_thumb)//2, (y_index + y_thumb)//2), 12, (0, 255, 0), cv2.FILLED)
                    if not left_click_active:
                        pyautogui.mouseDown()
                        left_click_active = True
                    pyautogui.moveTo(cursor_x, cursor_y)
                    right_click_active = False

                elif dist_middle_thumb < 35 and dist_index_thumb > 45:
                    # Right click
                    cv2.circle(img, ((x_middle + x_thumb)//2, (y_middle + y_thumb)//2), 12, (255, 0, 0), cv2.FILLED)
                    if not right_click_active:
                        pyautogui.click(button='right')
                        right_click_active = True
                    left_click_active = False

                elif dist_index_thumb > 50 and dist_middle_thumb > 50:
                    # Open palm (hover mode)
                    pyautogui.moveTo(cursor_x, cursor_y)
                    if left_click_active:
                        pyautogui.mouseUp()
                        left_click_active = False
                    right_click_active = False

                # Draw small tracker circle
                cv2.circle(img, (x_index, y_index), 8, (255, 0, 255), cv2.FILLED)

    # FPS counter (optional)
    cTime = time.time()
    fps = 1 / (cTime - pTime) if (cTime - pTime) != 0 else 0
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    cv2.imshow("Smooth Gesture Mouse", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
