import cv2
import mediapipe as mp
import math
import numpy as np
import screen_brightness_control as sbc

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for handlms in result.multi_hand_landmarks:
            landmarks = []
            for id, lm in enumerate(handlms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmarks.append((cx, cy))

            if landmarks:
                x1, y1 = landmarks[4]   # Thumb tip
                x2, y2 = landmarks[8]   # Index finger tip
                length = math.hypot(x2 - x1, y2 - y1)
                brightness = int(np.clip((length - 30) * 2, 0, 100))

                sbc.set_brightness(brightness)
                cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, f"Brightness: {brightness}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            mp_draw.draw_landmarks(img, handlms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Brightness control by gestures", img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
