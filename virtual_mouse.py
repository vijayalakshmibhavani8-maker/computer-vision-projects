import cv2
import mediapipe as mp
import pyautogui
import math

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()


while True:
    success, img = cap.read()

    if not success:
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = img.shape

            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            screen_x = screen_w * hand_landmarks.landmark[8].x
            screen_y = screen_h * hand_landmarks.landmark[8].y
            x2 = int(hand_landmarks.landmark[12].x * w)
            y2 = int(hand_landmarks.landmark[12].y * h)

            pyautogui.moveTo(screen_x, screen_y)
            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            distance = math.hypot(x2 - x, y2 - y)

        if distance < 30:
            pyautogui.click()
    cv2.putText(
        img,
        "CLICK",
        (20, 150),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )
    

    cv2.imshow("Virtual Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
