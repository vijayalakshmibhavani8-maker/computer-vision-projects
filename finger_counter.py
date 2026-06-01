import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

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

            # Green dot on index finger tip
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)

            cv2.circle(img, (x, y), 10, (0, 255, 0), cv2.FILLED)

            count = 0

            # Thumb
            if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x:
                count += 1

            # Index
            if hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y:
                count += 1

            # Middle
            if hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y:
                count += 1

            # Ring
            if hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y:
                count += 1

            # Pinky
            if hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y:
                count += 1

            cv2.putText(
                img,
                f"Fingers: {count}",
                (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 0, 0),
                2
            )

    cv2.imshow("Finger Counter", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()