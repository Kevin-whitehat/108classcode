import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence = 0.5)


def drawLandmarks(image, hand_landmarks):
   if hand_landmarks:   
       for landmarks in hand_landmarks:   
            # mp_drawing.draw_landmarks(image, landmarks, mp_hands.HAND_CONNECTIONS)

            mp_drawing.draw_landmarks(
                        image,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing_styles.get_default_hand_connections_style())
while True:
    success,image=cap.read()
    image = cv2.flip(image,1)
    cv2.imshow("Media controlled", image)
    results = hands.process(image)

    hand_landmarks = results.multi_hand_landmarks
    print(hand_landmarks)
    image.flags.writeable = True
    drawLandmarks(image, hand_landmarks)
    key = cv2.waitKey(1)
    if key ==32:
        break






cv2.destroyAllWindows()