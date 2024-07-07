import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing.
mp_drawing = mp.solutions.drawing_utils

# Open video capture.
cap = cv2.VideoCapture(0)

# Create a blank canvas for drawing.
canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Initialize the canvas with the same size as the frame if it's None.
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Flip the image horizontally for a selfie-view display.
    frame = cv2.flip(frame, 1)
    # Convert the BGR image to RGB.
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and draw landmarks.
    results = hands.process(rgb_image)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the positions of the tip of the index finger and the middle of the wrist.
            index_tip = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y])
            wrist = np.array([hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x, hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y])

            # Check if only the index and middle fingers are extended.
            if (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                # Move the cursor without drawing or erasing.
                pass

            # Check if only the index finger is extended.
            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y > hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                # Draw on the canvas.
                cv2.circle(canvas, (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])), 10, (255, 255, 255), -1)

            # Check if all fingers are extended.
            elif (hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y and
                hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y):
                # Erase on the canvas. Increase the radius of the circle to make the eraser bigger.
                cv2.circle(canvas, (int(index_tip[0] * frame.shape[1]), int(index_tip[1] * frame.shape[0])), 30, (0, 0, 0), -1)


            # Draw the hand landmarks on the frame.
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Combine the frame and the canvas.
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)

    # Display the image.
    cv2.imshow('MediaPipe Hands', frame)

    # Break loop on 'q' key press.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Release video capture.
cap.release()
