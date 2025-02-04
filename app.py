import cv2
import mediapipe as mp
from twilio.rest import Client

account_sid = 'AC4e96cf5587faff23a6940612a36f0bf4'  
auth_token = '95015ebbc3c960b2d1003bb0c02d6712'   
twilio_number = '+17274757615' 
recipient_number = '+919080183858'  

client = Client(account_sid, auth_token)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

previous_state = None
state_change_count = 0

def detect_gesture(landmarks):
    thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    ring_tip = landmarks[mp_hands.HandLandmark.RING_FINGER_TIP]
    pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

    if (index_tip.y < thumb_tip.y and middle_tip.y < thumb_tip.y and 
        ring_tip.y < thumb_tip.y and pinky_tip.y < thumb_tip.y):
        return 'open'
    elif (index_tip.y > thumb_tip.y and middle_tip.y > thumb_tip.y and 
          ring_tip.y > thumb_tip.y and pinky_tip.y > thumb_tip.y):
        return 'danger'
    else:
        return 'unknown'

def check_consecutive_state_changes(current_state):
    global previous_state, state_change_count

    if current_state != previous_state and current_state in ['open', 'danger']:
        state_change_count += 1
        print(f"State changed to: {current_state}. Change count: {state_change_count}")
        previous_state = current_state
    elif current_state == previous_state:
        return  

    if state_change_count >= 8:
        print("ALERT: State changed between 'open' and 'danger' four times consecutively!")
        
        send_alert_message("ALERT: State changed between 'open' and 'danger' four times consecutively!")
        state_change_count = 0  

def send_alert_message(message):
    client.messages.create(
        body=message,
        from_=twilio_number,
        to=recipient_number
    )
    print(f"Alert sent: {message}")

with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty frame.")
            continue

        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                gesture = detect_gesture(hand_landmarks.landmark)
                check_consecutive_state_changes(gesture)  

                cv2.putText(image, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Hand Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
