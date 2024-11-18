import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import time

# MediaPipe 초기화
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# 모델 로드
model = load_model('bestGRU_model.keras')
label_names = ['가능', '괜찮다', '기다리다', '끝', '도착', '돈', '되다', '맞다',
               '불편하다', '수고', '실종', '심하다', '영수증', '원하다', '유턴',
               '잃어버리다', '접근', '차밀리다', '카드', '필요', '화나다']

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand], axis=0)

def run_realtime_detection():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    holistic = mp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    sequence = []
    fixed_length = 60
    target_fps = 30
    frame_duration = 1 / target_fps
    predict_interval = 3
    frame_count = 0

    try:
        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("웹캠을 읽는 데 실패했습니다.")
                break

            frame = cv2.resize(frame, (320, 240))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > fixed_length:
                sequence.pop(0)

            if len(sequence) == fixed_length and frame_count % predict_interval == 0:
                input_data = np.expand_dims(sequence, axis=0).reshape(1, fixed_length, -1)
                prediction = model.predict(input_data, verbose=0)
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_names[predicted_label_idx]
                confidence = np.max(prediction) * 100

                if confidence > 50:
                    cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            elapsed_time = time.time() - start_time
            if elapsed_time < frame_duration:
                time.sleep(frame_duration - elapsed_time)

            fps = 1 / (time.time() - start_time)
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Real-time Detection', frame)

            frame_count += 1
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        holistic.close()

if __name__ == "__main__":
    run_realtime_detection()
