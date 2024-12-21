import cv2
import mediapipe as mpp
import numpy as np
from tensorflow.keras.models import load_model
import multiprocessing as mp
import time
import logging
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe 초기화
mpp_holistic = mpp.solutions.holistic
mpp_drawing = mpp.solutions.drawing_utils

# 모델 및 클래스 이름 로드
# model = load_model(r'tmp/best_model.keras')
model = load_model(r'tmp/best_transformer_model.keras')
with open(r"tmp/class_names_GRU.json", "r", encoding="utf-8") as f:
    label_names = json.load(f)

logging.info("모델 및 클래스 이름 로드 완료")


def extract_keypoints(results):
    # pose 33개, left_hand 21개, right_hand 21개 랜드마크 -> (75,3)
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand], axis=0)

def capture_frames(frame_queue):
    logging.info("캡처 프로세스 시작")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("카메라를 열 수 없습니다.")
        return
    
    # MediaPipe Holistic 초기화
    with mpp_holistic.Holistic(
        min_detection_confidence=0.7, 
        min_tracking_confidence=0.7
    ) as holistic:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            
            keypoints = extract_keypoints(results)
            
            if frame_queue.full():
                frame_queue.get()
            frame_queue.put((frame, keypoints))
    
    cap.release()

def predict_frames(frame_queue, result_queue, fixed_length=90):
    logging.info("예측 프로세스 시작")
    sequence = []
    while True:
        if not frame_queue.empty():
            _, keypoints = frame_queue.get()
            sequence.append(keypoints)
            if len(sequence) > fixed_length:
                sequence.pop(0)
            
            if len(sequence) == fixed_length:
                # (1, 90, 75*3) 형태로 입력
                input_data = np.expand_dims(sequence, axis=0).reshape(1, fixed_length, -1)
                prediction = model.predict(input_data, verbose=0)
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_names[predicted_label_idx]
                confidence = np.max(prediction) * 100
                logging.info(f"예측 결과: {predicted_label}, 신뢰도: {confidence:.2f}%")
                if result_queue.full():
                    result_queue.get()
                result_queue.put((predicted_label, confidence))

def display_frames(frame_queue, result_queue):
    logging.info("디스플레이 프로세스 시작")
    prev_time = time.time()
    while True:
        if not frame_queue.empty():
            frame, _ = frame_queue.get()
            if not result_queue.empty():
                predicted_label, confidence = result_queue.get()
                if confidence > 50:
                    cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            cv2.imshow('Real-time Sign Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    mp.freeze_support()
    frame_queue = mp.Queue(maxsize=5)
    result_queue = mp.Queue(maxsize=5)

    capture_process = mp.Process(target=capture_frames, args=(frame_queue,))
    predict_process = mp.Process(target=predict_frames, args=(frame_queue, result_queue, 90))
    display_process = mp.Process(target=display_frames, args=(frame_queue, result_queue))

    capture_process.start()
    predict_process.start()
    display_process.start()

    capture_process.join()
    predict_process.join()
    display_process.join()
