import cv2
import mediapipe as mpp
import numpy as np
from tensorflow.keras.models import load_model
import multiprocessing as mp
import time
import json
from tensorflow.keras.layers import Layer

# MediaPipe 초기화
mpp_holistic = mpp.solutions.holistic
mpp_drawing = mpp.solutions.drawing_utils

# 사용자 정의 레이어: GetItem
class GetItem(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index, :]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# Transformer 모델 로드
def load_transformer_model(model_path='transformer_sign_language_model.h5'):
    try:
        return load_model(model_path, custom_objects={'GetItem': GetItem})
    except Exception as e:
        print(f"모델 로드 중 오류 발생: {e}")
        return None

model = load_transformer_model()

# 클래스 이름 로드
def load_class_names(file_path="class_names.json"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{file_path} 파일을 찾을 수 없습니다.")
        return []
    except json.JSONDecodeError:
        print(f"{file_path} 파일을 읽는 중 오류가 발생했습니다.")
        return []

label_names = load_class_names()

# 키포인트 추출 함수
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand], axis=0)

# 프레임 캡처 프로세스
def capture_frames(frame_queue, stop_event):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    holistic = mpp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)

    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("웹캠 읽기 실패.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True

        keypoints = extract_keypoints(results)

        # 프레임과 키포인트를 큐에 추가
        if not frame_queue.full():
            frame_queue.put((frame, keypoints))

    cap.release()
    holistic.close()

# 모델 예측 프로세스
def predict_frames(frame_queue, result_queue, stop_event, sequence_length=30):
    sequence = []
    while not stop_event.is_set():
        if not frame_queue.empty():
            _, keypoints = frame_queue.get()
            keypoints_flattened = keypoints.flatten()
            sequence.append(keypoints_flattened)

            if len(sequence) > sequence_length:
                sequence.pop(0)

            if len(sequence) == sequence_length:
                input_data = np.expand_dims(sequence, axis=0)  # (1, sequence_length, 225)
                prediction = model.predict(input_data, verbose=0)
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_names[predicted_label_idx] if predicted_label_idx < len(label_names) else "Unknown"
                confidence = np.max(prediction) * 100

                # 콘솔에 출력
                print(f"Detected: {predicted_label}, Confidence: {confidence:.2f}%")

                # 결과 큐에 추가
                if not result_queue.full():
                    result_queue.put((predicted_label, confidence))

# 시각화 및 출력 프로세스
def display_frames(frame_queue, result_queue, stop_event):
    prev_time = time.time()

    while not stop_event.is_set():
        if not frame_queue.empty():
            frame, _ = frame_queue.get()

            # 예측 결과 표시
            if not result_queue.empty():
                predicted_label, confidence = result_queue.get()
                if confidence > 50:
                    cv2.putText(frame, f"{predicted_label} ({confidence:.2f}%)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # FPS 계산 및 표시
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # 프레임 출력
            cv2.imshow('Real-time Detection', frame)

            # 'q' 키로 종료
            if cv2.waitKey(10) & 0xFF == ord('q'):
                stop_event.set()
                break

    cv2.destroyAllWindows()

# 멀티프로세싱 실행
if __name__ == "__main__":
    frame_queue = mp.Queue(maxsize=10)
    result_queue = mp.Queue(maxsize=10)
    stop_event = mp.Event()

    capture_process = mp.Process(target=capture_frames, args=(frame_queue, stop_event))
    predict_process = mp.Process(target=predict_frames, args=(frame_queue, result_queue, stop_event))
    display_process = mp.Process(target=display_frames, args=(frame_queue, result_queue, stop_event))

    capture_process.start()
    predict_process.start()
    display_process.start()

    try:
        capture_process.join()
        predict_process.join()
        display_process.join()
    except KeyboardInterrupt:
        stop_event.set()
        capture_process.terminate()
        predict_process.terminate()
        display_process.terminate()
