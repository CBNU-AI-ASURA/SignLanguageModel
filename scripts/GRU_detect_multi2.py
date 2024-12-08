import cv2
import mediapipe as mpp
import numpy as np
from tensorflow.keras.models import load_model
import multiprocessing as mp
import time
import logging

print("OpenCV imported successfully")

# 로그 설정
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# MediaPipe 초기화
mpp_holistic = mpp.solutions.holistic
mpp_drawing = mpp.solutions.drawing_utils

# 모델 로드
model = load_model('bestGRU_model.keras')
# import os
# model_path = os.path.join(os.path.dirname(__file__), 'bestGRU_model.keras')
# model = load_model(model_path)

print("TensorFlow imported successfully")

label_names = ['10분', '119', '1호', '1회', '2호', '3시', '40분', '4사람', '5분', '8호', '9호', '가능', '가다',
               '갈아타다', '감사합니다', '건너다', '경찰', '고속터미널', '고장', '곳', '곳곳', '공기청정기', '괜찮다',
               '교통카드', '교환하다', '국립박물관', '그만', '급하다', '기다리다', '길', '까먹다', '끄다', '끝', '나',
               '나르다', '나사렛', '난방', '내리다', '냄새', '늦다', '다시', '다음', '단말기터치', '당신', '대로', '도와주다',
               '도움받다', '도착', '돈', '돈얼마', '돈주다', '되다', '들어올리다', '딱', '떨어지다', '마포대교', '막차', '만원',
               '맞다', '명동', '몇분', '몇사람', '몇호', '모르다', '목적', '무엇', '문자받다', '물품보관', '미안합니다', '반갑다',
               '받다', '발생하다', '방법', '방황', '백화점', '버스', '번호', '보건소', '보다', '부르다', '불가능', '불량', '불편하다',
               '빨리', '뼈곳', '사거리', '사다', '샛길', '서대문농아인복지관', '서울농아인협회', '서울역', '수고', '시간', '시청', '신분당',
               '신분증', '신호등', '실종', '심하다', '쓰러지다', '아니다', '아직', '아프다', '안내소', '안내하다', '안녕하세요', '안되다',
               '안전벨트', '알다', '알려받다', '알려주다', '어떻게', '어렵다', '어린이교통카드', '어린이집', '언덕', '얼마', '없다', '에어컨',
               '엘리베이터', '여기', '역무원', '연착', '영수증', '오다오다', '오른쪽', '오천원', '오케이', '올리다', '옷가게', '왜', '왼쪽',
               '용산역', '우회전', '원래', '원하다', '위아래', '위험', '유턴', '육교', '응급실', '일정하다', '잃어버리다', '있다', '자판기',
               '잘못', '잘못하다', '잠실대교', '장애인복지카드', '저기', '전', '전화걸다', '접근', '정기권', '조심', '좌회전', '주세요',
               '주차', '죽다', '중', '지름길', '지연되다', '지하철', '짐', '차내리다', '차두다', '차따라가다', '차밀리다', '찾다', '천안아산역',
               '천천히', '철로', '첫차', '청음회관', '충분', '충분하다', '충전', '카드', '카톡보내다', '켜다', '타다', '트렁크닫다', '트렁크열다',
               '틀리다', '파리바게트', '편의점', '편지', '표', '표지판', '필요', '필요없다', '한국농아인협회', '항상', '해보다', '화나다', '확인',
               '확인증', '회의실', '횡단보도', '힘들다']


# 키포인트 추출 함수
def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 3))
    left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose, left_hand, right_hand], axis=0)

# 프레임 캡처 프로세스
def capture_frames(frame_queue):
    logging.debug("Capture process started.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("카메라를 열 수 없습니다.")
        return
    holistic = mpp_holistic.Holistic(min_detection_confidence=0.3, min_tracking_confidence=0.3)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (320, 240))
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        keypoints = extract_keypoints(results)
        if frame_queue.full():
            frame_queue.get()
        frame_queue.put((frame, keypoints))
    cap.release()
    holistic.close()

# 모델 예측 프로세스
def predict_frames(frame_queue, result_queue, fixed_length=60):
    logging.debug("Predict process started.")
    sequence = []
    while True:
        if not frame_queue.empty():
            _, keypoints = frame_queue.get()
            sequence.append(keypoints)
            if len(sequence) > fixed_length:
                sequence.pop(0)
            if len(sequence) == fixed_length:
                input_data = np.expand_dims(sequence, axis=0).reshape(1, fixed_length, -1)
                prediction = model.predict(input_data, verbose=0)
                predicted_label_idx = np.argmax(prediction)
                predicted_label = label_names[predicted_label_idx]
                confidence = np.max(prediction) * 100
                logging.info(f"Prediction: {predicted_label}, Confidence: {confidence:.2f}%")
                if result_queue.full():
                    result_queue.get()
                result_queue.put((predicted_label, confidence))

# 시각화 및 출력 프로세스
def display_frames(frame_queue, result_queue):
    logging.debug("Display process started.")
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
            cv2.imshow('Real-time Detection', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
    cv2.destroyAllWindows()

# 멀티프로세싱 실행
if __name__ == "__main__":
    mp.freeze_support()
    frame_queue = mp.Queue(maxsize=5)
    result_queue = mp.Queue(maxsize=5)
    capture_process = mp.Process(target=capture_frames, args=(frame_queue,))
    predict_process = mp.Process(target=predict_frames, args=(frame_queue, result_queue))
    display_process = mp.Process(target=display_frames, args=(frame_queue, result_queue))
    capture_process.start()
    predict_process.start()
    display_process.start()
    capture_process.join()
    predict_process.join()
    display_process.join()
