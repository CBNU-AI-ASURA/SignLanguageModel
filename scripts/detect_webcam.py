import cv2
import mediapipe as mp
import csv

# MediaPipe 모듈 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1)
hands = mp_hands.Hands(max_num_hands=2)

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 비디오 저장을 위한 설정 (코덱 및 파일명 설정)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out_video = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# CSV 파일 열기 (랜드마크 데이터 저장)
landmarks_file = open('landmarks.csv', mode='w', newline='')
csv_writer = csv.writer(landmarks_file)
csv_writer.writerow(['frame_num', 'landmark_type', 'index', 'x', 'y', 'z'])  # CSV 파일 헤더 작성

frame_num = 0  # 프레임 번호 초기화

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1  # 프레임 번호 증가
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # MediaPipe에서 포즈 및 손 추적
    pose_result = pose.process(rgb_frame)
    hands_result = hands.process(rgb_frame)

    # 포즈 랜드마크 저장 및 그리기
    if pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            pose_result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
            csv_writer.writerow([frame_num, 'pose', idx, landmark.x, landmark.y, landmark.z])

    # 손 랜드마크 저장 및 그리기
    if hands_result.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )
            for idx, landmark in enumerate(hand_landmarks.landmark):
                csv_writer.writerow([frame_num, 'hand', idx, landmark.x, landmark.y, landmark.z])

    # 비디오에 현재 프레임 저장
    out_video.write(frame)

    # 프레임 크기를 2배로 키우기
    frame_resized = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

    # 프레임을 화면에 표시
    cv2.imshow('Upper Body and Hands Tracking', frame_resized)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제 및 파일 닫기
cap.release()
out_video.release()
landmarks_file.close()
cv2.destroyAllWindows()
