import cv2
import mediapipe as mp
import csv
import os

# MediaPipe 모듈 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1)
hands = mp_hands.Hands(max_num_hands=2)

# 비디오 파일이 있는 디렉토리
source_directory = './source/vedio/'  # 비디오 파일들이 있는 디렉토리

# 데이터셋 저장할 디렉토리 생성
output_directory = './dataset/'
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# 디렉토리 내 모든 비디오 파일 처리
for filename in os.listdir(source_directory):
    if filename.endswith('.mp4') or filename.endswith('.avi'):  # .mp4 또는 .avi 확장자 파일만 처리
        video_path = os.path.join(source_directory, filename)
        cap = cv2.VideoCapture(video_path)

        # 출력 비디오 파일명 생성
        output_video_path = os.path.join(output_directory, f'output_{filename}')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out_video = cv2.VideoWriter(output_video_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        # CSV 파일명 생성
        csv_filename = f'landmarks_{filename.split(".")[0]}.csv'
        csv_filepath = os.path.join(output_directory, csv_filename)
        landmarks_file = open(csv_filepath, mode='w', newline='')
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

            # 프레임을 화면에 표시 (선택적, 비활성화 가능)
            # cv2.imshow('Upper Body and Hands Tracking', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 자원 해제 및 파일 닫기
        cap.release()
        out_video.release()
        landmarks_file.close()

# 모든 창 닫기
cv2.destroyAllWindows()
