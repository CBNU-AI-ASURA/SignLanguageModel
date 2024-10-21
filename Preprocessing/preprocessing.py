import os
import json
import cv2
import mediapipe as mp
import csv

# MediaPipe 모듈 초기화
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
hands = mp_hands.Hands(max_num_hands=2)
pose = mp_pose.Pose()
mp_draw = mp.solutions.drawing_utils

# 최상위 디렉토리 설정
'''
SignLanguageModel/ 디렉토리에서 실행해야하며, 영상 파일이 담겨 있는 디렉토리 명은 'video'로 설정 바람
현재는 csv파일 내부에서 따로 분리하지 않으므로 한 묶음의 영상 디렉토리만 진행하길 바람 혹은 코드 수정 바람
'''
top_dir = os.getcwd()  # 현재 작업 디렉토리
morpheme_dir = os.path.join(top_dir, 'morpheme')
video_dir = os.path.join(top_dir, 'video')
csv_dir = os.path.join(top_dir, 'csv')

# CSV 디렉토리 생성
if not os.path.exists(csv_dir):
    os.makedirs(csv_dir)

# morpheme 디렉토리 내부 순회
for folder in sorted(os.listdir(morpheme_dir)):
    folder_path = os.path.join(morpheme_dir, folder)
    
    if os.path.isdir(folder_path):
        print(f"Processing folder: {folder}")

        # F 파일만 처리
        for file in os.listdir(folder_path):
            if file.endswith('_F_morpheme.json'):
                file_path = os.path.join(folder_path, file)
                print(f"Processing file: {file_path}")
                
                # JSON 파일 읽기
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 메타데이터에서 비디오 파일 이름 추출
                video_name = data['metaData']['name']
                video_base_name = os.path.splitext(video_name)[0]  # 확장자 제외
                video_path = os.path.join(video_dir, video_name)

                if not os.path.exists(video_path):
                    print(f"Video file not found: {video_path}")
                    continue
                
                # 해당 영상 파일 기반 디렉토리 생성
                video_csv_dir = os.path.join(csv_dir, video_base_name)
                if not os.path.exists(video_csv_dir):
                    os.makedirs(video_csv_dir)

                # 영상 처리 및 CSV 파일 생성
                print(f"Processing video: {video_path}")
                cap = cv2.VideoCapture(video_path)

                csv_file_path = os.path.join(video_csv_dir, f"{video_base_name}.csv")
                with open(csv_file_path, mode='w', newline='') as file:
                    writer = csv.writer(file)

                    # 헤더 작성
                    writer.writerow(['frame', 'landmark_type', 'x', 'y', 'z'])

                    frame_num = 0
                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            break

                        # 영상을 RGB로 변환 (MediaPipe는 RGB 입력을 받음)
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # 손과 상체(포즈) 인식
                        hand_results = hands.process(img_rgb)
                        pose_results = pose.process(img_rgb)

                        # 손이 감지되었을 경우
                        if hand_results.multi_hand_landmarks:
                            for hand_landmarks in hand_results.multi_hand_landmarks:
                                # 손 좌표 CSV로 저장
                                for id, lm in enumerate(hand_landmarks.landmark):
                                    writer.writerow([frame_num, 'hand', lm.x, lm.y, lm.z])

                        # 포즈(상체)가 감지되었을 경우
                        if pose_results.pose_landmarks:
                            # 상체 좌표 CSV로 저장
                            for id, lm in enumerate(pose_results.pose_landmarks.landmark):
                                writer.writerow([frame_num, 'pose', lm.x, lm.y, lm.z])

                        frame_num += 1

                # 자원 해제
                cap.release()

                print(f"CSV saved: {csv_file_path}")

print("Processing completed.")
