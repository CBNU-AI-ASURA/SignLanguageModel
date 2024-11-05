import cv2
import mediapipe as mp
import csv
import os
import time

# MediaPipe 모듈 초기화
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(model_complexity=1)
hands = mp_hands.Hands(max_num_hands=2)

# 비디오 파일이 있는 디렉토리
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
videos_dir = os.path.join(project_root, 'data', 'videos', 'Origin')

# CSV와 비디오를 저장할 디렉토리 생성
csv_output_directory = os.path.join(project_root, 'data', 'csv_output')
video_output_directory = os.path.join(project_root, 'data', 'video_output')

# 디렉토리가 없으면 생성
if not os.path.exists(csv_output_directory):
    os.makedirs(csv_output_directory)
if not os.path.exists(video_output_directory):
    os.makedirs(video_output_directory)

# 총 비디오 파일수 계산
video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4') or f.endswith('.avi')]
total_videos = len(video_files)
completed_videos = 0
start_time_all = time.time()  # 전체 작업 시작 시간 기록

# 디렉토리 내 모든 비디오 파일 처리
for filename in os.listdir(videos_dir):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        video_path = os.path.join(videos_dir, filename)
        cap = cv2.VideoCapture(video_path)

        # 각 비디오 작업 시작 시간 기록
        start_time_video = time.time()

        # 출력 비디오 파일 경로 생성
        output_video_path = os.path.join(video_output_directory, f'output_{filename}')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)  # 원본 비디오의 FPS 가져오기
        out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

        # CSV 파일명 생성
        csv_filename = f'landmarks_{filename.split(".")[0]}.csv'
        csv_filepath = os.path.join(csv_output_directory, csv_filename)
        with open(csv_filepath, mode='w', newline='') as landmarks_file:
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
                        hand_label = hands_result.multi_handedness[hand_idx].classification[0].label
                        hand_type = 'left_hand' if hand_label == 'Left' else 'right_hand'

                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS
                        )
                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            csv_writer.writerow([frame_num, hand_type, idx, landmark.x, landmark.y, landmark.z])

                # 비디오에 현재 프레임 저장
                out_video.write(frame)

                # 프레임을 화면에 표시 (선택적, 비활성화 가능)
                #cv2.imshow('Upper Body and Hands Tracking', frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        # 각 비디오 작업 완료 후 시간 측정
        elapsed_time_video = time.time() - start_time_video
        completed_videos += 1
        progress = (completed_videos / total_videos) * 100
        avg_time_per_video = (time.time() - start_time_all) / completed_videos
        remaining_time = avg_time_per_video * (total_videos - completed_videos)

        print(f"\rProgress: {progress:.2f}% | {completed_videos}/{total_videos} | Average Time per Video: {avg_time_per_video:.2f}s | Estimated Remaining Time: {remaining_time:.2f}s",end='')

        # 자원 해제
        cap.release()
        out_video.release()

# 완료 문자열 출력
print("All video processing completed.")
print(f"Total Time Taken: {time.time() - start_time_all:.2f}s")

# 모든 창 닫기
cv2.destroyAllWindows()