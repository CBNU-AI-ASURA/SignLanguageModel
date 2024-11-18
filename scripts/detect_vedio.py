import os
import sys
import csv
import time
from multiprocessing import Pool
from tqdm import tqdm
import argparse

# Set environment variables before imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # Suppress all TensorFlow logs except errors
os.environ['GLOG_minloglevel'] = '3'        # Suppress GLOG logs
os.environ['KMP_WARNINGS'] = '0'            # Suppress OpenMP warnings
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'   # Suppress OpenCV logs

def process_video(args):
    sys.stderr.flush()
    devnull = open(os.devnull, 'w')
    os.dup2(devnull.fileno(), 2)

    import cv2
    import mediapipe as mp

    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_pose.Pose(model_complexity=1)
    hands = mp_hands.Hands(max_num_hands=2)

    video_path, csv_output_directory = args
    filename = os.path.basename(video_path)

    # 임시 파일 경로 설정
    csv_filename = f'landmarks_{os.path.splitext(filename)[0]}.csv'
    csv_filepath = os.path.join(csv_output_directory, csv_filename)
    temp_csv_filepath = csv_filepath + '.tmp'  # 임시 파일

    try:
        cap = cv2.VideoCapture(video_path)
        with open(temp_csv_filepath, mode='w', newline='') as landmarks_file:
            csv_writer = csv.writer(landmarks_file)
            csv_writer.writerow(['frame_num', 'landmark_type', 'index', 'x', 'y', 'z'])

            frame_num = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_num += 1
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_result = pose.process(rgb_frame)
                hands_result = hands.process(rgb_frame)

                if pose_result.pose_landmarks:
                    for idx, landmark in enumerate(pose_result.pose_landmarks.landmark):
                        csv_writer.writerow([frame_num, 'pose', idx, landmark.x, landmark.y, landmark.z])

                if hands_result.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hands_result.multi_hand_landmarks):
                        hand_label = hands_result.multi_handedness[hand_idx].classification[0].label
                        hand_type = 'left_hand' if hand_label == 'Left' else 'right_hand'

                        for idx, landmark in enumerate(hand_landmarks.landmark):
                            csv_writer.writerow([frame_num, hand_type, idx, landmark.x, landmark.y, landmark.z])

        # 임시 파일을 최종 파일명으로 변경
        os.rename(temp_csv_filepath, csv_filepath)
        # print(f"{filename} 처리 완료.")
    except Exception as e:
        sys.stderr = sys.__stderr__
        print(f"Error processing {filename}: {e}", file=sys.stderr)
        # 에러가 발생하면 임시 파일 삭제
        if os.path.exists(temp_csv_filepath):
            os.remove(temp_csv_filepath)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    videos_dir = os.path.join(project_root, 'data', 'raw')

    csv_output_directory = os.path.join(project_root, 'data', 'csv_output')
    os.makedirs(csv_output_directory, exist_ok=True)

    # 인자 파서 설정
    parser = argparse.ArgumentParser(description='Process videos with MediaPipe.')
    parser.add_argument('--video_list', type=str, help='Path to a text file containing list of video files to process.')
    args = parser.parse_args()

    # 비디오 파일 목록 생성
    if args.video_list:
        # 파일에서 비디오 파일 목록 읽기
        print("Reading video list from file...")
        with open(args.video_list, 'r') as f:
            video_files = [line.strip() for line in f if line.strip()]
    else:
        # 기존 코드: 모든 비디오 파일 처리
        videos_dir = os.path.join(project_root, 'data', 'raw')
        video_files = [os.path.join(videos_dir, f) for f in os.listdir(videos_dir)
                       if f.endswith('.mp4') or f.endswith('.avi')]
        
    total_videos = len(video_files)
    start_time_all = time.time()

    args_list = [(video_path, csv_output_directory) for video_path in video_files]

    num_processes = 1  # Adjust as needed
    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_video, args_list), total=total_videos):
            pass
        pool.close()
        pool.join()

    print("All video processing completed.")
    print(f"Total Time Taken: {time.time() - start_time_all:.2f}s")