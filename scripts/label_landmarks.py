import pandas as pd
import os
import json
from glob import glob

# 상수 설정
FPS = 30  # 영상의 프레임 레이트

# 비디오 파일이 있는 디렉토리
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

morpheme_root = os.path.join(project_root, 'data', 'morpheme') 
# 파일 경로 설정 (사용자에 맞게 수정)
csv_path = os.path.join(project_root, 'data', 'csv_output')

# 출력 데이터 저장 디렉토리 생성
output_dir = os.path.join(project_root, 'data', 'labeled_keypoints')
os.makedirs(output_dir, exist_ok=True)

# Sentence 1532에 대한 keypoint CSV 파일 목록 가져오기
csv_files = glob(os.path.join(csv_path, "landmarks_*.csv"))

# 각 keypoint CSV 파일 처리
for csv_file in csv_files:
    # 파일명에서 정보 추출
    filename = os.path.basename(csv_file)
    parts = filename.split('_')
    sentence_id = parts[1]  # 예: 'NIA'
    video_info = '_'.join(parts[1:]).split('.')[0]  # 'NIA_SL_SEN1532_REAL01_F'

    # 참가자 번호 및 각도 정보 추출
    video_parts = video_info.split('_')
    participant_info = video_parts[3]  # 'REAL01'
    angle_info = video_parts[4]  # 'F'

    participant_num = participant_info[-2:]  # '01'

    # 키포인트 데이터 로드
    keypoint_df = pd.read_csv(csv_file)

    # 프레임 번호를 시간으로 변환
    keypoint_df['time'] = keypoint_df['frame_num'] / FPS

    # 해당하는 morpheme JSON 파일 로드
    morpheme_filename = f"{video_info}_morpheme.json"
    morpheme_file = os.path.join(morpheme_root, participant_num, morpheme_filename)

    if not os.path.exists(morpheme_file):
        print(f"Morpheme file not found for {csv_file}")
        continue

    with open(morpheme_file, 'r', encoding='utf-8') as f:
        morpheme_data = json.load(f)

    # Morpheme 세그먼트 처리
    segments = morpheme_data['data']

    labeled_frames = []

    for segment in segments:
        start_time = segment['start']
        end_time = segment['end']
        label = segment['attributes'][0]['name']

        # 해당 시간 구간의 키포인트 데이터 추출
        segment_df = keypoint_df[
            (keypoint_df['time'] >= start_time) &
            (keypoint_df['time'] <= end_time)
        ].copy()
        segment_df['label'] = label

        labeled_frames.append(segment_df)

    if labeled_frames:
        labeled_keypoint_df = pd.concat(labeled_frames)
        # 출력 CSV 파일명 설정
        output_csv_filename = f"labeled_{filename}"
        output_csv_path = os.path.join(output_dir, output_csv_filename)
        # CSV 파일로 저장
        labeled_keypoint_df.to_csv(output_csv_path, index=False)
        print(f"Labeled data saved to {output_csv_path}")
    else:
        print(f"No labeled data for {csv_file}")