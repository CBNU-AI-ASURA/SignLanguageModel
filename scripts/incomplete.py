import os
import pandas as pd
from collections import defaultdict

# 경로 설정
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))
keypoint_csv_dir = os.path.join(project_root, 'data', 'csv_output')
raw_videos_dir = os.path.join(project_root, 'data', 'raw')

# 불완전한 파일 목록 저장
incomplete_videos = []

# 키포인트 CSV 파일 목록 가져오기
keypoint_csv_files = [f for f in os.listdir(keypoint_csv_dir) if f.endswith('.csv')]

# 파일 정보를 저장할 딕셔너리 초기화
file_info = []

# 키포인트 CSV 파일에서 SEN 및 REAL 번호 추출 및 정보 저장
for csv_file in keypoint_csv_files:
    csv_path = os.path.join(keypoint_csv_dir, csv_file)
    try:
        # 파일명에서 정보 추출
        # 예시: landmarks_NIA_SL_SEN0001_REAL01_F.csv
        parts = csv_file.replace('landmarks_', '').replace('.csv', '').split('_')
        if len(parts) < 5:
            print(f"파일명 형식이 올바르지 않습니다: {csv_file}")
            continue
        sen = parts[2]  # 'SEN0001'
        real = parts[3]  # 'REAL01'

        # CSV 파일의 행 수 계산
        df = pd.read_csv(csv_path)
        num_rows = len(df)

        # 파일 정보 저장
        file_info.append({
            'filename': csv_file,
            'path': csv_path,
            'sen': sen,
            'real': real,
            'num_rows': num_rows
        })
    except Exception as e:
        print(f"{csv_file} 처리 중 오류 발생: {e}")
        continue

# 그룹별로 파일들을 묶기 위한 딕셔너리 초기화
groups = defaultdict(list)

# 파일 정보를 그룹화
for info in file_info:
    key = (info['sen'], info['real'])
    groups[key].append(info)

# 각 그룹 내에서 행 수를 비교하여 불완전한 파일 식별
for key, files in groups.items():
    # 행 수를 리스트로 추출
    num_rows_list = [file['num_rows'] for file in files]
    # 행 수의 평균 및 표준편차 계산
    mean_rows = sum(num_rows_list) / len(num_rows_list)
    threshold = mean_rows * 0.7  # 평균 행 수의 70% 미만인 경우 불완전한 파일로 간주
    for file in files:
        if file['num_rows'] < threshold:
            # 불완전한 CSV 파일의 영상 제목 추출
            video_title = file['filename'].replace('landmarks_', '').replace('.csv', '') + '.mp4'
            video_path = os.path.join(raw_videos_dir, video_title)
            if os.path.exists(video_path):
                incomplete_videos.append(video_path)
            else:
                print(f"{video_title}에 해당하는 비디오 파일이 없습니다.")

print(f"불완전한 비디오 파일 수: {len(incomplete_videos)}")

# 불완전한 비디오 파일 목록을 파일로 저장
incomplete_videos_path = os.path.join(project_root, 'incomplete_videos_to_process.txt')
with open(incomplete_videos_path, 'w') as f:
    for video_path in incomplete_videos:
        f.write(video_path + '\n')

print(f"불완전한 비디오 파일 목록이 {incomplete_videos_path}에 저장되었습니다.")