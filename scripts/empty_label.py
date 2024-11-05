import os
import pandas as pd

# 경로 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
csv_output_dir = os.path.join(project_root, 'data', 'csv_output')
videos_dir = os.path.join(project_root, 'data', 'raw')

# 비어 있는 CSV 파일 목록 수집
empty_csv_files = []
for csv_file in os.listdir(csv_output_dir):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(csv_output_dir, csv_file)
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                empty_csv_files.append(csv_file)
        except pd.errors.EmptyDataError:
            empty_csv_files.append(csv_file)

# 해당하는 비디오 파일 목록 생성
videos_to_reprocess = []
for csv_file in empty_csv_files:
    video_filename = csv_file.replace('landmarks_', '').replace('.csv', '')
    video_path = os.path.join(videos_dir, video_filename)
    
    # 확장자 추가 (필요한 경우)
    if not os.path.exists(video_path):
        if os.path.exists(video_path + '.mp4'):
            video_path += '.mp4'
        elif os.path.exists(video_path + '.avi'):
            video_path += '.avi'
        else:
            print(f"{video_filename}에 해당하는 비디오 파일을 찾을 수 없습니다.")
            continue
    videos_to_reprocess.append(video_path)

# 비디오 파일 목록을 파일로 저장
videos_to_reprocess_file = os.path.join(project_root, 'videos_to_reprocess.txt')
with open(videos_to_reprocess_file, 'w') as f:
    for video_path in videos_to_reprocess:
        f.write(video_path + '\n')

print(f"재처리할 비디오 파일 수: {len(videos_to_reprocess)}")