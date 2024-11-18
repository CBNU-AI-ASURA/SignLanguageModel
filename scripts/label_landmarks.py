import pandas as pd
import os
import json
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

# 상수 설정
FPS = 30  # 영상의 프레임 레이트

# 디렉토리 설정
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..'))

morpheme_root = os.path.join(project_root, 'data', 'morpheme')
csv_path = os.path.join(project_root, 'data', 'csv_output')
output_dir = os.path.join(project_root, 'data', 'labeled_keypoints')
os.makedirs(output_dir, exist_ok=True)

# 키포인트 CSV 파일 목록 가져오기
csv_files = glob(os.path.join(csv_path, "landmarks_*.csv"))

def process_csv_file(csv_file):
    # 파일명에서 정보 추출
    filename = os.path.basename(csv_file)
    parts = filename.split('_')
    sentence_id = parts[1]
    video_info = '_'.join(parts[1:]).split('.')[0]

    # 참가자 번호 및 각도 정보 추출
    video_parts = video_info.split('_')
    participant_info = video_parts[3]
    angle_info = video_parts[4]
    participant_num = participant_info[-2:]

    # 출력 CSV 파일명 설정
    output_csv_filename = f"labeled_{filename}"
    output_csv_path = os.path.join(output_dir, output_csv_filename)
    temp_output_csv_path = output_csv_path + '.tmp'

    try:
        # 키포인트 데이터 로드
        keypoint_df = pd.read_csv(csv_file)
        if keypoint_df.empty:
            print(f"{csv_file} 파일이 비어 있습니다. 건너뜁니다.")
            return

        # 프레임 번호를 시간으로 변환
        keypoint_df['time'] = keypoint_df['frame_num'] / FPS

        # 해당하는 morpheme JSON 파일 로드
        morpheme_filename = f"{video_info}_morpheme.json"
        morpheme_file = os.path.join(morpheme_root, participant_num, morpheme_filename)

        if not os.path.exists(morpheme_file):
            print(f"{csv_file}에 대한 Morpheme 파일을 찾을 수 없습니다.")
            return

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
            # 임시 파일로 저장
            labeled_keypoint_df.to_csv(temp_output_csv_path, index=False)
            # 임시 파일을 최종 파일로 변경
            os.rename(temp_output_csv_path, output_csv_path)
            # print(f"Labeled data saved to {output_csv_path}")
        else:
            print(f"{csv_file}에 대한 레이블된 데이터가 없습니다.")
    except pd.errors.EmptyDataError:
        print(f"{csv_file} 파일이 비어 있어 건너뜁니다.")
    except Exception as e:
        print(f"{csv_file} 처리 중 오류 발생: {e}")
        # 임시 파일이 남아 있으면 삭제
        if os.path.exists(temp_output_csv_path):
            os.remove(temp_output_csv_path)

if __name__ == '__main__':
    num_processes = 16  # 시스템의 CPU 코어 수에 맞게 설정

    with Pool(processes=num_processes) as pool:
        for _ in tqdm(pool.imap_unordered(process_csv_file, csv_files), total=len(csv_files)):
            pass

    print("모든 파일 처리가 완료되었습니다.")