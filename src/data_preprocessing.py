# data_preprocessing.py
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical, pad_sequences
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def find_keypoints_in_file(file):
    keypoints = set()
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        keypoints.add((row['landmark_type'], row['index']))
    return keypoints

def get_unique_keypoints(base_dir):
    """
    데이터 디렉토리 내의 모든 CSV 파일에서 유일한 키포인트를 추출합니다.

    Args:
        base_dir (str): 최상위 데이터 디렉토리 경로 (예: 'data/').

    Returns:
        list: 정렬된 고유 키포인트 리스트.
    """
    # `01`부터 `20`까지 폴더 탐색
    folder_paths = [os.path.join(base_dir, f"{i:02d}") for i in range(1, 21)]
    csv_files = []

    # 각 폴더 내의 CSV 파일 수집
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            csv_files.extend([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
        else:
            print(f"폴더를 찾을 수 없습니다: {folder_path}")

    # 멀티프로세싱으로 키포인트 추출
    with Pool(cpu_count()) as pool:
        results = list(
            tqdm(
                pool.imap(find_keypoints_in_file, csv_files),
                total=len(csv_files),
                desc="Finding unique keypoints"
            )
        )

    # 고유 키포인트 집합으로 병합 및 정렬
    unique_keypoints = set().union(*results)
    sorted_keypoints = sorted(list(unique_keypoints))
    return sorted_keypoints

def split_csv_into_segments(df):
    segments = []
    current_segment = []
    current_label = None
    df_sorted = df.sort_values(by='time').reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        label = row['label']
        if current_label is None:
            current_label = label
            current_segment.append(row)
        elif label == current_label:
            current_segment.append(row)
        else:
            segments.append((current_label, pd.DataFrame(current_segment)))
            current_label = label
            current_segment = [row]
    if current_segment:
        segments.append((current_label, pd.DataFrame(current_segment)))
    return segments

def process_file(args):
    file, sorted_keypoints = args
    sequences = []
    labels = []
    df = pd.read_csv(file)
    segments = split_csv_into_segments(df)

    for label, segment_df in segments:
        frame_data = []
        grouped = segment_df.groupby('frame_num')
        for frame_num, group in grouped:
            frame_keypoints = {}
            for _, row in group.iterrows():
                key = (row['landmark_type'], row['index'])
                frame_keypoints[key] = [row['x'], row['y'], row['z']]
            
            frame_vector = []
            for key in sorted_keypoints:
                if key in frame_keypoints:
                    frame_vector.extend(frame_keypoints[key])
                else:
                    frame_vector.extend([0.0, 0.0, 0.0])
            frame_data.append(frame_vector)
        
        sequences.append(frame_data)
        labels.append(label)
    
    return sequences, labels

def load_data(base_dir):
    """
    폴더 구조에 맞게 데이터를 로드하고, 패딩된 시퀀스와 라벨을 반환합니다.

    Args:
        base_dir (str): 최상위 데이터 디렉토리 경로 (예: 'data/').

    Returns:
        np.array, list: 패딩된 시퀀스 데이터와 라벨 리스트.
    """
    # `01`부터 `20`까지 폴더 탐색
    folder_paths = [os.path.join(base_dir, f"{i:02d}") for i in range(1, 6)]
    csv_files = []

    # 폴더별로 CSV 파일 수집
    for folder_path in folder_paths:
        if os.path.exists(folder_path):
            csv_files.extend([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')])
        else:
            print(f"폴더를 찾을 수 없습니다: {folder_path}")

    # 키포인트 불러오기 (get_unique_keypoints는 기존 함수 활용)
    sorted_keypoints = get_unique_keypoints(base_dir)

    # 멀티프로세싱으로 CSV 파일 처리
    with Pool(14) as pool:
        results = list(
            tqdm(
                pool.imap(process_file, [(file, sorted_keypoints) for file in csv_files]),
                total=len(csv_files),
                desc="Processing files"
            )
        )

    # 결과를 병합
    sequences, labels = zip(*results)
    sequences = [seq for batch in sequences for seq in batch]
    labels = [label for batch in labels for label in batch]

    # 패딩된 시퀀스 반환
    return pad_sequences(sequences, dtype='float32', padding='post', value=0.0), labels

def encode_labels(labels):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    return to_categorical(y_encoded), label_encoder