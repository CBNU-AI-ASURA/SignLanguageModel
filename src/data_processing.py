import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.utils import to_categorical

def load_and_process_data(file_path, sequence_length=30):
    sorted_keypoints = get_unique_keypoints(file_path)
    num_keypoints = len(sorted_keypoints)
    csv_files = load_datas(file_path)
    sequences = []
    labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        segments = split_csv_into_segments(df)
        print(f'Processing file: {file}, Number of segments: {len(segments)}')

        for label, segment_df in segments:
            frame_data = []
            grouped = segment_df.groupby('frame_num')
            for frame_num, group in grouped: # frame_num, group
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

            # 시퀀스 길이를 고정 (예: 30프레임)
            if len(frame_data) < sequence_length:
                pad_length = sequence_length - len(frame_data)
                frame_data += [[0.0] * (num_keypoints * 3)] * pad_length
            else:
                frame_data = frame_data[:sequence_length]

            sequences.append(frame_data)
            labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    # 레이블 인코딩
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f'Encoded labels: {y_encoded}')
    
    # 클래스 수 확인
    num_classes = len(label_encoder.classes_)
    print(f'Number of classes: {num_classes}')

    # 다중 클래스 분류를 위해 원-핫 인코딩
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    print(f'y_categorical shape: {y_categorical.shape}')

    # 학습/검증 데이터 분리
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_categorical, test_size=0.2, random_state=42, stratify=y
    )

    # 데이터 정규화
    X_min, X_max = X_train.min(), X_train.max()
    X_train = (X_train - X_min) / (X_max - X_min)
    X_val = (X_val - X_min) / (X_max - X_min)

    X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
    X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))
    
    return X_train_cnn, X_val_cnn, y_categorical, y_train, y_val, label_encoder, num_keypoints

# 모든 CSV 파일에서 고유한 (landmark_type, index) 조합을 찾기
def get_unique_keypoints(data_dir):
    keypoints = set()
    for file in load_datas(data_dir):
        df = pd.read_csv(file)
        for _, row in df.iterrows():
            keypoints.add((row['landmark_type'], row['index']))
    # 정렬하여 일관된 순서 유지
    sorted_keypoints = sorted(list(keypoints))
    return sorted_keypoints

# CSV 파일을 여러 샘플로 분리하는 함수
def split_csv_into_segments(df):
    """
    주어진 데이터프레임을 레이블 변경 기준으로 여러 세그먼트로 분리합니다.
    """
    segments = []
    current_segment = []
    current_label = None

    # 데이터프레임을 시간순으로 정렬
    df_sorted = df.sort_values(by='time').reset_index(drop=True)

    for idx, row in df_sorted.iterrows():
        label = row['label']
        if current_label is None:
            current_label = label
            current_segment.append(row)
        elif label == current_label:
            current_segment.append(row)
        else:
            # 레이블이 변경되었을 때 현재 세그먼트를 저장하고 새 세그먼트 시작
            segments.append((current_label, pd.DataFrame(current_segment)))
            current_label = label
            current_segment = [row]
    
    # 마지막 세그먼트 저장
    if current_segment:
        segments.append((current_label, pd.DataFrame(current_segment)))
    return segments

def load_datas(csv_path):
    return [os.path.join(csv_path, file) for file in os.listdir(csv_path) if file.endswith('.csv')]

def balance_data(X, y):
    """
    데이터 불균형을 해결하기 위해 RandomOverSampler를 사용하여 데이터를 오버샘플링합니다.
    """
    ros = RandomOverSampler(random_state=42)
    X_reshaped = X.reshape((X.shape[0], -1))  # (samples, features)
    X_resampled, y_resampled = ros.fit_resample(X_reshaped, np.argmax(y, axis=1))
    X_resampled = X_resampled.reshape(-1, 30, X.shape[2])  # (samples, 30, features)
    y_resampled = to_categorical(y_resampled, num_classes=y.shape[1])
    return X_resampled, y_resampled