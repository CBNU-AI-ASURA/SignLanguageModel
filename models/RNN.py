import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# JSON 파일에서 label을 추출하고 csv 데이터와 결합하는 함수
def extract_label_from_json(json_file, csv_data):
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        label = json_data['data'][0]['attributes'][0]['name']
        
        # 라벨을 추가 (동일한 라벨을 전체 데이터에 추가)
        csv_data['label'] = label
        return csv_data
    except FileNotFoundError:
        print(f"JSON 파일 {json_file}이 존재하지 않습니다.")
        return None

# 데이터 디렉토리 및 파일 경로 설정
csv_dir = './dataset/'  # CSV 파일들이 저장된 경로
json_dir = './source/morpheme/'  # JSON 파일들이 저장된 경로

# CSV 파일 리스트 불러오기
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

features_list = []
labels_list = []

# 각 CSV 파일에 대해 JSON 파일로부터 레이블을 가져와 처리
for csv_file in csv_files:
    csv_path = os.path.join(csv_dir, csv_file)
    
    # landmarks_ 부분을 제거한 후 JSON 파일명 추출
    json_filename = csv_file.replace('landmarks_', '').replace('.csv', '_morpheme.json')
    json_path = os.path.join(json_dir, json_filename)
    
    # CSV 데이터 로드
    try:
        csv_data = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        print(f"CSV 파일 {csv_path}을 읽는 중 오류가 발생했습니다.")
        continue
    
    # JSON 파일에서 레이블을 추출하고 CSV 데이터와 결합
    csv_data_with_label = extract_label_from_json(json_path, csv_data)
    
    if csv_data_with_label is not None:
        # landmark_type과 index를 하나의 특징으로 만들고, frame_num을 시간 정보로 사용
        csv_data_with_label['landmark_feature'] = csv_data_with_label['landmark_type'] + '_' + csv_data_with_label['index'].astype(str)
        
        # x, y, z 좌표를 특징으로 사용
        features = csv_data_with_label[['x', 'y', 'z']].values
        
        # 문자열 데이터가 아닌 숫자만 추출
        try:
            features = features.astype(float)  # 숫자 데이터로 변환
            features_list.append(features)
            
            # 해당 CSV 파일의 모든 행에 동일한 라벨을 반복해서 추가
            labels_list.extend([csv_data_with_label['label'].iloc[0]] * len(csv_data_with_label))
        except ValueError as e:
            print(f"숫자 변환 오류: {e}")
            continue

# 모든 데이터를 배열로 변환
if features_list:
    features = np.concatenate(features_list, axis=0)
    
    # 라벨을 숫자로 변환 (필요 시 인코딩 과정 포함)
    labels = np.array(labels_list)

    # 데이터에 문자열이 포함되어 있는지 확인
    if labels.dtype == 'O':  # 'O'는 일반적으로 문자열을 나타냅니다.
        unique_labels = np.unique(labels)
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_map[label] for label in labels])
    
    # 데이터 전처리
    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features)

    # 훈련과 테스트 세트로 분리
    X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

    # RNN 입력 형식에 맞게 데이터 차원 조정 (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # RNN 모델 정의
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        tf.keras.layers.SimpleRNN(64, return_sequences=False),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # 분류 문제이므로 sigmoid 사용
    ])

    # 모델 컴파일
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # 분류 문제일 경우

    # 모델 학습
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # 모델 평가
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

    # 예측
    predictions = model.predict(X_test)
    print(predictions)
