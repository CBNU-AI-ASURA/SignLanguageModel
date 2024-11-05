import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# 디렉토리 내 모든 CSV 파일 불러오기
data_dir = 'data/labeled_keypoints/'  # 디렉토리 경로
all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]

# 모든 CSV 파일 병합
data_list = []
for file in all_files:
    df = pd.read_csv(file)
    data_list.append(df)

# 데이터 병합
data = pd.concat(data_list, ignore_index=True)

# 데이터 전처리
features = data[['x', 'y', 'z']]
labels = data['label']

# 라벨 인코딩
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)  # 문자열 라벨을 숫자로 변환

# 시계열 데이터 형태로 변환
sequence_length = 10  # 시계열 길이
x_data = []
y_data = []

for i in range(len(features) - sequence_length):
    x_data.append(features.iloc[i:i + sequence_length].values)
    y_data.append(labels[i + sequence_length])

x_data = np.array(x_data)
y_data = np.array(y_data)

# 데이터셋 나누기
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# 모델 구축
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, 3), return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))  # 클래스 수에 따라 출력층 설정

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=20, batch_size=64, validation_split=0.2)

# 모델 평가
model.evaluate(x_test, y_test)

# 라벨 복원 예시
print("예측 결과:", label_encoder.inverse_transform([np.argmax(pred) for pred in model.predict(x_test[:5])]))
