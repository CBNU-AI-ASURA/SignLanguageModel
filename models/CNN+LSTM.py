import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, TimeDistributed, Dropout
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler  # 필요 시 설치: pip install imbalanced-learn
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import font_manager, rc
import seaborn as sns

# 한글 폰트 설정 (예: 맑은 고딕)
font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'  # Mac OS 예시
# Windows 사용자라면 'C:/Windows/Fonts/malgun.ttf' 등으로 변경
if os.path.exists(font_path):
    font = font_manager.FontProperties(fname=font_path).get_name()
    rc('font', family=font)
else:
    print("한글 폰트 파일을 찾을 수 없습니다. 시각화에서 한글이 제대로 표시되지 않을 수 있습니다.")

# 데이터 경로 설정
data_dir = 'data/labeled_keypoints/'

# 모든 CSV 파일에서 고유한 (landmark_type, index) 조합을 찾기
def get_unique_keypoints(data_dir):
    keypoints = set()
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    for file in csv_files:
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

# 데이터 로딩 및 전처리
def load_and_preprocess_data(data_dir, sequence_length=30):
    sorted_keypoints = get_unique_keypoints(data_dir)
    num_keypoints = len(sorted_keypoints)
    csv_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.csv')]
    sequences = []
    labels = []

    for file in csv_files:
        df = pd.read_csv(file)
        segments = split_csv_into_segments(df)
        print(f'Processing file: {file}, Number of segments: {len(segments)}')
        
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
            
            # 시퀀스 길이를 고정 (예: 30프레임)
            if len(frame_data) < sequence_length:
                pad_length = sequence_length - len(frame_data)
                frame_data += [ [0.0] * (num_keypoints * 3) ] * pad_length
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
    
    return X, y_categorical, label_encoder, num_keypoints

# 데이터셋 불균형 해결을 위한 오버샘플링
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

# 데이터 불러오기
X, y, label_encoder, num_keypoints = load_and_preprocess_data(data_dir)

# 레이블 분포 확인
unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
label_distribution = dict(zip(label_encoder.classes_, counts))
print(f'Label distribution: {label_distribution}')

# 시각화
plt.figure(figsize=(8, 6))
plt.bar(label_distribution.keys(), label_distribution.values(), color='skyblue')
plt.xlabel('Classes')
plt.ylabel('Counts')
plt.title('Label Distribution')
plt.show()

# 데이터셋이 단일 클래스일 경우, 모델 학습을 진행할 수 없습니다.
if len(label_distribution) < 2:
    print("Error: 데이터셋에 두 개 이상의 클래스가 필요합니다. 모든 샘플이 동일한 클래스입니다.")
    exit()

# 데이터 균형 맞추기 (필요 시)
# 현재 데이터셋은 균형 잡혀 있으므로 이 단계는 선택적입니다.
# X, y = balance_data(X, y)
# print("After balancing:")
# unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
# label_distribution = dict(zip(label_encoder.classes_, counts))
# print(f'Label distribution: {label_distribution}')


# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 데이터 정규화
X_min, X_max = X_train.min(), X_train.max()
X_train = (X_train - X_min) / (X_max - X_min)
X_val = (X_val - X_min) / (X_max - X_min)

# 입력 데이터 형태 조정
# Conv1D는 (steps_per_frame, channels) 형태를 기대하므로, 마지막 차원을 채널로 설정
X_train_cnn = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2], 1))
X_val_cnn = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2], 1))

# 모델 구축 (함수형 API 사용)
input_shape = (X_train_cnn.shape[1], X_train_cnn.shape[2], X_train_cnn.shape[3])
inputs = Input(shape=input_shape)

x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(inputs)
x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
x = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'))(x)
x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
x = TimeDistributed(Flatten())(x)
x = LSTM(128, return_sequences=False)(x)
x = Dropout(0.5)(x)

# 다중 클래스 분류를 위한 출력층 설정
outputs = Dense(y.shape[1], activation='softmax')(x)

# 모델 정의
model = Model(inputs=inputs, outputs=outputs)

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 모델 요약 출력
model.summary()

# 클래스 가중치 계산 (불균형 시)
y_integers = np.argmax(y_train, axis=1)
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights = dict(enumerate(class_weights_values))
print(f'Class weights: {class_weights}')

# 조기 종료 콜백 설정 (과적합 방지)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 모델 학습
history = model.fit(
    X_train_cnn, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val_cnn, y_val),
    class_weight=class_weights,  # 클래스 불균형 시 적용
    callbacks=[early_stopping]
)

# 학습 결과 시각화
plt.figure(figsize=(12, 4))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.show()

# 모델 평가
loss, accuracy = model.evaluate(X_val_cnn, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

# 모델 저장
model.save('sign_language_cnn_lstm_model.h5')

# 추가적인 평가 지표 (선택 사항)
# Classification Report 및 Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_val_cnn)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

print("Confusion Matrix:")
cm = confusion_matrix(y_true, y_pred_classes)
print(cm)

# 혼동 행렬 시각화
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
