import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 데이터 폴더 경로
data_folder = 'data/labeled_keypoints/'  # 실제 경로로 수정하세요.

# 모든 CSV 파일 읽기
all_dataframes = []
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

# 데이터 병합
combined_data = pd.concat(all_dataframes, ignore_index=True)

# 시퀀스별로 데이터 정리 (landmark_type 추가)
grouped = combined_data.groupby('frame_num')

sequence_data = []
sequence_labels = []

for frame_num, group in grouped:
    frame_data = {
        'pose': np.zeros((33, 3)),  # pose는 33개의 점
        'left_hand': np.zeros((21, 3)),  # 왼손은 21개의 점
        'right_hand': np.zeros((21, 3)),  # 오른손은 21개의 점
    }

    for _, row in group.iterrows():
        landmark_type = row['landmark_type']
        index = int(row['index'])
        coords = [row['x'], row['y'], row['z']]

        if landmark_type in frame_data:
            frame_data[landmark_type][index] = coords

    concatenated_data = np.concatenate([frame_data['pose'], frame_data['left_hand'], frame_data['right_hand']], axis=0)
    sequence_data.append(concatenated_data)
    sequence_labels.append(group['label'].iloc[0])  # 각 시퀀스의 첫 번째 라벨 사용

# 시퀀스 길이 패딩
max_seq_length = max(len(seq) for seq in sequence_data)
X = pad_sequences(sequence_data, maxlen=max_seq_length, dtype='float32', padding='post', value=0.0)
y = pd.get_dummies(sequence_labels).values  # One-hot 인코딩
label_names = pd.get_dummies(sequence_labels).columns  # 클래스 이름 저장

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE 적용을 위해 다차원 데이터를 2차원으로 변환
X_train_flat = X_train.reshape(X_train.shape[0], -1)
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_flat, y_train)

# 다시 다차원 형태로 변환
X_resampled = X_resampled.reshape(X_resampled.shape[0], max_seq_length, 3)

# GRU 모델 생성
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, 3)))  # 시퀀스 길이를 None으로 설정
model.add(GRU(64, return_sequences=True))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GRU(64, return_sequences=False))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))  # 출력층

# 옵티마이저 설정
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 콜백 설정 (가장 좋은 모델을 저장)
callbacks = [
    ModelCheckpoint('bestGRU_model.keras', monitor='val_loss', save_best_only=True, save_weights_only=False)
]

# 모델 학습
batch_size = 32
history = model.fit(X_resampled, y_resampled, batch_size=batch_size, epochs=200,
                    validation_data=(X_test, y_test), callbacks=callbacks)

# 모델 저장 완료 후 불러오기
best_model = load_model('bestGRU_model.keras')
print("Best model loaded successfully.")

# 모델 평가 (베스트 모델로)
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test Accuracy (Best Model): {test_accuracy * 100:.2f}%")

# 예측 수행
y_pred = best_model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# 혼동 행렬 계산 및 시각화
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 분류 보고서 출력
print("Classification Report:")
print(classification_report(y_true_classes, y_pred_classes, target_names=label_names))

# 학습 기록 시각화
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
