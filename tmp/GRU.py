import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Masking, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# 데이터 폴더 경로
data_folder = 'data/labeled_keypoints'  # 실제 경로로 수정하세요.

# 모든 CSV 파일 읽기
all_dataframes = []
for file_name in os.listdir(data_folder):
    if file_name.endswith('.csv'):
        file_path = os.path.join(data_folder, file_name)
        df = pd.read_csv(file_path)
        all_dataframes.append(df)

# 데이터 병합
combined_data = pd.concat(all_dataframes, ignore_index=True)

# 시퀀스별로 데이터 정리
grouped = combined_data.groupby('frame_num')
sequence_data = []
sequence_labels = []

for _, group in grouped:
    sequence_data.append(group[['x', 'y', 'z']].values)
    sequence_labels.append(group['label'].iloc[0])  # 각 시퀀스의 첫 번째 라벨 사용

# 시퀀스 길이 패딩 (최대 길이에 맞추어 패딩)
max_seq_length = max(len(seq) for seq in sequence_data)
X = pad_sequences(sequence_data, maxlen=max_seq_length, dtype='float32', padding='post', value=0.0)
y = pd.get_dummies(sequence_labels).values  # One-hot 인코딩

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# GRU 모델 생성
model = Sequential()
model.add(Masking(mask_value=0.0, input_shape=(None, 3)))  # 시퀀스 길이를 None으로 설정
model.add(GRU(64, return_sequences=True))  # 첫 번째 GRU 레이어
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(GRU(64, return_sequences=False))  # 두 번째 GRU 레이어
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(y.shape[1], activation='softmax'))  # 출력층, 분류를 위한 softmax 사용

# 학습률 조정된 옵티마이저 설정
optimizer = Adam(learning_rate=0.0001)

# 모델 컴파일
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# 콜백 함수 설정
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True)
]

# 데이터 배치 처리 시 가변 길이 지원
def generator(X, y, batch_size=32):
    while True:
        indices = np.random.permutation(len(X))
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            batch_indices = indices[start:end]
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]
            yield X_batch, y_batch

# 모델 학습
batch_size = 32
steps_per_epoch = len(X_train) // batch_size
history = model.fit(generator(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=100,  # 더 많은 epoch으로 학습
                    validation_data=generator(X_test, y_test, batch_size=batch_size),
                    validation_steps=len(X_test) // batch_size,
                    callbacks=callbacks)

# 모델 평가
test_loss, test_accuracy = model.evaluate(generator(X_test, y_test, batch_size=batch_size), steps=len(X_test) // batch_size)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

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

# 모델 불러오기 (필요시)
# model = load_model('best_model.h5')
