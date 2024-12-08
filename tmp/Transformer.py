import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
from matplotlib import font_manager
import json
from concurrent.futures import ProcessPoolExecutor

# -----------------------------
# 사용자 정의 레이어: GetItem
# -----------------------------
class GetItem(Layer):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index

    def call(self, inputs):
        return inputs[:, self.index, :]

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# -----------------------------
# 폰트 설정 (한글 깨짐 방지)
# -----------------------------
font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"  # 폰트 경로
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()
plt.rcParams['axes.unicode_minus'] = False  # 음수 기호 깨짐 방지

# -----------------------------
# 단일 CSV 파일 처리 함수
# -----------------------------
def process_single_csv(file_path, sequence_length=30):
    sequences, labels = [], []
    try:
        df = pd.read_csv(
            file_path,
            usecols=['frame_num', 'landmark_type', 'index', 'x', 'y', 'z', 'label'],
            dtype={
                'frame_num': 'int32',
                'landmark_type': 'category',
                'index': 'int8',
                'x': 'float32',
                'y': 'float32',
                'z': 'float32',
                'label': 'category'
            }
        )
        df = df.sort_values(by=['frame_num']).reset_index(drop=True)
        grouped = df.groupby('frame_num')

        current_sequence = []
        current_label = None

        for frame_num, group in grouped:
            label = group['label'].iloc[0]
            if current_label is None:
                current_label = label
            elif label != current_label:
                current_sequence = []
                current_label = label

            frame_data = {
                'pose': np.zeros((33, 3), dtype=np.float32),
                'left_hand': np.zeros((21, 3), dtype=np.float32),
                'right_hand': np.zeros((21, 3), dtype=np.float32)
            }
            for _, row in group.iterrows():
                landmark_type = row['landmark_type']
                index = row['index']
                coords = [row['x'], row['y'], row['z']]
                if landmark_type in frame_data:
                    frame_data[landmark_type][index] = coords

            concatenated_data = np.concatenate([
                frame_data['pose'],
                frame_data['left_hand'],
                frame_data['right_hand']
            ], axis=0)
            current_sequence.append(concatenated_data)

            if len(current_sequence) == sequence_length:
                sequences.append(np.array(current_sequence))
                labels.append(current_label)
                current_sequence = []
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return sequences, labels

# -----------------------------
# 병렬 데이터 처리 함수
# -----------------------------
def load_and_process_data(data_folder, sequence_length=30):
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    all_sequences, all_labels = [], []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_single_csv, csv_files, [sequence_length] * len(csv_files)),
                            desc="병렬 CSV 처리 중", total=len(csv_files)))

    for sequences, labels in results:
        all_sequences.extend(sequences)
        all_labels.extend(labels)

    return all_sequences, all_labels

# -----------------------------
# Transformer 모델 구축
# -----------------------------
def build_transformer_model(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(4):  # Transformer 블록 4개
        x = MultiHeadAttention(num_heads=8, key_dim=16)(x, x)
        x = LayerNormalization()(x)
        x = Dense(256, activation='relu')(x)
    x = GetItem(index=-1)(x)  # 마지막 시퀀스 토큰 선택
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)

# -----------------------------
# 클래스 이름 목록 저장
# -----------------------------
def save_class_names(label_names, output_path="class_names.json"):
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(label_names, f, ensure_ascii=False, indent=4)
        print(f"클래스 이름이 {output_path} 파일에 저장되었습니다.")
    except Exception as e:
        print(f"클래스 이름 저장 중 오류 발생: {e}")

# -----------------------------
# 메인 실행
# -----------------------------
if __name__ == "__main__":
    # 데이터 로드 및 전처리
    data_folder = 'data/labeled_keypoints/'  # 데이터 경로
    sequence_length = 30
    print("데이터 로딩 및 전처리 중...")
    all_sequences, all_labels = load_and_process_data(data_folder, sequence_length)

    # 라벨 인코딩
    print("라벨 인코딩 중...")
    y = pd.get_dummies(all_labels).values
    label_names = pd.get_dummies(all_labels).columns.tolist()

    # 클래스 이름 저장
    save_class_names(label_names)

    # 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_sequences).reshape(-1, sequence_length, 225),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"훈련 데이터 크기: {len(X_train)}, 테스트 데이터 크기: {len(X_test)}")

    # Transformer 모델 구축
    print("Transformer 모델 구축 중...")
    model = build_transformer_model(input_shape=(sequence_length, 225), num_classes=len(label_names))

    # 모델 컴파일
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)  # 그래디언트 클리핑 적용
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 학습
    print("모델 학습 중...")
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced", classes=np.unique(np.argmax(y_train, axis=1)), y=np.argmax(y_train, axis=1)
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=16,
        class_weight=class_weights
    )

    # 모델 저장
    model.save('transformer_sign_language_model.h5')
    print("모델이 transformer_sign_language_model.h5로 저장되었습니다.")

    # 결과 시각화
    print("결과 시각화 중...")
    plt.figure(figsize=(12, 5))

    # 정확도 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='훈련 정확도')
    plt.plot(history.history['val_accuracy'], label='검증 정확도')
    plt.title('정확도 변화', fontproperties=font_prop, fontsize=16)
    plt.xlabel('에포크', fontproperties=font_prop, fontsize=14)
    plt.ylabel('정확도', fontproperties=font_prop, fontsize=14)
    plt.legend(prop=font_prop)

    # 손실 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='훈련 손실')
    plt.plot(history.history['val_loss'], label='검증 손실')
    plt.title('손실 변화', fontproperties=font_prop, fontsize=16)
    plt.xlabel('에포크', fontproperties=font_prop, fontsize=14)
    plt.ylabel('손실', fontproperties=font_prop, fontsize=14)
    plt.legend(prop=font_prop)

    plt.tight_layout()
    plt.show()

    print("모든 작업이 완료되었습니다.")
