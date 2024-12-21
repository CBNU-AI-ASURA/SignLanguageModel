import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Masking
)
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

# --- 데이터 처리 함수 ---
def process_single_file(file_path, sequence_length=90):
    """
    CSV 파일 처리 후 시퀀스 및 레이블 생성
    """
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

        # 데이터 정렬 및 초기화
        df = df.sort_values(by=['frame_num']).reset_index(drop=True)
        grouped = df.groupby('frame_num')

        sequences, labels = [], []
        current_sequence = []
        current_label = None

        for frame_num, group in grouped:
            label = group['label'].iloc[0]
            if label == 'background':  # background 클래스는 무시
                continue

            if current_label is None:
                current_label = label
            elif label != current_label:
                if len(current_sequence) > 0:
                    sequences.append(np.array(current_sequence))
                    labels.append(current_label)
                current_sequence = []
                current_label = label

            # 프레임 데이터를 초기화
            frame_data = np.zeros((75, 3), dtype=np.float32)
            for _, row in group.iterrows():
                index = row['index']
                coords = [row['x'], row['y'], row['z']]
                frame_data[index] = coords

            current_sequence.append(frame_data)

        if len(current_sequence) > 0:
            sequences.append(np.array(current_sequence))
            labels.append(current_label)

        return sequences, labels

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return [], []


def load_and_process_data(data_folder, sequence_length=90):
    """
    데이터 폴더에서 CSV 파일을 처리해 시퀀스 및 레이블 생성
    """
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    all_sequences, all_labels = [], []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(lambda f: process_single_file(f, sequence_length), csv_files), total=len(csv_files), desc="파일 처리 중"))
        for sequences, labels in results:
            all_sequences.extend(sequences)
            all_labels.extend(labels)

    return all_sequences, all_labels

# --- 모델 정의 ---
def transformer_block(inputs, num_heads, ff_dim, dropout_rate=0.1):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=inputs.shape[-1])(inputs, inputs)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(inputs + attention_output)

    ff_output = Dense(ff_dim, activation="relu")(attention_output)
    ff_output = Dense(inputs.shape[-1])(ff_output)
    ff_output = Dropout(dropout_rate)(ff_output)
    return LayerNormalization(epsilon=1e-6)(attention_output + ff_output)


def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=128, num_layers=2):
    inputs = Input(shape=input_shape)
    x = Masking(mask_value=0.0)(inputs)  # 패딩 무시
    for _ in range(num_layers):
        x = transformer_block(x, num_heads=num_heads, ff_dim=ff_dim)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(num_classes, activation="softmax")(x)
    return Model(inputs, outputs)

# --- 메인 실행 ---
def main():
    data_folder = r'/mnt/c/Users/SBS/Desktop/WorkSpace/SignLanguageModel/data/labeled_keypoints'
    sequence_length = 90

    # 데이터 로드 및 처리
    all_sequences, all_labels = load_and_process_data(data_folder, sequence_length)

    # background 클래스 제외
    all_sequences = [seq for seq, label in zip(all_sequences, all_labels) if label != 'background']
    all_labels = [label for label in all_labels if label != 'background']

    y = pd.get_dummies(all_labels).values
    label_names = pd.get_dummies(all_labels).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        np.array(all_sequences).reshape(-1, sequence_length, 225),
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 클래스 가중치 계산
    class_weights = class_weight.compute_class_weight(
        class_weight="balanced",
        classes=np.unique(np.argmax(y_train, axis=1)),
        y=np.argmax(y_train, axis=1)
    )
    class_weights = {i: w for i, w in enumerate(class_weights)}

    # Transformer 모델 생성
    model = build_transformer_model(input_shape=(sequence_length, 225), num_classes=len(label_names))
    optimizer = Adam(learning_rate=0.0001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # 모델 학습
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=10,
        batch_size=16,
        class_weight=class_weights
    )

    model.save('transformer_sign_language_model.h5')
    print("모델이 transformer_sign_language_model.h5로 저장되었습니다.")


if __name__ == "__main__":
    main()
