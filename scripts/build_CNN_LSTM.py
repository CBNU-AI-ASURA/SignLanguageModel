# Build_CNN_LSTM
# Exacute this script on Project root directory

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
ssrc_path = os.path.join(project_root, 'src')

sys.path.append(ssrc_path)
from data_preprocessing import load_data, encode_labels
from model_utils import build_model, train_model, evaluate_model, plot_training_history


# 데이터 경로 설정
data_dir = 'data/labeled_keypoints/'
model_dir = 'models/'

# 학습 데이터 불러오기
X, labels = load_data(data_dir)
y, label_encoder = encode_labels(labels)

# 학습/검증 데이터 분리
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 모델 생성
input_shape = (None, X_train.shape[2])
num_classes = y.shape[1]
model = build_model(input_shape, num_classes)

# 모델 학습
history = train_model(model, X_train, y_train, X_val, y_val)

# 모델 저장
os.makedirs(model_dir, exist_ok=True)  # 디렉토리 생성
model_path = os.path.join(model_dir, 'sign_language_CNN_LSTM_model.h5')
model.save(model_path)

# 라벨 인코더 저장 (필요 시)
import pickle
label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')
with open(label_encoder_path, 'wb') as f:
    pickle.dump(label_encoder, f)

print(f"모델이 저장되었습니다: {model_path}")
print(f"라벨 인코더가 저장되었습니다: {label_encoder_path}")