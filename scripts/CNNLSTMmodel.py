# main.py - Main script to execute the CNN + LSTM model
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
import pandas as pd
from config import *
from data_processing import *
from build_model import build_cnn_lstm_model
from train import train_model, evaluate_model
from visualization import *
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report

# Define directory path for CSV files
data_dir_path = '/srv/asura/SignLanguageModel/data/labeled_keypoints'

# Process data for model input
X_train, X_test, y_categorical, y_train, y_test, label_encoder, num_keypoints = load_and_process_data(data_dir_path, sequence_length=30)

print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')


# 레이블 분포 확인
unique, counts = np.unique(np.argmax(y_categorical, axis=1), return_counts=True)
label_distribution = dict(zip(label_encoder.classes_, counts))
print(f'Label distribution: {label_distribution}')

# 시각화
plot_label_distribution(label_distribution)

if len(label_distribution) < 2:
    print("Error: 데이터셋에 두 개 이상의 클래스가 필요합니다. 모든 샘플이 동일한 클래스입니다.")
    exit()

# Build model
input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])  # Adjust shape as per your data requirements
model = build_cnn_lstm_model(input_shape, y_categorical)

# 클래스 가중치 계산 (불균형 시)
y_integers = np.argmax(y_categorical, axis=1)  # y_categorical은 인코딩된 레이블 정보
class_weights_values = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights = dict(enumerate(class_weights_values))
print(f'Class weights: {class_weights}')

# Train model
history = train_model(model, X_train, y_train, X_test, y_test, class_weights)

plot_training_history(history)

# Evaluate model
evaluate_model(model, X_test, y_test)

# 모델 저장
model.save('/srv/asura/SignLanguageModel/models/sign_language_cnn_lstm_model.h5')

# 혼동행렬 시각화
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, target_names=label_encoder.classes_))

plot_confusion_matrix(y_true_labels, y_pred_labels, label_encoder.classes_)