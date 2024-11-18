import os
import sys
import numpy as np
import pickle
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # TensorFlow 정보 출력 끄기
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
ssrc_path = os.path.join(project_root, 'src')

sys.path.append(ssrc_path)
from data_preprocessing import load_data, encode_labels
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
font_prop = font_manager.FontProperties(fname=font_path)
rc('font', family=font_prop.get_name())
mpl.rcParams['axes.unicode_minus'] = False

# 경로 설정
model_dir = 'models/'
data_dir = 'data/labeled_keypoints/'

# 저장된 모델 및 라벨 인코더 불러오기
model_path = os.path.join(model_dir, 'sign_language_CNN_LSTM_model.h5')
label_encoder_path = os.path.join(model_dir, 'label_encoder.pkl')

model = load_model(model_path)
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

print(f"모델이 로드되었습니다: {model_path}")
print(f"라벨 인코더가 로드되었습니다: {label_encoder_path}")

# 평가 데이터 불러오기 (데이터 분리 없이 전체 데이터 사용)
X, labels = load_data(data_dir)
y, _ = encode_labels(labels)  # 라벨 인코더 재사용
X_val, y_val = X, y  # 평가 데이터로 설정

# 평가 수행
y_pred = np.argmax(model.predict(X_val), axis=1)
y_true = np.argmax(y_val, axis=1)

# Classification Report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred, normalize='true')

# Confusion Matrix 시각화
plt.figure(figsize=(15, 15))
sns.heatmap(
    cm, 
    annot=True, 
    fmt='.2f', 
    cmap='Blues', 
    xticklabels=label_encoder.classes_, 
    yticklabels=label_encoder.classes_,
    annot_kws={'size': 10}
    )
plt.title('Confusion Matrix', fontproperties=font_prop, fontsize=16)
plt.xlabel('Predicted', fontproperties=font_prop, fontsize=14)
plt.ylabel('True', fontproperties=font_prop, fontsize=14)
plt.xticks(rotation=45, fontproperties=font_prop)
plt.yticks(rotation=0, fontproperties=font_prop)

# 파일로 저장
output_path = 'output/confusion_matrix.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Confusion Matrix가 저장되었습니다: {output_path}")
plt.close()
