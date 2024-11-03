import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_label_distribution(label_distribution):
    plt.figure(figsize=(8, 6))
    plt.bar(label_distribution.keys(), label_distribution.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Label Distribution')
    plt.show()

def plot_training_history(history):
    # 학습 과정 시각화
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

    # 그래프 간격 조정
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, labels):
    """
    Confusion matrix 시각화
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()