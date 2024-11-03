from tensorflow.keras.callbacks import EarlyStopping

def train_model(model, X_train, y_train, X_val, y_val, class_weights=None):
    # 모델 학습 및 조기 종료 설정
    #print(f"X_train.dtypes: {}, y_train.dtypes: {y_train.dtypes}")
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val),class_weight=class_weights, callbacks=[early_stopping]) # 클래스 불균형시 class_weight 추가
    return history

def evaluate_model(model, X_test, y_test):
    # 모델 평가
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
