from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, TimeDistributed, Dropout

def build_cnn_lstm_model(input_shape, y_categorical):
    inputs = Input(shape=input_shape)

    # 모델 구성 코드
    x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='relu'))(inputs)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Conv1D(filters=128, kernel_size=3, activation='relu'))(x)
    x = TimeDistributed(MaxPooling1D(pool_size=2))(x)
    x = TimeDistributed(Flatten())(x)
    
    # LSTM layers
    x = LSTM(128, return_sequences=False)(x)
    x = Dropout(0.5)(x)
    outputs = Dense(y_categorical.shape[1], activation='softmax')(x)
    
    # 모델 정의
    model = Model(inputs=inputs, outputs=outputs)
    
    # 모델 컴파일
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # 모델 요약 출력
    model.summary()
    return model
