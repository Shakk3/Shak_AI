import tensorflow as tf
from tensorflow.keras import layers, models

# 간단한 CNN 모델 정의
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 클래스 수에 맞게 변경
])

# 모델 컴파일
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 모델 저장
model.save('/Users/donggun/Desktop/cnn 모델/my_new_model.h5')  # 원하는 경로로 저장
