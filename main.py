# import tensorflow as tf
# from tensorflow.keras import layers, models

# # 간단한 CNN 모델 정의
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     layers.MaxPooling2D(pool_size=(2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(10, activation='softmax')  # 클래스 수에 맞게 변경
# ])

# # 모델 컴파일
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # 모델 저장
# model.save('/Users/donggun/Desktop/cnn 모델/my_new_model.h5')  # 원하는 경로로 저장


from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = FastAPI()

# 업로드할 디렉토리 설정
upload = "./upload/"
os.makedirs(upload, exist_ok=True)

# 새로운 CNN 모델 로드
model = tf.keras.models.load_model('/Users/donggun/Desktop/cnn 모델/my_new_model.h5')  # 새로운 모델 경로로 수정

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    file_path = os.path.join(upload, file.filename)

    # 파일 저장
    with open(file_path, "wb") as uploadfile:
        uploadfile.write(await file.read())

    # 이미지 전처리
    image = Image.open(file_path)
    image = image.resize((224, 224))  # 모델에 맞는 크기로 조정
    image_array = np.array(image) / 255.0  # 정규화
    image_array = np.expand_dims(image_array, axis=0)  # 배치 차원 추가

    # 예측
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions, axis=1)

    return JSONResponse(content={"filename": file.filename, "file_path": file_path, "predicted_class": int(predicted_class[0])})

# FastAPI 서버 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
