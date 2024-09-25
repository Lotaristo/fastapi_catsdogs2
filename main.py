from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image

# Загружаем модель
model = load_model('cat_dog_classifier.h5')

# Инициализация приложения FastAPI
app = FastAPI()

# Размеры изображения
IMG_WIDTH, IMG_HEIGHT = 224, 224

# Функция для предобработки изображения
def prepare_image(img):
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

# POST запрос для предсказания
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение файла из памяти
        contents = await file.read()
        img = Image.open(BytesIO(contents))

        # Проверяем исходные размеры изображения
        print(f"Original image size: {img.size}")

        # Предобработка изображения
        img = prepare_image(img)

        # Получаем предсказание
        prediction = model.predict(img)

        # Уверенность предсказания
        confidence = prediction[0][0]

        # Определение результата
        result = "Dog" if confidence > 0.9 else "Cat"

        return JSONResponse(content={"result": result, "confidence": float(confidence)})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
