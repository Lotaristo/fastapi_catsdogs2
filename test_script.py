import requests

# Путь к изображению для теста
image_path = "dog1.png"

url = "http://127.0.0.1:8000/predict/"

# Открытие изображения и отправка запроса
with open(image_path, "rb") as img_file:
    files = {"file": img_file}
    response = requests.post(url, files=files)

# Вывод ответа от сервера
print(response.json())
