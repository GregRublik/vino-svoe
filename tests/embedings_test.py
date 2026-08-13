import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

# 1. Настройка путей и устройства
TEST_IMAGE_PATH = "path_to_image.webp"  # Путь к картинке, которую хотим проверить
EMBEDDINGS_PATH = "data/embedings/wine_embeddings.npy"
NAMES_PATH = "data/embedings/wine_names.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Загрузка базы данных эмбеддингов и имен
print("Загрузка базы данных...")
database_embeddings = np.load(EMBEDDINGS_PATH)  # Матрица (N, 1024)

with open(NAMES_PATH, "r", encoding="utf-8") as f:
  wine_names = [line.strip() for line in f.readlines()]

# 3. Загрузка модели dinov2_vitl14
print("Загрузка модели DINOv2 Large...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)
model.eval()

# 4. Трансформация для тестового изображения
preprocess = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

# 5. Получение эмбеддинга тестового изображения
print(f"Обработка тестового изображения: {TEST_IMAGE_PATH}")
image = Image.open(TEST_IMAGE_PATH).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0).to(device)

with torch.no_grad():
  embedding = model(input_tensor)
  # L2-нормализация вектора
  embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)
  new_wine_vector = embedding.squeeze().cpu().numpy()

# 6. Поиск Топ-5 похожих через косинусное сходство
# Так как векторы уже нормализованы, скалярное произведение равно косинусному сходству
similarities = np.dot(database_embeddings, new_wine_vector)

# Получаем индексы 5 лучших совпадений (сортируем по возрастанию и берем последние 5)
top5_indices = np.argsort(similarities)[-5:][
    ::-1
]  # Разворачиваем, чтобы лучший был первым

# 7. Вывод результатов
print("\n--- Результаты поиска (Топ-5 похожих) ---")
for rank, idx in enumerate(top5_indices, start=1):
  name = wine_names[idx]
  score = similarities[idx]
  # Переводим косинусное сходство в проценты для наглядности
  print(f"{rank}. {name} — Сходство: {score:.2%}")
