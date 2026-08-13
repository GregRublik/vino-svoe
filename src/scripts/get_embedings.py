import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm  # Библиотека для красивой полосы загрузки (pip install tqdm)

# 1. Настройка путей и устройства
IMAGE_DIR = "data/images"
OUTPUT_EMBEDDINGS = "wine_embeddings.npy"
OUTPUT_NAMES = "wine_names.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# 2. Загрузка модели dinov2_vitl14 (Размерность вектора: 1024)
print("Загрузка модели DINOv2 Large...")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device)
model.eval()

# 3. Трансформация изображений
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

# 4. Сбор списка изображений
valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
image_files = [
    f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(valid_extensions)
]

if not image_files:
  print(f"В папке '{IMAGE_DIR}' не найдено подходящих изображений!")
  exit()

all_embeddings = []
processed_names = []

print(f"Найдено изображений для обработки: {len(image_files)}")

# 5. Цикл обработки картинок
with torch.no_grad():
  for file_name in tqdm(image_files, desc="Извлечение эмбеддингов"):
    image_path = os.path.join(IMAGE_DIR, file_name)

    try:
      # Открываем и предобрабатываем
      image = Image.open(image_path).convert("RGB")
      input_tensor = preprocess(image).unsqueeze(0).to(device)

      # Получаем эмбеддинг
      embedding = model(input_tensor)

      # Нормализуем вектор (L2-нормализация)
      embedding = embedding / embedding.norm(p=2, dim=-1, keepdim=True)

      # Переводим в NumPy и сохраняем в список
      embedding_np = embedding.squeeze().cpu().numpy()
      all_embeddings.append(embedding_np)
      processed_names.append(file_name)

    except Exception as e:
      print(f"\nОшибка при обработке файла {file_name}: {e}")

# 6. Сохранение результатов
if all_embeddings:
  # Превращаем список векторов в одну матрицу формы (N, 1024)
  embeddings_matrix = np.stack(all_embeddings, axis=0)
  np.save(OUTPUT_EMBEDDINGS, embeddings_matrix)

  # Сохраняем имена файлов, чтобы знать какой вектор кому принадлежит
  with open(OUTPUT_NAMES, "w", encoding="utf-8") as f:
    for name in processed_names:
      f.write(f"{name}\n")

  print("\nГотово!")
  print(f"Матрица эмбеддингов сохранена в '{OUTPUT_EMBEDDINGS}'")
  print(f"Имена файлов сохранены в '{OUTPUT_NAMES}'")
  print(f"Итоговая размерность матрицы: {embeddings_matrix.shape}")
