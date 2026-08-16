import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# ============================================================
# 1. Настройки
# ============================================================

IMAGE_DIR = "data/images"

OUTPUT_EMBEDDINGS = "wine_embeddings_siglip2_large.npy"
OUTPUT_NAMES = "wine_names_siglip2_large.txt"

# Большая модель SigLIP 2
MODEL_NAME = "google/siglip2-large-patch16-384"

# Для RTX 3050 8 GB начинаем с 4
BATCH_SIZE = 4

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Используется устройство: {device}")

if device.type == "cuda":
    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

# ============================================================
# 2. Загрузка SigLIP 2 Large
# ============================================================

print(
    f"\nЗагрузка модели {MODEL_NAME}..."
)

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(device)

model.eval()

print("Модель загружена.")

# ============================================================
# 3. Список изображений
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
)

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
]

if not image_files:
    print(
        f"В папке '{IMAGE_DIR}' "
        f"не найдено подходящих изображений!"
    )
    exit()

print(
    f"Найдено изображений: "
    f"{len(image_files)}"
)

# ============================================================
# 4. Извлечение эмбеддингов
# ============================================================

all_embeddings = []
processed_names = []

for start in tqdm(
    range(
        0,
        len(image_files),
        BATCH_SIZE
    ),
    desc="Извлечение эмбеддингов"
):

    batch_files = image_files[
        start:start + BATCH_SIZE
    ]

    images = []
    valid_names = []

    # --------------------------------------------------------
    # Загружаем изображения
    # --------------------------------------------------------

    for file_name in batch_files:

        image_path = os.path.join(
            IMAGE_DIR,
            file_name
        )

        try:

            image = Image.open(
                image_path
            ).convert("RGB")

            images.append(image)
            valid_names.append(file_name)

        except Exception as e:

            print(
                f"\nОшибка открытия "
                f"{file_name}: {e}"
            )

    if not images:
        continue

    try:

        # ----------------------------------------------------
        # Preprocessing SigLIP 2
        # ----------------------------------------------------

        inputs = processor(
            images=images,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Получаем image embeddings
        # ----------------------------------------------------

        with torch.inference_mode():

            vision_outputs = model.vision_model(
                pixel_values=inputs["pixel_values"]
            )

            image_features = (
                vision_outputs.pooler_output
            )

            # ------------------------------------------------
            # L2 normalization
            # ------------------------------------------------

            image_features = (
                image_features
                / image_features.norm(
                    p=2,
                    dim=-1,
                    keepdim=True
                )
            )

        # ----------------------------------------------------
        # Сохраняем batch
        # ----------------------------------------------------

        embeddings_np = (
            image_features
            .float()
            .cpu()
            .numpy()
        )

        all_embeddings.append(
            embeddings_np
        )

        processed_names.extend(
            valid_names
        )

    except torch.cuda.OutOfMemoryError:

        print(
            f"\nCUDA OOM на batch "
            f"{start}-{start + len(batch_files)}."
        )

        print(
            "Попробуй уменьшить "
            "BATCH_SIZE до 2 или 1."
        )

        torch.cuda.empty_cache()

        break

    except Exception as e:

        print(
            f"\nОшибка batch "
            f"{start}-{start + len(batch_files)}: "
            f"{e}"
        )

# ============================================================
# 5. Сохранение
# ============================================================

if all_embeddings:

    embeddings_matrix = np.concatenate(
        all_embeddings,
        axis=0
    )

    np.save(
        OUTPUT_EMBEDDINGS,
        embeddings_matrix
    )

    with open(
        OUTPUT_NAMES,
        "w",
        encoding="utf-8"
    ) as f:

        for name in processed_names:
            f.write(
                f"{name}\n"
            )

    print("\n==============================")
    print("Готово!")
    print("==============================")

    print(
        f"Изображений обработано: "
        f"{len(processed_names)}"
    )

    print(
        f"Матрица: "
        f"{OUTPUT_EMBEDDINGS}"
    )

    print(
        f"Имена: "
        f"{OUTPUT_NAMES}"
    )

    print(
        f"Размерность embedding: "
        f"{embeddings_matrix.shape}"
    )

else:

    print(
        "\nНе удалось получить "
        "ни одного embedding."
    )