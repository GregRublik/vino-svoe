import io
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from fastapi import UploadFile
from PIL import Image
from transformers import AutoModel, AutoProcessor

# Корень репозитория: src/services/retrieval.py -> 3 уровня вверх
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "embedings"

EMBEDDINGS_PATH = DATA_DIR / "wine_embeddings.npy"
NAMES_PATH = DATA_DIR / "wine_names.txt"
LINKS_PATHS = (
    DATA_DIR / "wine_links.json",
    REPO_ROOT / "wine_images_all" / "wine_links.json",
)

TOP_K = 5

# Та же модель, что в src/scripts/get_embedings.py (вектор 1024)
MODEL_NAME = "google/siglip2-large-patch16-384"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=1)
def _get_model():
    """SigLIP 2 Large. Грузится один раз."""
    return AutoModel.from_pretrained(MODEL_NAME).to(device).eval()


@lru_cache(maxsize=1)
def _get_processor():
    """Процессор SigLIP 2 (resize/normalize из коробки). Грузится один раз."""
    return AutoProcessor.from_pretrained(MODEL_NAME)


@lru_cache(maxsize=1)
def _get_database():
    """База эмбеддингов + имена вин + карта имя->ссылка. Грузится один раз."""
    database_embeddings = np.load(EMBEDDINGS_PATH)  # Матрица (N, 1024)

    with open(NAMES_PATH, "r", encoding="utf-8") as f:
        wine_names = [line.strip() for line in f]

    wine_links = {}
    for links_path in LINKS_PATHS:
        if links_path.exists():
            with open(links_path, "r", encoding="utf-8") as f:
                wine_links = json.load(f)
            break

    return database_embeddings, wine_names, wine_links


def _embed(image):
    """Эмбеддинг изображения через SigLIP 2 с L2-нормализацией."""
    inputs = _get_processor()(images=image, return_tensors="pt").to(device)
    with torch.inference_mode():
        features = _get_model().vision_model(pixel_values=inputs["pixel_values"]).pooler_output
    return (features / features.norm(p=2, dim=-1, keepdim=True)).squeeze().cpu().numpy()


class RetrievalService:

    def __init__(
            self,
            qdrant_repository=None,
            embedding_service=None,
            ocr_service=None,
            product_repository=None
    ):
        # Пока поиск идёт только по готовым эмбеддингам, зависимости не используются
        self.qdrant_repository = qdrant_repository
        self.embedding_service = embedding_service
        self.ocr_service = ocr_service
        self.product_repository = product_repository

    async def find_by_photo(self, file: UploadFile):
        """Поиск вина по фото (пока только по эмбеддингам)."""
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
        new_wine_vector = _embed(image)

        # Векторы уже нормализованы, поэтому скалярное произведение = косинусное сходство
        database_embeddings, wine_names, wine_links = _get_database()
        similarities = np.dot(database_embeddings, new_wine_vector)

        # Топ-K лучших (сортировка по возрастанию, потом разворот — лучший первым)
        top_indices = np.argsort(similarities)[-TOP_K:][::-1]

        return {
            "results": [
                {
                    "name": wine_names[idx],
                    "score": float(similarities[idx]),
                    "link": wine_links.get(wine_names[idx]),
                }
                for idx in top_indices
            ]
        }
