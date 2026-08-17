"""Индексация фото в Qdrant: SigLIP2-векторы → коллекция `siglip2-vectors`.

Запуск из корня репозитория:
    uv run python src/scripts/index_images.py

Upsert идемпотентен (id точки = sha256 от имени файла), повторный запуск
безопасен; уже загруженные файлы пропускаются (resume).
"""

import asyncio
import hashlib
import json
import sys
from pathlib import Path

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from config import settings
from repositories.qdrant import QdrantRepository
from services.embedding import EmbeddingService

REPO_ROOT = SRC_DIR.parent
IMAGES_DIR = REPO_ROOT / "data" / "images"
LINKS_PATH = REPO_ROOT / "data" / "embedings" / "wine_links.json"
BATCH_SIZE = 64


def stable_int_id(filename: str) -> int:
    """Стабильный числовой id точки из имени файла (встроенный hash() — salted, не годится)."""
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def load_links() -> dict[str, str | None]:
    with open(LINKS_PATH, encoding="utf-8") as f:
        return json.load(f)


async def get_indexed_filenames(repo: QdrantRepository, collection: str) -> set[str]:
    """Имена уже загруженных файлов (для resume)."""
    indexed: set[str] = set()
    offset = None
    while True:
        points, next_offset = await repo.client.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            if payload.get("filename"):
                indexed.add(payload["filename"])
        if next_offset is None:
            break
        offset = next_offset
    return indexed


async def main() -> None:
    links = load_links()
    files = sorted(IMAGES_DIR.glob("*.webp"))
    if not files:
        print(f"В '{IMAGES_DIR}' не найдено .webp файлов!")
        return

    embedding_service = EmbeddingService()
    dim = embedding_service.siglip_dim  # загружает модель (первый запуск — скачивание)
    repo = QdrantRepository(AsyncQdrantClient(url=settings.qdrant_url))

    collection = settings.qdrant_collection_siglip2
    await repo.create_collection(collection, dim)

    indexed = await get_indexed_filenames(repo, collection)
    pending = [p for p in files if p.name not in indexed]
    print(
        f"Всего файлов: {len(files)}, "
        f"уже проиндексировано: {len(indexed)}, осталось: {len(pending)}"
    )

    batch: list[PointStruct] = []
    for path in tqdm(pending, desc="Индексация"):
        vector = embedding_service.vectorize_photo(path.read_bytes())
        batch.append(
            PointStruct(
                id=stable_int_id(path.name),
                vector=vector,
                payload={"filename": path.name, "link": links.get(path.name)},
            )
        )
        if len(batch) >= BATCH_SIZE:
            await repo.upsert(collection, batch)
            batch.clear()

    if batch:
        await repo.upsert(collection, batch)

    print(f"Готово: коллекция '{collection}' содержит {len(files)} векторов")


if __name__ == "__main__":
    asyncio.run(main())
