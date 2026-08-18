"""Индексация OCR-текста фото в Qdrant → коллекция `ocr-data-vectors`.

Запуск из корня репозитория:
    uv run --extra ocr python src/scripts/index_ocr.py              # OCR каждой фотки
    uv run python src/scripts/index_ocr.py --filenames-only         # без OCR, только имена файлов

Текст для векторизации — результат OCR фото (PaddleOCR); при пустом результате
фолбэк на имя файла (название вина и описание цвета). Векторизация — та же
текстовая модель, что и на стороне запроса (`EmbeddingService.vectorize_text`).
Id точек — те же, что в `siglip2-vectors` (иначе RRF не сольёт списки по id).

Upsert идемпотентен, уже загруженные файлы пропускаются (resume).
"""

import argparse
import asyncio
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
from services.ocr import OCRService
from utils.ids import stable_int_id
from utils.ocr_text import build_index_text

REPO_ROOT = SRC_DIR.parent
IMAGES_DIR = REPO_ROOT / "data" / "images"
LINKS_PATH = REPO_ROOT / "data" / "embedings" / "wine_links.json"
BATCH_SIZE = 64


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


async def main(filenames_only: bool = False) -> None:
    links = load_links()
    files = sorted(IMAGES_DIR.glob("*.webp"))
    if not files:
        print(f"В '{IMAGES_DIR}' не найдено .webp файлов!")
        return

    embedding_service = EmbeddingService()
    text_dim = len(embedding_service.vectorize_text("dimension probe"))
    ocr_service = OCRService()
    repo = QdrantRepository(AsyncQdrantClient(url=settings.qdrant_url))

    collection = settings.qdrant_collection_ocr
    await repo.create_collection(collection, text_dim)

    indexed = await get_indexed_filenames(repo, collection)
    pending = [p for p in files if p.name not in indexed]
    print(
        f"Всего файлов: {len(files)}, "
        f"уже проиндексировано: {len(indexed)}, осталось: {len(pending)}"
    )

    stats = {"ocr": 0, "fallback": 0}
    batch: list[PointStruct] = []
    for path in tqdm(pending, desc="Индексация OCR"):
        if filenames_only:
            ocr_text = ""
        else:
            ocr_text = await ocr_service.text_detection_on_file(path.read_bytes())
        text = build_index_text(ocr_text, path.name)
        if ocr_text.strip():
            stats["ocr"] += 1
        else:
            stats["fallback"] += 1

        vector = embedding_service.vectorize_text(text)
        batch.append(
            PointStruct(
                id=stable_int_id(path.name),
                vector=vector,
                payload={
                    "filename": path.name,
                    "link": links.get(path.name),
                    "text": text,
                },
            )
        )
        if len(batch) >= BATCH_SIZE:
            await repo.upsert(collection, batch)
            batch.clear()

    if batch:
        await repo.upsert(collection, batch)

    print(
        f"Готово: коллекция '{collection}' содержит {len(files)} векторов "
        f"(ocr: {stats['ocr']}, фолбэк на имя файла: {stats['fallback']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filenames-only",
        action="store_true",
        help="Без OCR: векторизовать только имена файлов (быстро)",
    )
    args = parser.parse_args()
    asyncio.run(main(filenames_only=args.filenames_only))
