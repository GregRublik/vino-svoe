"""Индексация OCR-текста фото в Qdrant → коллекция `ocr-data-vectors`.

Запуск из корня репозитория:
    uv run --extra ocr python src/scripts/index_ocr.py              # OCR каждой фотки
    uv run python src/scripts/index_ocr.py --filenames-only         # без OCR, каталог + имена
    uv run python src/scripts/index_ocr.py --filenames-only --rebuild

Текст для векторизации — профиль товара из `wine_catalog.json` и, если включён,
результат OCR фото. При отсутствии профиля используется имя файла. Векторизация
та же, что и на стороне запроса (`EmbeddingService.vectorize_text`).
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
from utils.catalog import find_catalog_record, index_catalog_photos, load_catalog
from utils.ids import stable_int_id
from utils.links import build_wine_link, extract_wine_slug
from utils.ocr_text import build_catalog_text, build_index_text

REPO_ROOT = SRC_DIR.parent
IMAGES_DIR = REPO_ROOT / "data" / "images"
LINKS_PATH = REPO_ROOT / "data" / "embedings" / "wine_links.json"
CATALOG_PATH = REPO_ROOT / "data" / "embedings" / "wine_catalog.json"
BATCH_SIZE = 64
IMAGE_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}


def load_links() -> dict[str, str | None]:
    with open(LINKS_PATH, encoding="utf-8") as f:
        return json.load(f)


def catalog_only_records(
    files: list[Path], catalog: dict[str, dict], links: dict[str, str | None]
) -> list[tuple[str, dict, str, str]]:
    """Возвращает карточки каталога, для которых нет фото в локальном индексе.

    Формат записи: slug, профиль каталога, исходное имя фото и уникальное имя
    точки. Такие записи позволяют искать карточку по OCR, даже если картинку
    товара не удалось извлечь из архива.
    """
    indexed_names = {path.name for path in files}
    photo_index = index_catalog_photos(catalog)
    indexed_slugs = set()
    for path in files:
        record = find_catalog_record(catalog, path.name, photo_index)
        slug = record.get("Slug") if record else None
        indexed_slugs.add(
            slug
            or extract_wine_slug(
                content={"filename": path.name},
                link=links.get(path.name),
            )
        )
    used_names = set(indexed_names)
    records: list[tuple[str, dict, str, str]] = []

    for slug, record in catalog.items():
        if slug in indexed_slugs:
            continue

        source_filename = str(record.get("Название фото") or f"{slug}.webp").strip()
        point_filename = source_filename
        if point_filename in used_names:
            point_filename = f"{slug}.catalog"
        used_names.add(point_filename)
        records.append((slug, record, source_filename, point_filename))

    return records


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


async def main(filenames_only: bool = False, rebuild: bool = False) -> None:
    links = load_links()
    catalog = load_catalog(CATALOG_PATH)
    catalog_photos = index_catalog_photos(catalog)
    files = sorted(
        path
        for path in IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    catalog_records = catalog_only_records(files, catalog, links)
    if not files and not catalog_records:
        print(f"В '{IMAGES_DIR}' не найдено .webp файлов и каталог пуст!")
        return

    embedding_service = EmbeddingService()
    text_dim = len(embedding_service.vectorize_text("dimension probe"))
    ocr_service = OCRService()
    repo = QdrantRepository(AsyncQdrantClient(url=settings.qdrant_url))

    collection = settings.qdrant_collection_ocr
    if rebuild and await repo.collection_exists(collection):
        await repo.client.delete_collection(collection_name=collection)
    await repo.create_collection(collection, text_dim)

    indexed = await get_indexed_filenames(repo, collection)
    pending = [p for p in files if p.name not in indexed]
    pending_catalog = [
        record for record in catalog_records if record[3] not in indexed
    ]
    print(
        f"Всего файлов: {len(files)}, "
        f"карточек без фото: {len(catalog_records)}, "
        f"уже проиндексировано: {len(indexed)}, "
        f"осталось фото: {len(pending)}, карточек: {len(pending_catalog)}"
    )

    stats = {"ocr": 0, "catalog": 0, "filename": 0, "catalog_only": 0}
    batch: list[PointStruct] = []
    for path in tqdm(pending, desc="Индексация OCR"):
        if filenames_only:
            ocr_text = ""
        else:
            ocr_text = await ocr_service.text_detection_on_file(path.read_bytes())
        catalog_record = find_catalog_record(catalog, path.name, catalog_photos)
        catalog_text = build_catalog_text(catalog_record)
        text = build_index_text(ocr_text, path.name, catalog_text)
        if ocr_text.strip():
            stats["ocr"] += 1
        elif catalog_text:
            stats["catalog"] += 1
        else:
            stats["filename"] += 1

        vector = embedding_service.vectorize_text(text)
        link = links.get(path.name)
        slug = (
            catalog_record.get("Slug")
            if catalog_record
            else extract_wine_slug(content={"filename": path.name}, link=link)
        )
        batch.append(
            PointStruct(
                id=stable_int_id(path.name),
                vector=vector,
                payload={
                    "filename": path.name,
                    "slug": slug,
                    "link": link or build_wine_link(slug),
                    "text": text,
                },
            )
        )
        if len(batch) >= BATCH_SIZE:
            await repo.upsert(collection, batch)
            batch.clear()

    for slug, record, source_filename, point_filename in tqdm(
        pending_catalog, desc="Индексация карточек без фото"
    ):
        catalog_text = build_catalog_text(record)
        text = build_index_text("", point_filename, catalog_text)
        vector = embedding_service.vectorize_text(text)
        link = links.get(source_filename) or links.get(point_filename)
        batch.append(
            PointStruct(
                id=stable_int_id(point_filename),
                vector=vector,
                payload={
                    "filename": point_filename,
                    "source_filename": source_filename,
                    "slug": slug,
                    "link": link or build_wine_link(slug),
                    "text": text,
                    "catalog_only": True,
                },
            )
        )
        stats["catalog_only"] += 1
        if len(batch) >= BATCH_SIZE:
            await repo.upsert(collection, batch)
            batch.clear()

    if batch:
        await repo.upsert(collection, batch)

    print(
        f"Готово: коллекция '{collection}' содержит "
        f"{len(files) + len(catalog_records)} векторов "
        f"(ocr: {stats['ocr']}, каталог: {stats['catalog']}, "
        f"имя файла: {stats['filename']}, "
        f"карточки без фото: {stats['catalog_only']})"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--filenames-only",
        action="store_true",
        help="Без OCR: использовать профиль каталога и имя файла",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Пересоздать OCR-коллекцию перед индексацией",
    )
    args = parser.parse_args()
    asyncio.run(main(filenames_only=args.filenames_only, rebuild=args.rebuild))
