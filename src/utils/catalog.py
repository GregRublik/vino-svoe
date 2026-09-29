"""Загрузка каталога и сопоставление его карточек с файлами изображений."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def load_catalog(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise TypeError("Каталог должен содержать JSON-массив")

    catalog: dict[str, dict] = {}
    for row_number, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise TypeError(f"Строка каталога {row_number} должна быть объектом")

        slug = str(row.get("Slug") or "").strip()
        if not slug:
            continue
        if slug in catalog:
            logger.warning("Повторный slug '%s' в строке каталога %s", slug, row_number)

        normalized_row = dict(row)
        normalized_row["Slug"] = slug
        catalog[slug] = normalized_row
    return catalog


def index_catalog_photos(catalog: dict[str, dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in catalog.values():
        # В каталоге может храниться путь, а в data/images нужен только basename.
        photo_value = record.get("Название фото")
        catalog_photo_name = photo_value.strip() if isinstance(photo_value, str) else ""
        if not catalog_photo_name:
            continue

        filename = Path(catalog_photo_name.replace("\\", "/")).name
        previous = indexed.get(filename)
        if previous is not None and previous.get("Slug") != record.get("Slug"):
            logger.warning(
                "Одинаковое имя фото '%s' у slug '%s' и '%s'; "
                "используется первая запись",
                filename,
                previous.get("Slug"),
                record.get("Slug"),
            )
            continue
        indexed[filename] = record
    return indexed


def find_catalog_record(
    catalog: dict[str, dict],
    filename: str,
    photo_index: dict[str, dict] | None = None,
) -> dict | None:
    path = Path(filename)
    if path.stem in catalog:
        return catalog[path.stem]
    index = photo_index if photo_index is not None else index_catalog_photos(catalog)
    return index.get(path.name)
