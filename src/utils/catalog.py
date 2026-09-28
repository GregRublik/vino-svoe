"""Загрузка каталога и сопоставление его карточек с файлами изображений."""

import json
from pathlib import Path


def load_catalog(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as file:
        rows = json.load(file)
    return {row["Slug"]: row for row in rows if row.get("Slug")}


def index_catalog_photos(catalog: dict[str, dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for record in catalog.values():
        photo_name = str(record.get("Название фото") or "").strip()
        if photo_name:
            indexed.setdefault(Path(photo_name).name, record)
    return indexed


def find_catalog_record(
    catalog: dict[str, dict],
    filename: str,
    photo_index: dict[str, dict] | None = None,
) -> dict | None:
    path = Path(filename)
    if path.stem in catalog:
        return catalog[path.stem]
    return (photo_index or index_catalog_photos(catalog)).get(path.name)

