"""Импорт исходного JSON-каталога в PostgreSQL при запуске приложения."""

import json
import os
from pathlib import Path

from config import resolve_project_path, settings
from repositories.catalog import WineCatalogRepository
from utils.catalog import load_catalog


def _catalog_path() -> Path:
    return resolve_project_path(settings.catalog_path)


def _links_path() -> Path:
    return resolve_project_path(settings.links_path)


def _load_links() -> dict[str, str | None]:
    with _links_path().open(encoding="utf-8") as file:
        links = json.load(file)
    if not isinstance(links, dict):
        raise TypeError("Файл ссылок должен содержать JSON-объект")
    return links


def main(force: bool = False) -> int:
    repository = WineCatalogRepository(settings.database_url)
    try:
        current_count = repository.count_records()
        if (
            not force
            and current_count > 0
            and repository.count_records_without_links() == 0
        ):
            print(f"Каталог уже загружен в PostgreSQL: {current_count} записей")
            return current_count

        catalog = load_catalog(_catalog_path())
        count = repository.replace_records(catalog, _load_links())
        print(f"Каталог импортирован в PostgreSQL: {count} записей")
        return count
    finally:
        repository.close()


if __name__ == "__main__":
    force = os.getenv("APP_CATALOG_REIMPORT", "").lower() == "true"
    main(force=force)
