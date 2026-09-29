import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.database import Base
from db.models import WineCatalogRow
from exceptions import CatalogUnavailableError
from repositories.catalog import WineCatalogRepository
from schemas.search import SearchResponse, SearchResult
from utils.catalog import find_catalog_record, index_catalog_photos, load_catalog


def _repository(tmp_path, *rows: WineCatalogRow) -> WineCatalogRepository:
    database_url = f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add_all(rows)
        session.commit()
    engine.dispose()
    return WineCatalogRepository(database_url)


def test_catalog_photo_name_resolves_to_slug(tmp_path):
    path = tmp_path / "wine_catalog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Slug": "new-wine-2026",
                    "Название фото": "Новая этикетка.webp",
                }
            ],
        ),
        encoding="utf-8",
    )

    catalog = load_catalog(path)
    record = find_catalog_record(
        catalog,
        "Новая этикетка.webp",
        index_catalog_photos(catalog),
    )

    assert record["Slug"] == "new-wine-2026"


def test_catalog_photo_name_handles_windows_style_path(tmp_path):
    path = tmp_path / "wine_catalog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Slug": "windows-path-wine",
                    "Название фото": r"archive\\labels\\wine.webp",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    catalog = load_catalog(path)
    record = find_catalog_record(
        catalog,
        "wine.webp",
        index_catalog_photos(catalog),
    )

    assert record["Slug"] == "windows-path-wine"


def test_catalog_repository_builds_wine_card_from_database(tmp_path):
    repository = _repository(
        tmp_path,
        WineCatalogRow(
            slug="new-wine-2026",
            name="Новое вино",
            winery="Новая винодельня",
            region="Крым",
            grape_variety="Шардоне",
            category="Белое",
            color="Золотистый",
            description="Свежий и минеральный вкус.",
            photo_name="new-wine-2026.webp",
            link="https://vino-svoe.ru/wines/new-wine-2026",
        ),
    )

    card = repository.get_cards(["new-wine-2026"])["new-wine-2026"]

    assert card.name == "Новое вино"
    assert card.description == "Свежий и минеральный вкус."
    assert card.link == "https://vino-svoe.ru/wines/new-wine-2026"
    repository.close()


def test_catalog_repository_does_not_fallback_to_json_when_database_is_empty(tmp_path):
    repository = _repository(tmp_path)

    with pytest.raises(CatalogUnavailableError):
        repository.initialize()

    repository.close()


def test_catalog_repository_reads_card_after_reopening_database(tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}"
    repository = _repository(
        tmp_path,
        WineCatalogRow(
            slug="db-wine",
            name="Вино из базы",
            winery="Тестовая винодельня",
            region="Крым",
            grape_variety="Сира",
            category="Красное",
            color="Красный",
            description="Карточка загружена в SQLAlchemy.",
            photo_name="db-wine.webp",
            link="https://vino-svoe.ru/wines/db-wine",
        ),
    )
    repository.close()

    reopened = WineCatalogRepository(database_url)
    card = reopened.get_cards(["db-wine"])["db-wine"]

    assert card.name == "Вино из базы"
    assert card.description == "Карточка загружена в SQLAlchemy."
    reopened.close()


def test_catalog_repository_replaces_and_reads_imported_records(tmp_path):
    repository = _repository(tmp_path)

    count = repository.replace_records(
        {
            "imported-wine": {
                "Slug": "imported-wine",
                "Название вина": "Импортированное вино",
                "Название фото": r"archive\\labels\\imported.webp",
                "Описание": "Импортировано в PostgreSQL.",
            }
        },
        {"imported.webp": "https://vino-svoe.ru/wines/imported-wine"},
    )

    assert count == 1
    record = repository.get_records()["imported-wine"]
    assert record["Название фото"] == r"archive\\labels\\imported.webp"
    assert record["link"] == "https://vino-svoe.ru/wines/imported-wine"
    assert record["Описание"] == "Импортировано в PostgreSQL."
    repository.close()


def test_catalog_repository_enriches_known_results_and_keeps_unknown_results(
    tmp_path,
):
    repository = _repository(
        tmp_path,
        WineCatalogRow(
            slug="known-wine",
            name="Известное вино",
            description="Описание известного вина",
            link="https://vino-svoe.ru/wines/known-wine",
        ),
    )
    response = SearchResponse(
        found=False,
        results=[
            SearchResult(
                id=1,
                score=0.8,
                content={"slug": "known-wine"},
                metadata={},
                link="https://qdrant.example/known-wine",
            ),
            SearchResult(
                id=2,
                score=0.7,
                content={"slug": "missing-wine"},
                metadata={},
                link="https://vino-svoe.ru/wines/missing-wine",
            ),
        ],
    )

    enriched = repository.enrich_response(response)

    assert enriched.found is False
    assert enriched.results[0].card.name == "Известное вино"
    assert enriched.results[0].card.description == "Описание известного вина"
    assert enriched.results[0].card.link == "https://vino-svoe.ru/wines/known-wine"
    assert enriched.results[1].card is None
    repository.close()


def test_catalog_repository_downgrades_found_when_top_result_is_not_in_catalog(
    tmp_path,
):
    repository = _repository(
        tmp_path,
        WineCatalogRow(slug="known-wine", name="Известное вино"),
    )
    response = SearchResponse(
        found=True,
        results=[
            SearchResult(
                id=1,
                score=0.99,
                content={"slug": "stale-qdrant-slug"},
                metadata={},
                link="https://vino-svoe.ru/wines/stale-qdrant-slug",
            )
        ],
    )

    enriched = repository.enrich_response(response)

    assert enriched.found is False
    assert enriched.results[0].card is None
    repository.close()
