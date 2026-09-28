import json
from pathlib import Path

from repositories.catalog import WineCatalogRepository
from schemas.search import SearchResponse, SearchResult
from utils.catalog import find_catalog_record, index_catalog_photos, load_catalog


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


def test_catalog_repository_builds_wine_card_from_catalog(tmp_path):
    path = tmp_path / "wine_catalog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Slug": "new-wine-2026",
                    "Название вина": "Новое вино",
                    "Винодельня": "Новая винодельня",
                    "Регион": "Крым",
                    "Сорт винограда": "Шардоне",
                    "Категория": "Белое",
                    "Цвет": "Золотистый",
                    "Описание": "Свежий и минеральный вкус.",
                    "Рейтинг Роскачества": 90,
                    "С чем подавать": "Рыба и морепродукты",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    card = WineCatalogRepository(Path(path)).get_card(
        "new-wine-2026",
        link="https://vino-svoe.ru/wines/new-wine-2026",
    )

    assert card is not None
    assert card.name == "Новое вино"
    assert card.description == "Свежий и минеральный вкус."
    assert card.rating == 90
    assert card.food_pairing == "Рыба и морепродукты"


def test_catalog_repository_enriches_known_results_and_keeps_unknown_results(tmp_path):
    path = tmp_path / "wine_catalog.json"
    path.write_text(
        json.dumps(
            [
                {
                    "Slug": "known-wine",
                    "Название вина": "Известное вино",
                    "Описание": "Описание известного вина",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = WineCatalogRepository(path)
    response = SearchResponse(
        found=False,
        results=[
            SearchResult(
                id=1,
                score=0.8,
                content={"slug": "known-wine"},
                metadata={},
                link="https://vino-svoe.ru/wines/known-wine",
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
    assert enriched.results[1].card is None
