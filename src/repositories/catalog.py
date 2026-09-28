"""Репозиторий карточек вин из локального JSON-каталога."""

import logging
from pathlib import Path

from schemas.search import SearchResponse, SearchResult, WineCard
from utils.catalog import load_catalog
from utils.links import build_wine_link, extract_wine_slug

logger = logging.getLogger(__name__)


class WineCatalogRepository:
    """Лениво загружает каталог и обогащает результаты поиска карточками."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._catalog: dict[str, dict] | None = None

    def _get_catalog(self) -> dict[str, dict]:
        if self._catalog is None:
            try:
                self._catalog = load_catalog(self.path)
            except (OSError, KeyError, TypeError, ValueError) as exc:
                logger.warning("Не удалось загрузить каталог вин '%s': %s", self.path, exc)
                self._catalog = {}
        return self._catalog

    @staticmethod
    def _first_value(record: dict, *fields: str):
        for field in fields:
            value = record.get(field)
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            return value
        return None

    def get_card(self, slug: str | None, link: str | None = None) -> WineCard | None:
        normalized_slug = (slug or "").strip().strip("/")
        if not normalized_slug:
            return None

        record = self._get_catalog().get(normalized_slug)
        if record is None:
            return None

        return WineCard(
            slug=normalized_slug,
            name=self._first_value(record, "Название вина"),
            winery=self._first_value(record, "Винодельня"),
            region=self._first_value(record, "Регион"),
            grape_variety=self._first_value(record, "Сорт винограда"),
            category=self._first_value(record, "Категория"),
            color=self._first_value(record, "Цвет"),
            description=self._first_value(record, "Описание"),
            rating=self._first_value(
                record,
                "Рейтинг Роскачества",
                "Оценка Роскачества",
                "Рейтинг",
            ),
            food_pairing=self._first_value(
                record,
                "С чем подавать",
                "К чему подать",
                "Гастрономия",
                "Сочетание с едой",
            ),
            link=link or build_wine_link(normalized_slug),
        )

    def enrich_result(self, result: SearchResult) -> SearchResult:
        slug = extract_wine_slug(
            content=result.content,
            metadata=result.metadata,
            link=result.link,
        )
        return result.model_copy(
            update={"card": self.get_card(slug, link=result.link)}
        )

    def enrich_response(self, response: SearchResponse) -> SearchResponse:
        return response.model_copy(
            update={"results": [self.enrich_result(result) for result in response.results]}
        )
