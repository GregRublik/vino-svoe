"""Репозиторий карточек вин в PostgreSQL через SQLAlchemy."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from threading import Lock
from typing import Any

from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.database import Base
from db.models import WineCatalogRow
from exceptions import CatalogUnavailableError
from schemas.search import SearchResponse, WineCard
from utils.links import build_wine_link, extract_wine_slug


class WineCatalogRepository:
    """Читает и заменяет каталог только в подключённой SQL-базе."""

    def __init__(self, database_url: str) -> None:
        if not database_url.strip():
            raise CatalogUnavailableError("Не задан URL базы данных каталога")

        self._engine = self._create_engine(database_url)
        self._session_factory = sessionmaker(
            self._engine,
            expire_on_commit=False,
        )
        self._initialized = False
        self._init_lock = Lock()

    @staticmethod
    def _create_engine(database_url: str) -> Engine:
        kwargs: dict[str, Any] = {"pool_pre_ping": True}
        if database_url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in database_url:
                kwargs["poolclass"] = StaticPool
        return create_engine(database_url, **kwargs)

    def _ensure_local_schema(self) -> None:
        # SQLite используется только в тестах и локальных проверках. В рабочем
        # окружении PostgreSQL схема создаётся Alembic-миграциями.
        if self._engine.dialect.name == "sqlite":
            Base.metadata.create_all(self._engine)

    @staticmethod
    def _first_value(record: Mapping[str, Any], *fields: str) -> str | None:
        for field in fields:
            value = record.get(field)
            if value is None:
                continue
            value = str(value).strip()
            if value:
                return value
        return None

    @classmethod
    def _row_from_record(
        cls,
        record: Mapping[str, Any],
        link: str | None = None,
    ) -> WineCatalogRow:
        slug = cls._first_value(record, "Slug", "slug")
        if not slug:
            raise ValueError("У записи каталога отсутствует Slug")

        photo_name = cls._first_value(record, "Название фото", "photo_name")
        return WineCatalogRow(
            slug=slug,
            name=cls._first_value(record, "Название вина", "name"),
            winery=cls._first_value(record, "Винодельня", "winery"),
            region=cls._first_value(record, "Регион", "region"),
            grape_variety=cls._first_value(
                record,
                "Сорт винограда",
                "grape_variety",
            ),
            category=cls._first_value(record, "Категория", "category"),
            color=cls._first_value(record, "Цвет", "color"),
            description=cls._first_value(record, "Описание", "description"),
            photo_name=photo_name,
            link=cls._first_value(record, "link") or link or build_wine_link(slug),
        )

    @staticmethod
    def _record_from_row(row: WineCatalogRow) -> dict[str, str | None]:
        return {
            "Slug": row.slug,
            "Название вина": row.name,
            "Винодельня": row.winery,
            "Регион": row.region,
            "Сорт винограда": row.grape_variety,
            "Категория": row.category,
            "Цвет": row.color,
            "Описание": row.description,
            "Название фото": row.photo_name,
            "link": row.link,
        }

    @staticmethod
    def _photo_link(
        record: Mapping[str, Any],
        links: Mapping[str, str | None],
    ) -> str | None:
        photo_name = WineCatalogRepository._first_value(
            record,
            "Название фото",
            "photo_name",
        )
        if not photo_name:
            return None
        normalized_name = photo_name.replace("\\", "/")
        return links.get(photo_name) or links.get(Path(normalized_name).name)

    def count_records(self) -> int:
        """Возвращает число записей без требования непустого каталога."""
        self._ensure_local_schema()
        try:
            with self._session_factory() as session:
                return int(
                    session.scalar(select(func.count()).select_from(WineCatalogRow))
                    or 0
                )
        except SQLAlchemyError as exc:
            raise CatalogUnavailableError(
                "Не удалось проверить каталог в базе данных"
            ) from exc

    def count_records_without_links(self) -> int:
        """Считает строки без сохранённой ссылки на карточку вина."""
        self._ensure_local_schema()
        try:
            with self._session_factory() as session:
                return int(
                    session.scalar(
                        select(func.count())
                        .select_from(WineCatalogRow)
                        .where(WineCatalogRow.link.is_(None))
                    )
                    or 0
                )
        except SQLAlchemyError as exc:
            raise CatalogUnavailableError(
                "Не удалось проверить полноту каталога в базе данных"
            ) from exc

    def replace_records(
        self,
        records: Mapping[str, Mapping[str, Any]],
        links: Mapping[str, str | None] | None = None,
    ) -> int:
        """Атомарно заменяет каталог импортированными записями."""
        links = links or {}
        rows_by_slug: dict[str, WineCatalogRow] = {}
        try:
            for key, record in records.items():
                normalized_record = dict(record)
                normalized_record.setdefault("Slug", key)
                row = self._row_from_record(
                    normalized_record,
                    link=self._photo_link(normalized_record, links),
                )
                rows_by_slug[row.slug] = row
        except (TypeError, ValueError) as exc:
            raise CatalogUnavailableError(
                "Импортируемый каталог содержит некорректную запись"
            ) from exc

        if not rows_by_slug:
            raise CatalogUnavailableError("Импортируемый каталог пуст")

        self._ensure_local_schema()
        try:
            with self._session_factory.begin() as session:
                session.execute(delete(WineCatalogRow))
                session.add_all(list(rows_by_slug.values()))
        except SQLAlchemyError as exc:
            raise CatalogUnavailableError(
                "Не удалось сохранить каталог в базе данных"
            ) from exc

        self._initialized = True
        return len(rows_by_slug)

    def initialize(self) -> None:
        """Проверяет доступность непустого каталога в базе данных."""
        with self._init_lock:
            if self._initialized:
                return

            if self.count_records() == 0:
                raise CatalogUnavailableError("Каталог вин в базе данных пуст")
            self._initialized = True

    @staticmethod
    def _card_from_row(row: WineCatalogRow) -> WineCard:
        return WineCard(
            slug=row.slug,
            name=row.name,
            winery=row.winery,
            region=row.region,
            grape_variety=row.grape_variety,
            category=row.category,
            color=row.color,
            description=row.description,
            link=row.link or build_wine_link(row.slug),
        )

    @staticmethod
    def _normalize_slugs(slugs: Iterable[str | None]) -> set[str]:
        return {
            slug.strip().strip("/")
            for slug in slugs
            if isinstance(slug, str) and slug.strip().strip("/")
        }

    def get_cards(self, slugs: Iterable[str | None]) -> dict[str, WineCard]:
        normalized_slugs = self._normalize_slugs(slugs)
        if not normalized_slugs:
            return {}

        self.initialize()
        try:
            with self._session_factory() as session:
                rows = session.scalars(
                    select(WineCatalogRow).where(
                        WineCatalogRow.slug.in_(normalized_slugs)
                    )
                ).all()
            return {row.slug: self._card_from_row(row) for row in rows}
        except SQLAlchemyError as exc:
            raise CatalogUnavailableError(
                "Не удалось прочитать карточки вин из базы данных"
            ) from exc

    def get_records(self) -> dict[str, dict[str, str | None]]:
        """Возвращает профиль каталога для офлайн-индексации из базы данных."""
        self.initialize()
        try:
            with self._session_factory() as session:
                rows = session.scalars(
                    select(WineCatalogRow).order_by(WineCatalogRow.slug)
                ).all()
            return {row.slug: self._record_from_row(row) for row in rows}
        except SQLAlchemyError as exc:
            raise CatalogUnavailableError(
                "Не удалось прочитать каталог из базы данных"
            ) from exc

    def enrich_response(self, response: SearchResponse) -> SearchResponse:
        result_slugs = [
            extract_wine_slug(
                content=result.content,
                metadata=result.metadata,
                link=result.link,
            )
            for result in response.results
        ]
        cards = self.get_cards(result_slugs)
        enriched_results = []
        for result, slug in zip(response.results, result_slugs, strict=True):
            card = cards.get(slug) if slug else None
            enriched_results.append(result.model_copy(update={"card": card}))
        top_card = None
        if result_slugs:
            top_card = cards.get(result_slugs[0]) if result_slugs[0] else None
        return response.model_copy(
            update={
                "results": enriched_results,
                # Нельзя сообщать о точном совпадении, если Qdrant вернул
                # устаревший или отсутствующий в каталоге slug.
                "found": response.found and top_card is not None,
            }
        )

    def close(self) -> None:
        self._engine.dispose()
