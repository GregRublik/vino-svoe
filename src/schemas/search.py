from uuid import UUID

from pydantic import BaseModel

SearchResultId = int | str | UUID


class BaseSearchRequest(BaseModel):
    top_k: int = 5
    filters: dict | None = None
    collection: str


class WineCard(BaseModel):
    """Данные карточки вина, которые отображает клиент."""

    slug: str
    name: str | None = None
    winery: str | None = None
    region: str | None = None
    grape_variety: str | None = None
    category: str | None = None
    color: str | None = None
    description: str | None = None
    roskachestvo_rating: str | None = None
    serving_recommendation: str | None = None
    link: str | None = None


class SearchResult(BaseModel):
    id: SearchResultId
    score: float
    content: dict
    metadata: dict
    link: str | None = None
    card: WineCard | None = None


class VectorSearchRequest(BaseSearchRequest):
    vector: list[float]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    found: bool = True
    margin: float | None = None
    ocr_matches: int = 0


class EvaluatorResponse(BaseModel):
    """Минимальный контракт для автоматического оценочного скрипта."""

    slug: str | None = None


class PairingResponse(BaseModel):
    """Детерминированная рекомендация блюда для найденного вина."""

    slug: str
    recommendation: str
    rationale: str
