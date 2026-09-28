from pydantic import BaseModel
from typing import Optional


class BaseSearchRequest(BaseModel):
    top_k: int = 5
    filters: Optional[dict] = None
    collection: str


class WineCard(BaseModel):
    """Данные карточки вина, которые отображает клиент."""

    slug: str
    name: Optional[str] = None
    winery: Optional[str] = None
    region: Optional[str] = None
    grape_variety: Optional[str] = None
    category: Optional[str] = None
    color: Optional[str] = None
    description: Optional[str] = None
    rating: str | int | float | None = None
    food_pairing: Optional[str] = None
    link: Optional[str] = None


class SearchResult(BaseModel):
    id: int
    score: float
    content: dict
    metadata: dict
    link: Optional[str] = None
    card: Optional[WineCard] = None


class VectorSearchRequest(BaseSearchRequest):
    vector: list[float]


class SearchResponse(BaseModel):
    results: list[SearchResult]
    found: bool = True
    confidence: Optional[float] = None
    margin: Optional[float] = None
    ocr_matches: int = 0


class EvalPrediction(BaseModel):
    slug: str
