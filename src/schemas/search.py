from pydantic import BaseModel
from typing import Optional

# from schemas.websearch import ExtractedDocument


class BaseSearchRequest(BaseModel):
    top_k: int = 5
    filters: Optional[dict] = None
    collection: str

class SearchRequest(BaseSearchRequest):
    query: str

class SearchResult(BaseModel):
    id: int
    score: float
    content: dict
    metadata: dict
    link: Optional[str] = None

# class SearchQueryTextRequest(BaseModel):
#     query: str
#     documents: list[ExtractedDocument]
#     top_k: int = 5
#
# class SearchQueryTextResponse(BaseModel):
#     results: list[ExtractedDocument]

class VectorSearchRequest(BaseSearchRequest):
    vector: list[float]

class SearchResponse(BaseModel):
    results: list[SearchResult]
