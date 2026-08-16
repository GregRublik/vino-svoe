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


class WineSearchResult(BaseModel):
    name: str
    score: float
    link: str


class WineSearchResponse(BaseModel):
    results: list[WineSearchResult]
