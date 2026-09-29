from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from exceptions import QdrantCollectionNotFoundException
from schemas.search import SearchResult, VectorSearchRequest
from utils.links import build_wine_link, extract_wine_slug


class QdrantRepository:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    @staticmethod
    def _is_collection_not_found(error: UnexpectedResponse) -> bool:
        content = (
            error.content.decode()
            if isinstance(error.content, bytes)
            else str(error.content)
        )
        return "doesn't exist" in content

    async def search(self, payload: VectorSearchRequest) -> list[SearchResult]:
        try:
            query_filter = None
            if payload.filters:
                query_filter = Filter(
                    must=[
                        FieldCondition(key=key, match=MatchValue(value=value))
                        for key, value in payload.filters.items()
                    ]
                )

            response = await self.client.query_points(
                collection_name=payload.collection,
                query=payload.vector,
                query_filter=query_filter,
                with_payload=True,
                limit=payload.top_k,
            )

            results = []
            for point in response.points:
                point_payload = point.payload or {}
                slug = extract_wine_slug(
                    content=point_payload,
                    metadata=point_payload,
                    link=point_payload.get("link"),
                )
                results.append(
                    SearchResult(
                        id=point.id,
                        score=point.score,
                        content=point_payload,
                        metadata=point_payload,
                        link=point_payload.get("link") or build_wine_link(slug),
                    )
                )
            return results

        except UnexpectedResponse as e:
            if self._is_collection_not_found(e):
                raise QdrantCollectionNotFoundException(
                    f"Collection {payload.collection} doesn't exist"
                ) from e
            raise

    async def collection_exists(self, name: str) -> bool:
        return await self.client.collection_exists(collection_name=name)

    async def count(self, collection: str) -> int:
        """Возвращает точное число точек в коллекции."""
        response = await self.client.count(collection_name=collection, exact=True)
        return int(response.count)

    async def create_collection(self, name: str, dim: int) -> None:
        """Создаёт коллекцию с косинусной метрикой, если её ещё нет (идемпотентно)."""
        if not await self.collection_exists(name):
            await self.client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    async def upsert(self, collection: str, points: list[PointStruct]) -> None:
        await self.client.upsert(collection_name=collection, points=points)
