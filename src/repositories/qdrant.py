from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from qdrant_client.models import QueryResponse
from qdrant_client.http.exceptions import UnexpectedResponse, ResponseHandlingException

from schemas.search import VectorSearchRequest

from exceptions import QdrantCollectionNotFoundException


class QdrantRepository:

    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client

    @staticmethod
    def _is_collection_not_found(error: UnexpectedResponse) -> bool:
        content = error.content.decode() if isinstance(error.content, bytes) else str(error.content)
        return "doesn't exist" in content

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
            return True
        except ResponseHandlingException:
            return False

    async def search(self, payload: VectorSearchRequest) -> QueryResponse:

        try:
            query_filter = None
            if payload.filters:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                        for key, value in payload.filters.items()
                    ]
                )

            return await self.client.query_points(
                collection_name=payload.collection,
                query=payload.vector,
                query_filter=query_filter,
                with_payload=True,
                limit=10
            )

        except UnexpectedResponse as e:
            if self._is_collection_not_found(e):
                raise QdrantCollectionNotFoundException
            raise
