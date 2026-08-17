import httpx
import pytest
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.http.models.models import QueryResponse
from qdrant_client.models import Distance, Filter, PointStruct, ScoredPoint

from exceptions import QdrantCollectionNotFoundException
from repositories.qdrant import QdrantRepository
from schemas.search import VectorSearchRequest


def collection_not_found_error() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=404,
        reason_phrase="Not Found",
        content=b'{"status":{"error":"Not found: Collection c doesn\'t exist!"}}',
        headers=httpx.Headers(),
    )


def internal_error() -> UnexpectedResponse:
    return UnexpectedResponse(
        status_code=500,
        reason_phrase="Internal Server Error",
        content=b'{"status":{"error":"Internal"}}',
        headers=httpx.Headers(),
    )


class FakeClient:
    def __init__(self):
        self.query_log = []
        self.responses = []
        self.existing = set()
        self.create_calls = []
        self.upsert_calls = []
        self.ping_ok = True

    async def query_points(self, **kwargs):
        self.query_log.append(kwargs)
        if self.responses:
            item = self.responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        return QueryResponse(points=[])

    async def collection_exists(self, collection_name):
        return collection_name in self.existing

    async def create_collection(self, collection_name, vectors_config):
        self.create_calls.append((collection_name, vectors_config))
        self.existing.add(collection_name)

    async def upsert(self, collection_name, points):
        self.upsert_calls.append((collection_name, points))

    async def get_collections(self):
        if not self.ping_ok:
            raise ResponseHandlingException("connection failed")
        return {"collections": []}


@pytest.fixture
def client():
    return FakeClient()


@pytest.fixture
def repo(client):
    return QdrantRepository(client)


async def test_search_uses_top_k_as_limit(repo, client):
    await repo.search(VectorSearchRequest(vector=[0.1, 0.2], top_k=3, collection="c"))
    call = client.query_log[0]
    assert call["limit"] == 3
    assert call["collection_name"] == "c"
    assert call["query"] == [0.1, 0.2]
    assert call["with_payload"] is True


async def test_search_builds_filter_from_dict(repo, client):
    await repo.search(
        VectorSearchRequest(vector=[0.1], top_k=5, collection="c", filters={"color": "red"})
    )
    query_filter = client.query_log[0]["query_filter"]
    assert isinstance(query_filter, Filter)
    condition = query_filter.must[0]
    assert condition.key == "color"
    assert condition.match.value == "red"


async def test_search_maps_points_to_search_results(repo, client):
    client.responses.append(
        QueryResponse(
            points=[
                ScoredPoint(
                    id=5,
                    version=0,
                    score=0.87,
                    payload={"filename": "x.webp", "link": "https://vino-svoe.ru/wines/x"},
                )
            ]
        )
    )
    results = await repo.search(VectorSearchRequest(vector=[0.1], top_k=5, collection="c"))
    assert len(results) == 1
    result = results[0]
    assert result.id == 5
    assert result.score == 0.87
    assert result.content == {"filename": "x.webp", "link": "https://vino-svoe.ru/wines/x"}
    assert result.metadata == {"filename": "x.webp", "link": "https://vino-svoe.ru/wines/x"}
    assert result.link == "https://vino-svoe.ru/wines/x"


async def test_search_point_without_payload(repo, client):
    client.responses.append(QueryResponse(points=[ScoredPoint(id=7, version=0, score=0.5)]))
    results = await repo.search(VectorSearchRequest(vector=[0.1], top_k=5, collection="c"))
    assert results[0].content == {}
    assert results[0].metadata == {}
    assert results[0].link is None


async def test_search_collection_not_found_raises(repo, client):
    client.responses.append(collection_not_found_error())
    with pytest.raises(QdrantCollectionNotFoundException):
        await repo.search(VectorSearchRequest(vector=[0.1], top_k=5, collection="c"))


async def test_search_other_errors_reraised(repo, client):
    client.responses.append(internal_error())
    with pytest.raises(UnexpectedResponse):
        await repo.search(VectorSearchRequest(vector=[0.1], top_k=5, collection="c"))


async def test_collection_exists(repo, client):
    client.existing.add("a")
    assert await repo.collection_exists("a") is True
    assert await repo.collection_exists("b") is False


async def test_create_collection_skips_when_exists(repo, client):
    client.existing.add("c")
    await repo.create_collection("c", 1024)
    assert client.create_calls == []


async def test_create_collection_with_cosine(repo, client):
    await repo.create_collection("new", 1024)
    name, vectors_config = client.create_calls[0]
    assert name == "new"
    assert vectors_config.size == 1024
    assert vectors_config.distance == Distance.COSINE


async def test_upsert_passes_points(repo, client):
    points = [PointStruct(id=1, vector=[0.1, 0.2], payload={"link": "https://..."})]
    await repo.upsert("c", points)
    assert client.upsert_calls == [("c", points)]


async def test_ping_ok(repo):
    assert await repo.ping() is True


async def test_ping_failure(repo, client):
    client.ping_ok = False
    assert await repo.ping() is False
