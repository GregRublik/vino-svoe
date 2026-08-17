import pytest
from fastapi.testclient import TestClient

from depends import get_retrieval_service
from exceptions import QdrantCollectionNotFoundException
from main import app
from schemas.search import SearchResponse, SearchResult

LINK = "https://vino-svoe.ru/wines/x"


class FakeRetrievalService:
    def __init__(self, response: SearchResponse | None = None, error: Exception | None = None):
        self.response = response or SearchResponse(results=[])
        self.error = error
        self.calls = []

    async def find_by_photo(self, file, top_k=5):
        self.calls.append((file.filename, top_k))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def client():
    app.dependency_overrides.clear()
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def override_service(client):
    fake = FakeRetrievalService(
        response=SearchResponse(
            results=[SearchResult(id=1, score=0.5, content={}, metadata={}, link=LINK)]
        )
    )
    app.dependency_overrides[get_retrieval_service] = lambda: fake
    return fake


def test_search_post_returns_results(client, override_service, webp_bytes):
    resp = client.post(
        "/search",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
        data={"top_k": "3"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["link"] == LINK
    assert override_service.calls == [("w.webp", 3)]


def test_search_default_top_k_is_five(client, override_service, webp_bytes):
    resp = client.post("/search", files={"photo": ("w.webp", webp_bytes, "image/webp")})
    assert resp.status_code == 200
    assert override_service.calls == [("w.webp", 5)]


def test_search_rejects_bad_extension(client, override_service):
    resp = client.post("/search", files={"photo": ("w.txt", b"hello", "text/plain")})
    assert resp.status_code == 400
    assert override_service.calls == []


def test_search_503_when_collection_missing(client, webp_bytes):
    fake = FakeRetrievalService(error=QdrantCollectionNotFoundException)
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post("/search", files={"photo": ("w.webp", webp_bytes, "image/webp")})
    assert resp.status_code == 503
    assert "detail" in resp.json()
