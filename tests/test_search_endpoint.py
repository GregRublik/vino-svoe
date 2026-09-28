import pytest
from fastapi.testclient import TestClient

from depends import get_retrieval_service
from schemas.search import SearchResult
from exceptions import QdrantCollectionNotFoundException
from main import app
from schemas.search import SearchResponse

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


def test_search_includes_wine_card_with_description(client, webp_bytes):
    fake = FakeRetrievalService(
        response=SearchResponse(
            results=[
                SearchResult(
                    id=1,
                    score=0.9,
                    content={"slug": "aligote-barrel-2024"},
                    metadata={},
                    link="https://vino-svoe.ru/wines/aligote-barrel-2024",
                )
            ]
        )
    )
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post(
        "/search",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    card = resp.json()["results"][0]["card"]
    assert card["name"] == "Алиготе Баррель, 2024"
    assert card["description"]


def test_search_returns_cards_for_unknown_wine_alternatives(client, webp_bytes):
    fake = FakeRetrievalService(
        response=SearchResponse(
            found=False,
            confidence=0.2,
            results=[
                SearchResult(
                    id=1,
                    score=0.5,
                    content={"slug": "aligote-barrel-2024"},
                    metadata={},
                    link="https://vino-svoe.ru/wines/aligote-barrel-2024",
                )
            ],
        )
    )
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post(
        "/search",
        files={"photo": ("unknown.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["results"][0]["card"]["description"]


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


def test_eval_predict_returns_tz_slug(client, override_service, webp_bytes):
    resp = client.post(
        "/v1/eval/predict",
        files={"image": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"slug": "x"}
    assert override_service.calls == [("w.webp", 5)]


def test_eval_predict_returns_404_without_results(client, webp_bytes):
    fake = FakeRetrievalService(response=SearchResponse(results=[]))
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post(
        "/v1/eval/predict",
        files={"image": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 404


def test_eval_predict_does_not_return_low_confidence_slug(client, webp_bytes):
    fake = FakeRetrievalService(
        response=SearchResponse(
            results=[SearchResult(id=1, score=0.69, content={}, metadata={}, link=LINK)],
            found=False,
            confidence=0.31,
        )
    )
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post(
        "/v1/eval/predict",
        files={"image": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 404
