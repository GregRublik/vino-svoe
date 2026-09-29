import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from db.database import Base
from db.models import WineCatalogRow
from depends import get_retrieval_service, get_wine_catalog_repository
from exceptions import (
    InvalidImageError,
    PhotoTooLargeError,
    QdrantCollectionNotFoundException,
)
from main import app
from repositories.catalog import WineCatalogRepository
from schemas.search import SearchResponse, SearchResult

LINK = "https://vino-svoe.ru/wines/aligote-barrel-2024"


class FakeRetrievalService:
    def __init__(
        self, response: SearchResponse | None = None, error: Exception | None = None
    ):
        self.response = response or SearchResponse(results=[])
        self.error = error
        self.calls = []

    async def find_by_photo(self, file, top_k=5):
        self.calls.append((file.filename, top_k))
        if self.error:
            raise self.error
        return self.response


@pytest.fixture
def client(tmp_path):
    database_url = f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}"
    engine = create_engine(database_url)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(
            WineCatalogRow(
                slug="aligote-barrel-2024",
                name="Алиготе Баррель, 2024",
                description="Описание Алиготе Баррель.",
                roskachestvo_rating="Высокое качество",
                serving_recommendation="Рыба и молодые сыры.",
                link=LINK,
            )
        )
        session.commit()
    engine.dispose()

    repository = WineCatalogRepository(database_url)
    app.dependency_overrides.clear()
    app.dependency_overrides[get_wine_catalog_repository] = lambda: repository
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    repository.close()
    app.dependency_overrides.clear()


@pytest.fixture
def override_service(client):
    fake = FakeRetrievalService(
        response=SearchResponse(
            results=[
                SearchResult(
                    id=1,
                    score=0.5,
                    content={"slug": "aligote-barrel-2024"},
                    metadata={},
                    link=LINK,
                )
            ]
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
    assert resp.json() == {"slug": "aligote-barrel-2024"}
    assert override_service.calls == [("w.webp", 3)]


def test_search_default_top_k_is_five(client, override_service, webp_bytes):
    resp = client.post("/search", files={"photo": ("w.webp", webp_bytes, "image/webp")})
    assert resp.status_code == 200
    assert override_service.calls == [("w.webp", 5)]


def test_case_holder_evaluator_contract_accepts_image(client, override_service, webp_bytes):
    resp = client.post(
        "/v1/eval/predict",
        files={"image": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    assert resp.json() == {"slug": "aligote-barrel-2024"}
    assert override_service.calls == [("w.webp", 5)]


def test_evaluator_returns_null_slug_for_empty_results(client, webp_bytes):
    fake = FakeRetrievalService(response=SearchResponse(results=[]))
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post("/search", files={"photo": ("empty.webp", webp_bytes, "image/webp")})

    assert resp.status_code == 200
    assert resp.json() == {"slug": None}


def test_ui_search_returns_detailed_contract(client, override_service, webp_bytes):
    resp = client.post(
        "/search/details",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["results"][0]["link"] == LINK
    assert set(body) == {"results", "found", "margin", "ocr_matches"}


@pytest.mark.parametrize("top_k", ["0", "101"])
def test_search_rejects_invalid_top_k(client, override_service, webp_bytes, top_k):
    resp = client.post(
        "/search/details",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
        data={"top_k": top_k},
    )

    assert resp.status_code == 422
    assert override_service.calls == []


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
        "/search/details",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    card = resp.json()["results"][0]["card"]
    assert card["name"] == "Алиготе Баррель, 2024"
    assert card["description"]
    assert card["roskachestvo_rating"] == "Высокое качество"
    assert card["serving_recommendation"] == "Рыба и молодые сыры."


def test_search_returns_cards_for_unknown_wine_alternatives(client, webp_bytes):
    fake = FakeRetrievalService(
        response=SearchResponse(
            found=False,
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
        "/search/details",
        files={"photo": ("unknown.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["found"] is False
    assert body["results"][0]["card"]["description"]


def test_pairing_endpoint_returns_deterministic_recommendation(client):
    resp = client.get("/pairing/aligote-barrel-2024")

    assert resp.status_code == 200
    assert resp.json() == {
        "slug": "aligote-barrel-2024",
        "recommendation": "Рыба и молодые сыры.",
        "rationale": "Рекомендация взята из карточки каталога.",
    }


def test_pairing_endpoint_handles_unknown_slug(client):
    resp = client.get("/pairing/unknown-wine")

    assert resp.status_code == 404
    assert resp.json() == {"detail": "Карточка вина не найдена"}


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


def test_search_hides_unexpected_file_paths(client, webp_bytes):
    fake = FakeRetrievalService(
        error=FileNotFoundError("/home/user/project/data/model.pt")
    )
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post("/search", files={"photo": ("w.webp", webp_bytes, "image/webp")})

    assert resp.status_code == 500
    assert resp.json() == {"detail": "Не удалось обработать изображение"}
    assert "/home/user/project" not in resp.text


@pytest.mark.parametrize(
    ("error", "status_code"),
    [(PhotoTooLargeError(), 413), (InvalidImageError(), 400)],
)
def test_search_maps_processing_errors(client, webp_bytes, error, status_code):
    fake = FakeRetrievalService(error=error)
    app.dependency_overrides[get_retrieval_service] = lambda: fake

    resp = client.post(
        "/search",
        files={"photo": ("w.webp", webp_bytes, "image/webp")},
    )

    assert resp.status_code == status_code
