from unittest.mock import AsyncMock, Mock

import pytest

from config import settings
from exceptions import QdrantCollectionNotFoundException
from schemas.search import SearchResult
from services.retrieval import RetrievalService

LINK = "https://vino-svoe.ru/wines/abrau-dyurso"


def make_result(doc_id: int, score: float = 0.9, link: str | None = LINK) -> SearchResult:
    return SearchResult(
        id=doc_id,
        score=score,
        content={"link": link},
        metadata={"filename": f"wine_{doc_id}.webp"},
        link=link,
    )


@pytest.fixture
def qdrant_repo():
    repo = Mock()
    repo.search = AsyncMock()
    repo.collection_exists = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def embedding_service():
    emb = Mock()
    emb.vectorize_photo = Mock(return_value=[0.1, 0.2, 0.3])
    emb.vectorize_text = Mock(return_value=[0.4, 0.5])
    return emb


@pytest.fixture
def ocr_service():
    ocr = Mock()
    ocr.text_detection_on_file = AsyncMock(return_value="Вино сухое белое")
    return ocr


def make_service(qdrant_repo, embedding_service, ocr_service) -> RetrievalService:
    return RetrievalService(
        qdrant_repository=qdrant_repo,
        embedding_service=embedding_service,
        ocr_service=ocr_service,
    )


async def test_both_collections_exist_full_flow(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.search = AsyncMock(side_effect=[[make_result(1)], [make_result(1)]])
    service = make_service(qdrant_repo, embedding_service, ocr_service)
    file = make_fake_file(webp_bytes)

    response = await service.find_by_photo(file)

    ocr_service.text_detection_on_file.assert_awaited_once_with(webp_bytes)
    assert qdrant_repo.search.await_count == 2
    first_call = qdrant_repo.search.await_args_list[0].args[0]
    second_call = qdrant_repo.search.await_args_list[1].args[0]
    assert first_call.collection == settings.qdrant_collection_siglip2
    assert second_call.collection == settings.qdrant_collection_ocr

    # общий документ на 1-м ранге в обоих списках: 2/61
    assert len(response.results) == 1
    assert response.results[0].id == 1
    assert response.results[0].score == pytest.approx(2 / 61)
    assert response.results[0].link == LINK


async def test_ocr_collection_missing_skips_ocr_entirely(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(2)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    ocr_service.text_detection_on_file.assert_not_awaited()
    embedding_service.vectorize_text.assert_not_called()
    assert qdrant_repo.search.await_count == 1
    assert response.results[0].id == 2


async def test_siglip_collection_missing_propagates(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(side_effect=QdrantCollectionNotFoundException)
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    with pytest.raises(QdrantCollectionNotFoundException):
        await service.find_by_photo(make_fake_file(webp_bytes))


async def test_empty_results_return_empty_response(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))
    assert response.results == []


async def test_file_read_exactly_once(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(1)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)
    file = make_fake_file(webp_bytes)

    await service.find_by_photo(file)

    assert file.read_count == 1
    embedding_service.vectorize_photo.assert_called_once_with(webp_bytes)


async def test_top_k_threaded_to_both_searches(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.search = AsyncMock(side_effect=[[make_result(1)], [make_result(2)]])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    await service.find_by_photo(make_fake_file(webp_bytes), top_k=7)

    for call in qdrant_repo.search.await_args_list:
        assert call.args[0].top_k == 7


async def test_default_top_k_is_five(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(1)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    await service.find_by_photo(make_fake_file(webp_bytes))
    assert qdrant_repo.search.await_args_list[0].args[0].top_k == 5


async def test_empty_ocr_text_skips_text_search(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    ocr_service.text_detection_on_file = AsyncMock(return_value="   ")
    qdrant_repo.search = AsyncMock(return_value=[make_result(1)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    embedding_service.vectorize_text.assert_not_called()
    assert qdrant_repo.search.await_count == 1
    assert len(response.results) == 1
