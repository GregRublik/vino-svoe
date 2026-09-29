from unittest.mock import AsyncMock, Mock

import pytest

from config import settings
from exceptions import (
    InvalidImageError,
    PhotoTooLargeError,
    QdrantCollectionNotFoundException,
)
from schemas.search import SearchResult
from services.bottle_detector import PreparedImage
from services.retrieval import RetrievalService

LINK = "https://vino-svoe.ru/wines/abrau-dyurso"


def make_result(
    doc_id: int,
    score: float = 0.9,
    link: str | None = LINK,
    slug: str | None = None,
) -> SearchResult:
    content = {"link": link}
    if slug is not None:
        content["slug"] = slug
    return SearchResult(
        id=doc_id,
        score=score,
        content=content,
        metadata={"filename": f"wine_{doc_id}.webp"},
        link=link,
    )


@pytest.fixture
def qdrant_repo():
    repo = Mock()
    repo.search = AsyncMock()
    repo.collection_exists = AsyncMock(return_value=True)
    repo.count = AsyncMock(return_value=0)
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


async def test_detector_crop_is_used_for_visual_search_and_ocr(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    cropped = b"cropped-image"
    detector = Mock()
    detector.prepare = Mock(
        return_value=PreparedImage(
            data=cropped,
        )
    )
    qdrant_repo.search = AsyncMock(side_effect=[[make_result(1)], [make_result(1)]])
    service = RetrievalService(
        qdrant_repository=qdrant_repo,
        embedding_service=embedding_service,
        ocr_service=ocr_service,
        bottle_detector=detector,
    )

    await service.find_by_photo(make_fake_file(webp_bytes))

    detector.prepare.assert_called_once_with(webp_bytes)
    embedding_service.vectorize_photo.assert_called_once_with(cropped)
    ocr_service.text_detection_on_file.assert_awaited_once_with(cropped)


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


async def test_ocr_can_be_disabled_for_low_memory(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file, monkeypatch
):
    monkeypatch.setattr(settings, "ocr_enabled", False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(2)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    ocr_service.text_detection_on_file.assert_not_awaited()
    assert response.results[0].id == 2


async def test_single_source_preserves_cosine_score(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    """Без OCR-коллекции список один — RRF не нужен, скор остаётся косинусным."""
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(2, score=0.75)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert len(response.results) == 1
    assert response.results[0].id == 2
    assert response.results[0].score == pytest.approx(0.75)


async def test_visual_results_are_grouped_by_slug(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(
        return_value=[
            make_result(1, score=0.80, slug="massandra-muskat"),
            make_result(2, score=0.79, slug="massandra-muskat"),
            make_result(3, score=0.70, slug="massandra-muskatel"),
        ]
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert [result.id for result in response.results] == [1, 3]
    assert response.margin == pytest.approx(0.10)


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
    assert response.found is False


async def test_weak_visual_returns_candidates_as_alternatives(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(
        return_value=[
            make_result(1, score=0.695),
            make_result(2, score=0.670),
        ]
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.found is False
    assert response.results[0].id == 1


async def test_ocr_match_confirms_candidate(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    candidate = make_result(1, score=0.62)
    candidate = candidate.model_copy(
        update={
            "content": {
                "filename": "abrau-dyurso.webp",
                "text": "Абрау Дюрсо брют шардоне",
            },
            "metadata": {
                "filename": "abrau-dyurso.webp",
                "text": "Абрау Дюрсо брют шардоне",
            },
        }
    )
    qdrant_repo.search = AsyncMock(side_effect=[[candidate], [candidate]])
    ocr_service.text_detection_on_file = AsyncMock(return_value="Абрау Дюрсо шардоне")
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.found is True
    assert response.ocr_matches >= settings.retrieval_min_ocr_matches


async def test_strong_visual_top1_with_ocr_confirmation_is_accepted(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    candidate = make_result(1, score=0.770879)
    candidate = candidate.model_copy(
        update={
            "content": {
                "filename": "fanagoriya-formula-q-saperavi-krasnoe-suhoe-135.webp",
                "link": "https://vino-svoe.ru/wines/fanagoriya-formula-q-saperavi-krasnoe-suhoe-135",
            },
            "metadata": {
                "filename": "fanagoriya-formula-q-saperavi-krasnoe-suhoe-135.webp",
                "link": "https://vino-svoe.ru/wines/fanagoriya-formula-q-saperavi-krasnoe-suhoe-135",
            },
            "link": "https://vino-svoe.ru/wines/fanagoriya-formula-q-saperavi-krasnoe-suhoe-135",
        }
    )
    visual_competitor = make_result(2, score=0.769860)
    ocr_candidate = candidate.model_copy(update={"score": 0.0327868852})
    qdrant_repo.search = AsyncMock(
        side_effect=[[candidate, visual_competitor], [ocr_candidate]]
    )
    ocr_service.text_detection_on_file = AsyncMock(
        return_value="ФОРМУЛА Q САПЕРАВИ"
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.results[0].id == candidate.id
    assert response.ocr_matches >= settings.retrieval_min_ocr_matches
    assert response.found is True


async def test_equal_ocr_matches_keep_stronger_visual_candidate(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    visual_top1 = make_result(1, score=0.818322)
    visual_top1 = visual_top1.model_copy(
        update={
            "content": {
                "filename": "relikta-relikta-sira-kaberne-fran-krasnoe-suhoe-13.webp",
                "link": "https://vino-svoe.ru/wines/relikta-relikta-sira-kaberne-fran-krasnoe-suhoe-13",
            },
            "metadata": {
                "filename": "relikta-relikta-sira-kaberne-fran-krasnoe-suhoe-13.webp",
                "link": "https://vino-svoe.ru/wines/relikta-relikta-sira-kaberne-fran-krasnoe-suhoe-13",
            },
        }
    )
    visual_competitor = make_result(2, score=0.7955905)
    visual_competitor = visual_competitor.model_copy(
        update={
            "content": {
                "filename": "relikta-relikta-kaberne-sovinon-krasnoe-suhoe-135.webp",
                "link": "https://vino-svoe.ru/wines/relikta-relikta-kaberne-sovinon-krasnoe-suhoe-135",
            },
            "metadata": {
                "filename": "relikta-relikta-kaberne-sovinon-krasnoe-suhoe-135.webp",
                "link": "https://vino-svoe.ru/wines/relikta-relikta-kaberne-sovinon-krasnoe-suhoe-135",
            },
        }
    )
    ocr_filler = make_result(3, score=0.7)
    qdrant_repo.search = AsyncMock(
        side_effect=[
            [visual_top1, visual_competitor],
            [
                visual_competitor,
                ocr_filler,
                ocr_filler.model_copy(update={"id": 4}),
                visual_top1,
            ],
        ]
    )
    ocr_service.text_detection_on_file = AsyncMock(
        return_value="РЕЛИКТА КАБЕРНЕ КРАСНОЕ"
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.results[0].id == visual_top1.id
    assert response.ocr_matches >= settings.retrieval_min_ocr_matches


async def test_strong_visual_top1_survives_one_generic_ocr_match_lead(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    visual_top1 = make_result(1, score=0.76558447)
    visual_top1 = visual_top1.model_copy(
        update={
            "content": {
                "filename": "golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136.webp",
                "link": "https://vino-svoe.ru/wines/golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136",
            },
            "metadata": {
                "filename": "golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136.webp",
                "link": "https://vino-svoe.ru/wines/golubitskoe-estate-noble-selection-red-blend-kaberne-sovinon-krasnoe-suhoe-136",
            },
        }
    )
    visual_competitor = make_result(2, score=0.75671184)
    visual_competitor = visual_competitor.model_copy(
        update={
            "content": {
                "filename": "golubitskoe-estate-red-blend-kaberne-sovinon-krasnoe-suhoe-136.webp",
                "link": "https://vino-svoe.ru/wines/golubitskoe-estate-red-blend-kaberne-sovinon-krasnoe-suhoe-136",
            },
            "metadata": {
                "filename": "golubitskoe-estate-red-blend-kaberne-sovinon-krasnoe-suhoe-136.webp",
                "link": "https://vino-svoe.ru/wines/golubitskoe-estate-red-blend-kaberne-sovinon-krasnoe-suhoe-136",
            },
        }
    )
    qdrant_repo.search = AsyncMock(
        side_effect=[[visual_top1, visual_competitor], [visual_competitor, visual_top1]]
    )
    ocr_service.text_detection_on_file = AsyncMock(
        return_value="S ESTATE RED BLEND 2019"
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.results[0].id == visual_top1.id
    assert response.found is True


async def test_year_compound_does_not_beat_strong_visual_product_match(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    visual_top1 = make_result(1, score=0.821611)
    visual_top1 = visual_top1.model_copy(
        update={
            "content": {
                "filename": "zhemchuzhnaya-9-pino-nuar-muskat-rozovyj-2.webp",
                "link": "https://vino-svoe.ru/wines/zhemchuzhnaya-9-pino-nuar-muskat-rozovyj-2",
            },
            "metadata": {
                "filename": "zhemchuzhnaya-9-pino-nuar-muskat-rozovyj-2.webp",
                "link": "https://vino-svoe.ru/wines/zhemchuzhnaya-9-pino-nuar-muskat-rozovyj-2",
            },
        }
    )
    visual_competitor = make_result(2, score=0.818)
    visual_competitor = visual_competitor.model_copy(
        update={
            "content": {
                "filename": "leto-kaberne-fran-2024-polusladkoe-krasnoe.webp",
                "slug": "leto-kaberne-fran-2024-polusladkoe-krasnoe",
                "text": (
                    "LETO Каберне Фран 2024 полусладкое красное\n"
                    "leto-kaberne-fran-2024-polusladkoe-krasnoe\n"
                    "LETO\nКрасное\nКубань\nКаберне Фран"
                ),
                "link": "https://vino-svoe.ru/wines/leto-kaberne-fran-2024-polusladkoe-krasnoe",
            },
            "metadata": {
                "filename": "leto-kaberne-fran-2024-polusladkoe-krasnoe.webp",
                "slug": "leto-kaberne-fran-2024-polusladkoe-krasnoe",
                "text": (
                    "LETO Каберне Фран 2024 полусладкое красное\n"
                    "leto-kaberne-fran-2024-polusladkoe-krasnoe\n"
                    "LETO\nКрасное\nКубань\nКаберне Фран"
                ),
                "link": "https://vino-svoe.ru/wines/leto-kaberne-fran-2024-polusladkoe-krasnoe",
            },
        }
    )
    ocr_service.text_detection_on_file = AsyncMock(
        return_value="1600 01 ЖЕМЧУЖНАЯ 9 A PAT 2024"
    )
    qdrant_repo.search = AsyncMock(
        side_effect=[[visual_top1, visual_competitor], [visual_competitor, visual_top1]]
    )
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    response = await service.find_by_photo(make_fake_file(webp_bytes))

    assert response.results[0].id == visual_top1.id


def test_ocr_only_exact_match_is_accepted():
    candidate = make_result(42)
    candidate = candidate.model_copy(
        update={
            "content": {
                "slug": "czitronnyj-magaracha",
                "text": "Цитрон\nТабия\nБелое\nКубань",
            },
            "metadata": {},
        }
    )

    response = RetrievalService._build_response(
        [candidate],
        top_k=5,
        visual_results=[],
        lexical_scores={candidate.id: settings.retrieval_min_ocr_matches},
    )

    assert response.found is True


def test_weak_ocr_match_does_not_confirm_visual_candidate():
    candidate = make_result(42, score=0.62)

    response = RetrievalService._build_response(
        [candidate],
        top_k=5,
        visual_results=[candidate],
        lexical_scores={candidate.id: settings.retrieval_min_ocr_matches},
    )

    assert response.found is False


def test_strong_ocr_match_confirms_visual_candidate():
    candidate = make_result(42, score=0.62)

    response = RetrievalService._build_response(
        [candidate],
        top_k=5,
        visual_results=[candidate],
        lexical_scores={candidate.id: settings.retrieval_strong_ocr_matches},
    )

    assert response.found is True


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


async def test_rejects_oversized_file(
    qdrant_repo,
    embedding_service,
    ocr_service,
    webp_bytes,
    make_fake_file,
    monkeypatch,
):
    monkeypatch.setattr(settings, "max_upload_size_bytes", 4)
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    with pytest.raises(PhotoTooLargeError):
        await service.find_by_photo(make_fake_file(webp_bytes))

    embedding_service.vectorize_photo.assert_not_called()


async def test_rejects_invalid_image_bytes(
    qdrant_repo,
    embedding_service,
    ocr_service,
    make_fake_file,
):
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    with pytest.raises(InvalidImageError):
        await service.find_by_photo(make_fake_file(b"not an image"))

    embedding_service.vectorize_photo.assert_not_called()


async def test_top_k_threaded_to_both_searches(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.search = AsyncMock(side_effect=[[make_result(1)], [make_result(2)]])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    await service.find_by_photo(make_fake_file(webp_bytes), top_k=7)

    assert qdrant_repo.search.await_args_list[0].args[0].top_k == max(
        7, settings.retrieval_candidate_k
    )
    assert qdrant_repo.search.await_args_list[1].args[0].top_k == max(
        7, settings.retrieval_ocr_candidate_k, 0
    )


async def test_ocr_search_covers_collection_size(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file, monkeypatch
):
    monkeypatch.setattr(settings, "retrieval_ocr_candidate_k", 2)
    qdrant_repo.count = AsyncMock(return_value=42)
    qdrant_repo.search = AsyncMock(side_effect=[[make_result(1)], [make_result(2)]])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    await service.find_by_photo(make_fake_file(webp_bytes), top_k=5)

    qdrant_repo.count.assert_awaited_once_with(settings.qdrant_collection_ocr)
    assert qdrant_repo.search.await_args_list[1].args[0].top_k == 42


async def test_default_top_k_is_five(
    qdrant_repo, embedding_service, ocr_service, webp_bytes, make_fake_file
):
    qdrant_repo.collection_exists = AsyncMock(return_value=False)
    qdrant_repo.search = AsyncMock(return_value=[make_result(1)])
    service = make_service(qdrant_repo, embedding_service, ocr_service)

    await service.find_by_photo(make_fake_file(webp_bytes))
    assert qdrant_repo.search.await_args_list[0].args[0].top_k == max(
        5, settings.retrieval_candidate_k
    )


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
