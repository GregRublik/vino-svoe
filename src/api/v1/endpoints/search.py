import asyncio
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from depends import get_retrieval_service, get_wine_catalog_repository
from repositories.catalog import WineCatalogRepository
from schemas.search import EvaluatorResponse, PairingResponse, SearchResponse
from services.pairing import build_pairing
from services.retrieval import RetrievalService
from utils.links import extract_wine_slug

router = APIRouter()

ALLOWED_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}


def _validate_extension(filename: str | None) -> None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемое расширение файла: {ext or 'отсутствует'}",
        )


async def _run_search(
    photo: UploadFile = File(...),
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
) -> SearchResponse:
    _validate_extension(photo.filename)
    response = await retrieval_service.find_by_photo(photo, top_k=top_k)
    return await asyncio.to_thread(catalog_repository.enrich_response, response)


def _top_slug(response: SearchResponse) -> str | None:
    if not response.results:
        return None
    result = response.results[0]
    if result.card is not None:
        return result.card.slug
    return extract_wine_slug(
        content=result.content,
        metadata=result.metadata,
        link=result.link,
    )


@router.post("/search", response_model=EvaluatorResponse)
async def search(
    photo: UploadFile = File(...),
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
) -> EvaluatorResponse:
    """Плоский контракт для автоматического оценочного скрипта."""
    response = await _run_search(
        photo=photo,
        top_k=top_k,
        retrieval_service=retrieval_service,
        catalog_repository=catalog_repository,
    )
    return EvaluatorResponse(slug=_top_slug(response))


@router.post("/v1/eval/predict", response_model=EvaluatorResponse)
async def eval_predict(
    image: UploadFile = File(...),
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
) -> EvaluatorResponse:
    """Совместимость со скриптом оценки кейсодержателя."""
    response = await _run_search(
        photo=image,
        top_k=top_k,
        retrieval_service=retrieval_service,
        catalog_repository=catalog_repository,
    )
    return EvaluatorResponse(slug=_top_slug(response))


@router.post("/search/details", response_model=SearchResponse)
async def search_details(
    photo: UploadFile = File(...),
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
) -> SearchResponse:
    """Подробный контракт для пользовательского интерфейса."""
    return await _run_search(
        photo=photo,
        top_k=top_k,
        retrieval_service=retrieval_service,
        catalog_repository=catalog_repository,
    )


@router.get("/pairing/{slug}", response_model=PairingResponse)
async def pairing(
    slug: str,
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
) -> PairingResponse:
    """Подбирает блюдо для найденной карточки без внешнего API."""
    normalized_slug = slug.strip().strip("/")
    card = (await asyncio.to_thread(catalog_repository.get_cards, [normalized_slug])).get(
        normalized_slug
    )
    if card is None:
        raise HTTPException(status_code=404, detail="Карточка вина не найдена")
    return build_pairing(card)
