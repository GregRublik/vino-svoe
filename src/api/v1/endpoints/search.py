import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from depends import get_retrieval_service, get_wine_catalog_repository
from repositories.catalog import WineCatalogRepository
from schemas.search import EvalPrediction, SearchResponse
from services.retrieval import RetrievalService
from utils.links import extract_wine_slug

router = APIRouter()

ALLOWED_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}


def _validate_extension(filename: str) -> None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемое расширение файла: {ext or 'отсутствует'}",
        )


@router.post("/search", response_model=SearchResponse)
async def search(
    photo: UploadFile = File(...),
    top_k: int = Form(5),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
):
    """Поиск вина по фото"""
    _validate_extension(photo.filename)
    response = await retrieval_service.find_by_photo(photo, top_k=top_k)
    return catalog_repository.enrich_response(response)


@router.post("/v1/eval/predict", response_model=EvalPrediction, tags=["eval"])
async def predict(
    image: UploadFile = File(...),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Контракт eval-ТЗ: возвращает slug лучшего совпадения по фото."""
    _validate_extension(image.filename)
    # Для Top-1 оставляем запас кандидатов, чтобы OCR и визуальный поиск
    # могли объединиться, а не сравнивались только по одному результату.
    response = await retrieval_service.find_by_photo(image, top_k=5)

    if not response.results or not response.found:
        raise HTTPException(status_code=404, detail="Вино по изображению не найдено")

    result = response.results[0]
    slug = extract_wine_slug(
        content=result.content,
        metadata=result.metadata,
        link=result.link,
    )
    if not slug:
        raise HTTPException(status_code=404, detail="Для результата не найден slug")

    return EvalPrediction(slug=slug)
