import asyncio
import os
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from depends import get_retrieval_service, get_wine_catalog_repository
from repositories.catalog import WineCatalogRepository
from schemas.search import SearchResponse
from services.retrieval import RetrievalService

router = APIRouter()

ALLOWED_EXTENSIONS = {".webp", ".jpg", ".jpeg", ".png"}


def _validate_extension(filename: str | None) -> None:
    ext = os.path.splitext(filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемое расширение файла: {ext or 'отсутствует'}",
        )


@router.post("/search", response_model=SearchResponse)
async def search(
    photo: UploadFile = File(...),
    top_k: Annotated[int, Form(ge=1, le=20)] = 5,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    catalog_repository: WineCatalogRepository = Depends(get_wine_catalog_repository),
):
    """Поиск вина по фото"""
    _validate_extension(photo.filename)
    response = await retrieval_service.find_by_photo(photo, top_k=top_k)
    return await asyncio.to_thread(catalog_repository.enrich_response, response)
