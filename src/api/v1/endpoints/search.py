import os

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from depends import get_retrieval_service
from schemas.search import SearchResponse
from services.retrieval import RetrievalService

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
):
    """Поиск вина по фото"""
    _validate_extension(photo.filename)
    return await retrieval_service.find_by_photo(photo, top_k=top_k)
