from fastapi import APIRouter, UploadFile, Depends

from services.retrieval import RetrievalService
from schemas.search import WineSearchResponse

from depends import get_retrieval_service

router = APIRouter()


@router.post("/search", response_model=WineSearchResponse)
async def search(
    photo: UploadFile,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Поиск вина по фото"""
    return await retrieval_service.find_by_photo(photo)
