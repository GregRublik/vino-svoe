from fastapi import APIRouter, UploadFile, Depends

from src.services.retrieval import RetrievalService

from depends import get_retrieval_service

router = APIRouter()


@router.get("/search")
async def search(
    photo: UploadFile,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Поиск вина по фото"""

    return await retrieval_service.find_by_photo(photo)