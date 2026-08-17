from functools import lru_cache

from fastapi import Depends
from qdrant_client import AsyncQdrantClient

from config import settings
from repositories.product import ProductRepository
from repositories.qdrant import QdrantRepository
from services.embedding import EmbeddingService
from services.ocr import OCRService
from services.retrieval import RetrievalService


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)


def get_qdrant_repository(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> QdrantRepository:
    return QdrantRepository(client)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


def get_ocr_service() -> OCRService:
    return OCRService()


def get_retrieval_service(
    qdrant_repository: QdrantRepository = Depends(get_qdrant_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    ocr_service: OCRService = Depends(get_ocr_service),
) -> RetrievalService:
    return RetrievalService(
        qdrant_repository=qdrant_repository,
        embedding_service=embedding_service,
        ocr_service=ocr_service,
        product_repository=ProductRepository(),
    )
