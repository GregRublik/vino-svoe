from functools import lru_cache

from fastapi import Depends
from qdrant_client import AsyncQdrantClient
from fastapi.templating import Jinja2Templates

from config import settings
from repositories.catalog import WineCatalogRepository
from repositories.qdrant import QdrantRepository
from services.bottle_detector import BottleDetectionService
from services.embedding import EmbeddingService
from services.ocr import OCRService
from services.retrieval import RetrievalService

templates = Jinja2Templates(directory="src/templates")


@lru_cache
def get_qdrant_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=settings.qdrant_url)


def get_qdrant_repository(
    client: AsyncQdrantClient = Depends(get_qdrant_client),
) -> QdrantRepository:
    return QdrantRepository(client)


@lru_cache
def get_wine_catalog_repository() -> WineCatalogRepository:
    return WineCatalogRepository(settings.catalog_path)


@lru_cache
def get_embedding_service() -> EmbeddingService:
    return EmbeddingService()


@lru_cache
def get_ocr_service() -> OCRService:
    return OCRService()


@lru_cache
def get_bottle_detector() -> BottleDetectionService:
    return BottleDetectionService()


def get_retrieval_service(
    qdrant_repository: QdrantRepository = Depends(get_qdrant_repository),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    ocr_service: OCRService = Depends(get_ocr_service),
    bottle_detector: BottleDetectionService = Depends(get_bottle_detector),
) -> RetrievalService:
    return RetrievalService(
        qdrant_repository=qdrant_repository,
        embedding_service=embedding_service,
        ocr_service=ocr_service,
        bottle_detector=bottle_detector,
    )
