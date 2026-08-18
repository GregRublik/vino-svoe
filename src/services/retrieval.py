import asyncio

from fastapi import UploadFile

from config import settings
from repositories.product import ProductRepository
from repositories.qdrant import QdrantRepository
from schemas.search import SearchResponse, VectorSearchRequest
from services.embedding import EmbeddingService
from services.ocr import OCRService
from utils.fusion import reciprocal_rank_fusion


class RetrievalService:

    def __init__(
        self,
        qdrant_repository: QdrantRepository,
        embedding_service: EmbeddingService,
        ocr_service: OCRService,
        product_repository: ProductRepository | None = None,
    ):
        self.qdrant_repository = qdrant_repository
        self.embedding_service = embedding_service
        self.ocr_service = ocr_service
        self.product_repository = product_repository

    async def find_by_photo(self, file: UploadFile, top_k: int = 5) -> SearchResponse:
        """Поиск вина по фото: SigLIP2 (+ OCR, когда коллекция проиндексирована) → RRF."""
        data = await file.read()

        photo_vector = await asyncio.to_thread(
            self.embedding_service.vectorize_photo, data
        )
        siglip_results = await self.qdrant_repository.search( # ищем в коллекции siglip2
            VectorSearchRequest(
                vector=photo_vector,
                top_k=top_k,
                collection=settings.qdrant_collection_siglip2,
            )
        )
        print(siglip_results[0])
        ranked_lists = [siglip_results]

        if await self.qdrant_repository.collection_exists(
            settings.qdrant_collection_ocr
        ):
            ocr_text = await self.ocr_service.text_detection_on_file(data)
            print(ocr_text)
            if ocr_text.strip():
                ocr_vector = await asyncio.to_thread(
                    self.embedding_service.vectorize_text, ocr_text
                )
                ocr_results = await self.qdrant_repository.search(
                    VectorSearchRequest(
                        vector=ocr_vector,
                        top_k=top_k,
                        collection=settings.qdrant_collection_ocr,
                    )
                )
                ranked_lists.append(ocr_results)

        return SearchResponse(
            results=reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
        )
