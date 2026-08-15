from fastapi import UploadFile

from repositories.qdrant import QdrantRepository
from repositories.product import ProductRepository

from services.embedding import EmbeddingService
from services.ocr import OCRService

from schemas.search import VectorSearchRequest


class RetrievalService:

    def __init__(
            self,
            qdrant_repository: QdrantRepository,
            embedding_service: EmbeddingService,
            ocr_service: OCRService,
            product_repository: ProductRepository
    ):
        self.qdrant_repository = qdrant_repository
        self.embedding_service = embedding_service,
        self.ocr_service = ocr_service
        self.product_repository = product_repository


    async def find_by_photo(self, file: UploadFile):
        print(file.filename)
        top_k = 5

        ocr_data = self.ocr_service.text_detection_on_file(file) #"PaddleOCR PP - OCRv5"

        photo_to_text_vector = self.embedding_service.vectorize_text(ocr_data)
        photo_to_vector = self.embedding_service.vectorize_siglip(file)

        ocr_retrieval_result = await self.qdrant_repository.search(
            VectorSearchRequest(
                vector=photo_to_text_vector,
                top_k=top_k,
                # filters=, можно llm моделью проверять что за тип вина и добавлять в фильтры
                collection="ocr-data-vectors"
            )
        )
        siglip2_retrieval_result = await self.qdrant_repository.search(
            VectorSearchRequest(
                vector=photo_to_vector,
                top_k=top_k,
                # filters=, можно llm моделью проверять что за тип вина и добавлять в фильтры
                collection="siglip2-vectors"
            )
        )
        ocr_data_search_result = await self.product_repository.search(ocr_data)



        # todo use model photo-to-text, vectorization, search to collection (photo-to-text-vectors)
        # todo use model photo-to-vector, search to collection (photo-to-vector-vectors)
        # todo use model ocr, vectorization, HYBRID search to collection (ocr-data-vector)

        # todo Подсчет результатов, возврат ответа