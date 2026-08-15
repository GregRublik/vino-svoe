from fastapi import UploadFile

from repositories.qdrant import QdrantRepository

from services.embedding import EmbeddingService

from schemas.search import VectorSearchRequest


class RetrievalService:

    def __init__(self, qdrant_repository: QdrantRepository, embedding_service: EmbeddingService):
        self.qdrant_repository = qdrant_repository
        self.embedding_service = embedding_service,


    async def find_by_photo(self, file: UploadFile):
        print(file.filename)
        top_k = 5

        photo_to_text_vector =
        photo_to_vector =
        ocr_data_vector =

        results_photo_to_text = await self.qdrant_repository.search(
            VectorSearchRequest(
                vector=photo_to_text_vector,
                top_k=top_k,
                # filters=, можно llm моделью проверять что за тип вина и добавлять в фильтры
                collection="photo-to-text-vectors"
            )
        )

        # todo use model photo-to-text, vectorization, search to collection (photo-to-text-vectors)
        # todo use model photo-to-vector, search to collection (photo-to-vector-vectors)

        # todo use model ocr, vectorization, HYBRID search to collection (ocr-data-vector)

        # todo Подсчет результатов, возврат ответа