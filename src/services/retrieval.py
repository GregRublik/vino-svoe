import asyncio
import io

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from config import settings
from exceptions import InvalidImageError, OCRNotAvailableError, PhotoTooLargeError
from repositories.qdrant import QdrantRepository
from schemas.search import SearchResponse, SearchResultId, VectorSearchRequest
from services.bottle_detector import BottleDetectionService
from services.embedding import EmbeddingService
from services.ocr import OCRService
from utils.fusion import reciprocal_rank_fusion
from utils.ocr_text import ocr_result_match_score, rank_ocr_results


class RetrievalService:
    def __init__(
        self,
        qdrant_repository: QdrantRepository,
        embedding_service: EmbeddingService,
        ocr_service: OCRService,
        bottle_detector: BottleDetectionService | None = None,
    ):
        self.qdrant_repository = qdrant_repository
        self.embedding_service = embedding_service
        self.ocr_service = ocr_service
        self.bottle_detector = bottle_detector

    @staticmethod
    def _build_response(
        results,
        top_k: int,
        visual_results,
        lexical_scores: dict[SearchResultId, int],
    ) -> SearchResponse:
        candidates = results[:top_k]
        if not candidates:
            return SearchResponse(results=[], found=False)

        visual_scores = {result.id: result.score for result in visual_results}
        visual_margin = None
        if len(visual_results) > 1:
            visual_margin = max(0.0, visual_results[0].score - visual_results[1].score)

        best = candidates[0]
        best_visual_score = visual_scores.get(best.id)
        best_margin = visual_margin if best_visual_score is not None else None
        best_ocr_matches = lexical_scores.get(best.id, 0)
        confirmed_ocr_matches = (
            best_ocr_matches
            if best_ocr_matches >= settings.retrieval_min_ocr_matches
            else 0
        )
        is_visual_top1 = bool(visual_results and best.id == visual_results[0].id)
        visual_match = (
            is_visual_top1
            and best_visual_score is not None
            and best_visual_score >= settings.retrieval_visual_score_threshold
            and (
                best_margin is None
                or best_margin >= settings.retrieval_visual_margin_threshold
                or best_visual_score >= settings.retrieval_visual_strong_score_threshold
            )
        )
        ocr_match = confirmed_ocr_matches >= settings.retrieval_min_ocr_matches and (
            best_visual_score is None
            or (
                is_visual_top1
                and best_visual_score >= settings.retrieval_ocr_visual_score_threshold
            )
        )
        return SearchResponse(
            results=candidates,
            found=visual_match or ocr_match,
            margin=round(visual_margin, 4) if visual_margin is not None else None,
            ocr_matches=best_ocr_matches,
        )

    async def find_by_photo(self, file: UploadFile, top_k: int = 5) -> SearchResponse:
        """Поиск вина по фото: SigLIP2 (+ OCR, когда коллекция проиндексирована) → RRF."""
        data = await file.read(settings.max_upload_size_bytes + 1)
        if len(data) > settings.max_upload_size_bytes:
            raise PhotoTooLargeError
        try:
            with Image.open(io.BytesIO(data)) as image:
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError from exc

        prepared_data = data
        if self.bottle_detector is not None and settings.yolo_enabled:
            prepared_data = (
                await asyncio.to_thread(self.bottle_detector.prepare, data)
            ).data

        candidate_k = max(top_k, settings.retrieval_candidate_k)

        photo_vector = await asyncio.to_thread(
            self.embedding_service.vectorize_photo, prepared_data
        )
        siglip_results = (
            await self.qdrant_repository.search(  # ищем в коллекции siglip2
                VectorSearchRequest(
                    vector=photo_vector,
                    top_k=candidate_k,
                    collection=settings.qdrant_collection_siglip2,
                )
            )
        )
        if not siglip_results:
            return self._build_response([], top_k, [], {})

        ranked_lists = [siglip_results]
        lexical_scores: dict[SearchResultId, int] = {}

        if settings.ocr_enabled and await self.qdrant_repository.collection_exists(
            settings.qdrant_collection_ocr
        ):
            try:
                ocr_result = await self.ocr_service.text_detection_on_file(
                    prepared_data
                )
            except OCRNotAvailableError:
                ocr_text = ""
            else:
                ocr_text = str(ocr_result)
            if ocr_text.strip():
                ocr_vector = await asyncio.to_thread(
                    self.embedding_service.vectorize_text, ocr_text
                )
                ocr_candidate_k = max(
                    candidate_k,
                    settings.retrieval_ocr_candidate_k,
                    await self.qdrant_repository.count(settings.qdrant_collection_ocr),
                )
                ocr_results = await self.qdrant_repository.search(
                    VectorSearchRequest(
                        vector=ocr_vector,
                        top_k=ocr_candidate_k,
                        collection=settings.qdrant_collection_ocr,
                    )
                )
                if ocr_results:
                    # Для RRF используем тот же короткий список, что и для
                    # визуального поиска. Полный список нужен только для
                    # точного лексического совпадения по этикетке.
                    ranked_lists.append(ocr_results[:candidate_k])
                    lexical_scores = {
                        result.id: ocr_result_match_score(result, ocr_text)
                        for result in ocr_results
                    }
                    if (
                        max(lexical_scores.values(), default=0)
                        >= settings.retrieval_min_ocr_matches
                    ):
                        lexical_results = rank_ocr_results(ocr_results, ocr_text)
                        ranked_lists.append(lexical_results)

        if len(ranked_lists) == 1:
            return self._build_response(
                siglip_results,
                top_k,
                siglip_results,
                lexical_scores,
            )

        fused_results = reciprocal_rank_fusion(ranked_lists, k=settings.rrf_k)
        max_lexical_score = max(lexical_scores.values(), default=0)
        if max_lexical_score >= settings.retrieval_min_ocr_matches:
            visual_scores = {result.id: result.score for result in siglip_results}
            fused_results.sort(
                key=lambda result: (
                    lexical_scores.get(result.id, 0),
                    visual_scores.get(result.id, -1.0),
                    result.score,
                ),
                reverse=True,
            )
        return self._build_response(
            fused_results,
            top_k,
            siglip_results,
            lexical_scores,
        )
