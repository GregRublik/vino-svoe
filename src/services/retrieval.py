import asyncio

from fastapi import UploadFile

from config import settings
from exceptions import OCRNotAvailableError
from repositories.qdrant import QdrantRepository
from schemas.search import SearchResponse, VectorSearchRequest
from services.bottle_detector import BottleDetectionService
from services.embedding import EmbeddingService
from services.ocr import OCRService
from utils.confidence import calculate_match_confidence
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
        lexical_scores: dict[int, int],
    ) -> SearchResponse:
        candidates = results[:top_k]
        if not candidates:
            return SearchResponse(results=[], found=False, confidence=0.0)

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
        confidence = calculate_match_confidence(
            visual_score=best_visual_score,
            visual_margin=best_margin,
            lexical_matches=confirmed_ocr_matches,
            visual_floor=settings.retrieval_confidence_visual_floor,
            visual_ceiling=settings.retrieval_confidence_visual_ceiling,
            margin_scale=settings.retrieval_confidence_margin_scale,
            lexical_scale=settings.retrieval_confidence_lexical_scale,
        )
        if (
            best_visual_score is None
            and confirmed_ocr_matches >= settings.retrieval_min_ocr_matches
        ):
            confidence = max(confidence, settings.retrieval_ocr_only_confidence)
        if (
            best_visual_score is not None
            and visual_results
            and best.id == visual_results[0].id
            and best_visual_score >= settings.retrieval_confidence_visual_floor
            and confirmed_ocr_matches >= settings.retrieval_min_ocr_matches
        ):
            confidence = max(confidence, settings.retrieval_visual_ocr_confidence)
        return SearchResponse(
            results=candidates,
            found=confidence >= settings.retrieval_min_confidence,
            confidence=confidence,
            margin=round(visual_margin, 4) if visual_margin is not None else None,
            ocr_matches=best_ocr_matches,
        )

    async def find_by_photo(self, file: UploadFile, top_k: int = 5) -> SearchResponse:
        """Поиск вина по фото: SigLIP2 (+ OCR, когда коллекция проиндексирована) → RRF."""
        data = await file.read()
        prepared_data = data
        if self.bottle_detector is not None and settings.yolo_enabled:
            prepared_data = (
                await asyncio.to_thread(self.bottle_detector.prepare, data)
            ).data

        candidate_k = max(top_k, settings.retrieval_candidate_k)

        photo_vector = await asyncio.to_thread(
            self.embedding_service.vectorize_photo, prepared_data
        )
        siglip_results = await self.qdrant_repository.search( # ищем в коллекции siglip2
            VectorSearchRequest(
                vector=photo_vector,
                top_k=candidate_k,
                collection=settings.qdrant_collection_siglip2,
            )
        )
        if not siglip_results:
            return self._build_response([], top_k, [], {})

        ranked_lists = [siglip_results]
        lexical_scores: dict[int, int] = {}

        if settings.ocr_enabled and await self.qdrant_repository.collection_exists(
            settings.qdrant_collection_ocr
        ):
            try:
                ocr_result = await self.ocr_service.text_detection_on_file(prepared_data)
            except OCRNotAvailableError:
                ocr_text = ""
            else:
                ocr_confidence = getattr(ocr_result, "confidence", None)
                ocr_text = str(ocr_result)
                if (
                    ocr_confidence is not None
                    and ocr_confidence < settings.ocr_min_confidence
                ):
                    ocr_text = ""
            if ocr_text.strip():
                ocr_vector = await asyncio.to_thread(
                    self.embedding_service.vectorize_text, ocr_text
                )
                ocr_candidate_k = max(
                    candidate_k,
                    settings.retrieval_ocr_candidate_k,
                    await self.qdrant_repository.count(
                        settings.qdrant_collection_ocr
                    ),
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
                    if max(lexical_scores.values(), default=0) >= settings.retrieval_min_ocr_matches:
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
