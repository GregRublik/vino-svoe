import asyncio
import io

import numpy as np
from PIL import Image

from config import settings
from exceptions import OCRNotAvailableError


class OCRService:
    """Распознавание текста на фото (PaddleOCR 3.x).

    paddleocr — опциональная зависимость: импортируется лениво при первом
    использовании. Без неё API работает, но распознавание недоступно.
    """

    def __init__(self, engine=None):
        self._engine = engine

    @staticmethod
    def _normalize_device(requested: str) -> str:
        requested = requested.strip().lower()
        if requested == "cuda" or requested == "gpu":
            return "gpu:0"
        if requested.startswith("cuda:"):
            return f"gpu:{requested.split(':', 1)[1]}"
        return requested

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import paddle

            return bool(
                paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0
            )
        except (ImportError, AttributeError, RuntimeError):
            return False

    @classmethod
    def _device(cls) -> str:
        requested = cls._normalize_device(settings.ocr_device)
        if requested == "auto":
            return "gpu:0" if cls._cuda_available() else "cpu"
        if requested.startswith("gpu:") and not cls._cuda_available():
            return "cpu"
        return requested

    def _create_engine(self, recognition_model_id: str, fallback_lang: str):
        try:
            from paddleocr import PaddleOCR
        except ImportError as e:
            raise OCRNotAvailableError(
                "paddleocr не установлен. Установите: uv sync --extra ocr"
            ) from e

        try:
            return PaddleOCR(
                text_detection_model_name=settings.ocr_detection_model_id,
                text_recognition_model_name=recognition_model_id,
                enable_mkldnn=False,
                device=self._device(),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )
        except Exception:
            return PaddleOCR(
                lang=fallback_lang,
                enable_mkldnn=False,
                device=self._device(),
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=False,
            )

    def _get_engine(self):
        if self._engine is None:
            self._engine = self._create_engine(
                settings.ocr_recognition_model_id,
                fallback_lang="ru",
            )
        return self._engine

    async def text_detection_on_file(self, data: bytes) -> str:
        """Распознаёт текст с фото (CPU-bound — выполняется в потоке)."""
        return await asyncio.to_thread(self._run_ocr, data)

    def _run_ocr(self, data: bytes) -> str:
        engine = self._get_engine()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        image.thumbnail(
            (settings.ocr_max_side, settings.ocr_max_side),
            Image.Resampling.LANCZOS,
        )
        image_array = np.array(image)
        texts = self._predict_texts(engine, image_array)
        return "\n".join(texts)

    @staticmethod
    def _predict_texts(engine, image_array: np.ndarray) -> list[str]:
        try:
            raw = engine.predict(image_array)
        except Exception:
            return []

        texts: list[str] = []
        # PaddleOCR 3.x: predict() возвращает list[dict] (по одному на изображение)
        # с ключом "rec_texts" — распаковываем защитно.
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    rec_texts = item.get("rec_texts") or []
                    texts.extend(str(t) for t in rec_texts)
        return [text for text in texts if text]
