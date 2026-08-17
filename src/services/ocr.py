import asyncio
import io

import numpy as np
from PIL import Image

from exceptions import OCRNotAvailableError


class OCRService:
    """Распознавание текста на фото (PaddleOCR 3.x).

    paddleocr — опциональная зависимость: импортируется лениво при первом
    использовании. Без неё API работает, но распознавание недоступно.
    """

    def __init__(self, engine=None):
        self._engine = engine

    def _get_engine(self):
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as e:
                raise OCRNotAvailableError(
                    "paddleocr не установлен. Установите: uv sync --extra ocr"
                ) from e
            try:
                self._engine = PaddleOCR(lang="ru")
            except Exception:
                self._engine = PaddleOCR()
        return self._engine

    async def text_detection_on_file(self, data: bytes) -> str:
        """Распознаёт текст с фото (CPU-bound — выполняется в потоке)."""
        return await asyncio.to_thread(self._run_ocr, data)

    def _run_ocr(self, data: bytes) -> str:
        engine = self._get_engine()
        image = Image.open(io.BytesIO(data)).convert("RGB")
        try:
            raw = engine.predict(np.array(image))
        except Exception:
            return ""

        texts: list[str] = []
        # PaddleOCR 3.x: predict() возвращает list[dict] (по одному на изображение)
        # с ключом "rec_texts" — распаковываем защитно.
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    rec_texts = item.get("rec_texts") or []
                    texts.extend(str(t) for t in rec_texts)
        return "\n".join(t for t in texts if t)
