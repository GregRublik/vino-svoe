"""Подготовка фото для поиска: выделение бутылки моделью YOLO."""

import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps

from config import resolve_project_path, settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedImage:
    """Изображение, которое будет передано в SigLIP и OCR."""

    data: bytes


class BottleDetectionService:
    """Ленивая загрузка YOLO и выбор бутылки с лучшим detection score."""

    def __init__(self, model=None):
        self._model = model
        self._warning_logged = False

    @staticmethod
    def _to_list(value):
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "tolist"):
            value = value.tolist()
        return value

    @staticmethod
    def _device():
        if settings.yolo_device != "auto":
            return settings.yolo_device
        try:
            import torch

            if torch.cuda.is_available():
                return 0
            if (
                getattr(torch.backends, "mps", None) is not None
                and torch.backends.mps.is_available()
            ):
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_model(self):
        if self._model is not None:
            return self._model

        model_path = resolve_project_path(settings.yolo_model_path)
        if not model_path.is_file():
            raise FileNotFoundError("YOLO-модель не найдена")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics не установлен. Выполните: uv sync") from exc

        self._model = YOLO(str(model_path))
        return self._model

    @staticmethod
    def _encode(image: Image.Image) -> bytes:
        output = io.BytesIO()
        image.save(output, format="WEBP", quality=95, method=6)
        return output.getvalue()

    def _fallback(self, data: bytes, reason: Exception | None = None) -> PreparedImage:
        if reason is not None and not self._warning_logged:
            logger.warning("YOLO недоступен, используется исходное фото")
            self._warning_logged = True
        return PreparedImage(data=data)

    def prepare(self, data: bytes) -> PreparedImage:
        """Возвращает crop бутылки с лучшим detection score или исходное фото."""
        if not settings.yolo_enabled:
            return self._fallback(data)

        try:
            model = self._get_model()
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
            results = model.predict(
                source=image,
                conf=settings.yolo_detection_threshold,
                imgsz=settings.yolo_image_size,
                device=self._device(),
                verbose=False,
            )
        except (FileNotFoundError, ImportError, RuntimeError, OSError) as exc:
            return self._fallback(data, exc)

        if not results:
            return self._fallback(data)

        boxes = getattr(results[0], "boxes", None)
        if boxes is None:
            return self._fallback(data)

        detection_scores = self._to_list(getattr(boxes, "conf", [])) or []
        coordinates = self._to_list(getattr(boxes, "xyxy", [])) or []
        if not detection_scores or not coordinates:
            return self._fallback(data)

        count = min(len(detection_scores), len(coordinates))
        best_index = max(
            range(count), key=lambda index: float(detection_scores[index])
        )
        x1, y1, x2, y2 = (float(value) for value in coordinates[best_index])

        width, height = image.size
        box_width = max(1.0, x2 - x1)
        box_height = max(1.0, y2 - y1)
        margin_x = box_width * settings.yolo_crop_margin
        margin_y = box_height * settings.yolo_crop_margin
        crop_box = (
            max(0, int(x1 - margin_x)),
            max(0, int(y1 - margin_y)),
            min(width, int(x2 + margin_x)),
            min(height, int(y2 + margin_y)),
        )
        crop = image.crop(crop_box)
        return PreparedImage(
            data=self._encode(crop),
        )
