import builtins
import io

import pytest
from PIL import Image

from config import settings
from exceptions import OCRNotAvailableError
from services.ocr import OCRService


def test_default_ocr_recognition_model_is_cyrillic_pp_ocr_v5():
    assert settings.ocr_recognition_model_id == "cyrillic_PP-OCRv5_mobile_rec"


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.predict_calls = 0

    def predict(self, image_array):
        self.predict_calls += 1
        return self.results


class ShapeRecordingEngine(FakeEngine):
    def predict(self, image_array):
        self.shape = image_array.shape
        return super().predict(image_array)


async def test_engine_import_error_raises(webp_bytes, monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "paddleocr":
            raise ImportError("No module named 'paddleocr'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    service = OCRService()
    with pytest.raises(OCRNotAvailableError):
        await service.text_detection_on_file(webp_bytes)


async def test_ocr_returns_joined_texts(webp_bytes):
    engine = FakeEngine([{"rec_texts": ["Вино", "Абрау-Дюрсо"]}])
    service = OCRService(engine=engine)
    result = await service.text_detection_on_file(webp_bytes)
    assert result == "Вино\nАбрау-Дюрсо"
    assert engine.predict_calls == 1


async def test_ocr_joins_recognized_text_without_extra_metadata(webp_bytes):
    engine = FakeEngine(
        [
            {
                "rec_texts": ["Вино", "Абрау-Дюрсо"],
            }
        ]
    )

    result = await OCRService(engine=engine).text_detection_on_file(webp_bytes)

    assert result == "Вино\nАбрау-Дюрсо"


async def test_ocr_empty_result_returns_empty_string(webp_bytes):
    service = OCRService(engine=FakeEngine([]))
    assert await service.text_detection_on_file(webp_bytes) == ""


async def test_ocr_unexpected_structure_returns_empty_string(webp_bytes):
    # PaddleOCR 3.x: нет ключа rec_texts / другая структура → пустая строка
    service = OCRService(engine=FakeEngine([{"rec_polys": [[1, 2, 3, 4]]}]))
    assert await service.text_detection_on_file(webp_bytes) == ""


async def test_ocr_injected_engine_used_immediately(webp_bytes, monkeypatch):
    """Если движок передан в конструктор, paddleocr не импортируется вообще."""
    real_import = builtins.__import__

    def fail_paddle(name, *args, **kwargs):
        if name == "paddleocr":
            raise AssertionError("paddleocr should not be imported")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_paddle)
    service = OCRService(engine=FakeEngine([{"rec_texts": ["OK"]}]))
    assert await service.text_detection_on_file(webp_bytes) == "OK"


async def test_ocr_limits_input_image_size(monkeypatch):
    buffer = io.BytesIO()
    Image.new("RGB", (200, 100), (255, 255, 255)).save(buffer, format="WEBP")
    engine = ShapeRecordingEngine([{"rec_texts": ["OK"]}])
    monkeypatch.setattr(settings, "ocr_max_side", 64)

    result = await OCRService(engine=engine).text_detection_on_file(buffer.getvalue())

    assert result == "OK"
    assert max(engine.shape[:2]) == 64
