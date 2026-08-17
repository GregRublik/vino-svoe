import builtins

import pytest

from exceptions import OCRNotAvailableError
from services.ocr import OCRService


class FakeEngine:
    def __init__(self, results):
        self.results = results
        self.predict_calls = 0

    def predict(self, image_array):
        self.predict_calls += 1
        return self.results


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
