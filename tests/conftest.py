import io

import pytest
from PIL import Image


def make_webp_bytes(size=(64, 64), color=(255, 0, 0)) -> bytes:
    """Создаёт in-memory WEBP-изображение (без записи на диск)."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="WEBP")
    return buf.getvalue()


@pytest.fixture
def webp_bytes() -> bytes:
    return make_webp_bytes()


class FakeFile:
    """Замена UploadFile: async read + счётчик вызовов."""

    def __init__(self, data: bytes, filename: str = "wine.webp"):
        self.data = data
        self.filename = filename
        self.read_count = 0

    async def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        return self.data if size < 0 else self.data[:size]


@pytest.fixture
def make_fake_file():
    def factory(data: bytes, filename: str = "wine.webp") -> FakeFile:
        return FakeFile(data, filename)

    return factory
