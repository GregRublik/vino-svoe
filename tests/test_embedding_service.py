import io
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from services.embedding import EmbeddingService


class FakeSiglipConfig:
    hidden_size = 8


class FakeOutput:
    """Имитация BaseModelOutputWithPooling (как у реального get_image_features)."""

    def __init__(self, tensor):
        self.pooler_output = tensor


class FakeSiglipModel:
    """Модель с get_image_features, возвращает единичный вектор (1, hidden)."""
    config = FakeSiglipConfig()
    instances = []

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def __init__(self):
        FakeSiglipModel.instances.append(self)

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_image_features(self, pixel_values, pixel_mask=None):
        return FakeOutput(
            torch.ones(pixel_values.shape[0], FakeSiglipConfig.hidden_size)
        )


class FakePoolerModel:
    """Модель БЕЗ get_image_features — проверяем fallback на vision_model(...).pooler_output."""
    config = FakeSiglipConfig()

    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def to(self, device):
        return self

    def eval(self):
        return self

    def vision_model(self, pixel_values, pixel_mask=None):
        return FakeOutput(
            torch.ones(pixel_values.shape[0], FakeSiglipConfig.hidden_size)
        )


class FakeSiglipProcessor:
    @classmethod
    def from_pretrained(cls, model_id):
        return cls()

    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros(1, 3, 16, 16)}


class FakeTextModel:
    loaded = False

    def __init__(self, model_id):
        FakeTextModel.loaded = True

    def encode(self, text, normalize_embeddings=True):
        return torch.tensor([1.0, 0.0, 0.0, 0.0])


@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch):
    FakeSiglipModel.instances.clear()
    FakeTextModel.loaded = False
    monkeypatch.setattr("services.embedding.SiglipModel", FakeSiglipModel)
    monkeypatch.setattr("services.embedding.SiglipImageProcessor", FakeSiglipProcessor)
    monkeypatch.setattr("services.embedding.SentenceTransformer", FakeTextModel)


def test_lazy_load_models_not_loaded_on_init():
    service = EmbeddingService()
    assert FakeSiglipModel.instances == []
    assert FakeTextModel.loaded is False


def test_vectorize_photo_bytes_returns_normalized_vector(webp_bytes):
    service = EmbeddingService()
    vector = service.vectorize_photo(webp_bytes)
    assert isinstance(vector, list)
    assert all(isinstance(x, float) for x in vector)
    assert len(vector) == FakeSiglipConfig.hidden_size
    # единичный вектор после L2-нормализации: каждый элемент = 1/sqrt(8)
    assert all(x == pytest.approx(1 / (8 ** 0.5)) for x in vector)
    assert len(FakeSiglipModel.instances) == 1  # модель загрузилась при первом вызове


def test_vectorize_photo_accepts_pil_image(webp_bytes):
    service = EmbeddingService()
    image = Image.open(io.BytesIO(webp_bytes)).convert("RGB")
    vector = service.vectorize_photo(image)
    assert len(vector) == FakeSiglipConfig.hidden_size


def test_vectorize_photo_falls_back_to_pooler_output(monkeypatch, webp_bytes):
    monkeypatch.setattr("services.embedding.SiglipModel", FakePoolerModel)
    service = EmbeddingService()
    vector = service.vectorize_photo(webp_bytes)
    assert len(vector) == FakeSiglipConfig.hidden_size
    assert all(x == pytest.approx(1 / (8 ** 0.5)) for x in vector)


def test_vectorize_text_lazy_load_and_normalize(webp_bytes):
    service = EmbeddingService()
    assert FakeTextModel.loaded is False
    vector = service.vectorize_text("Вино сухое красное")
    assert vector == [1.0, 0.0, 0.0, 0.0]
    assert FakeTextModel.loaded is True


def test_device_mps_when_available(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True))
    assert EmbeddingService().device == "mps"


def test_device_cpu_when_mps_unavailable(monkeypatch):
    monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: False))
    assert EmbeddingService().device == "cpu"


def test_siglip_dim_from_model_config():
    assert EmbeddingService().siglip_dim == FakeSiglipConfig.hidden_size
    assert len(FakeSiglipModel.instances) == 1  # dim читается из конфига загруженной модели
