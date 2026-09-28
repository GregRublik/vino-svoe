import io

import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import SiglipImageProcessor, SiglipModel

from config import settings


class EmbeddingService:
    """Векторизация фото (SigLIP2) и текста (sentence-transformer).

    Модели загружаются лениво при первом обращении — сервис можно создавать
    без скачивания весов (важно для тестов и быстрого старта API).
    """

    def __init__(
        self,
        siglip_model_id: str | None = None,
        text_model_id: str | None = None,
    ):
        self._siglip_model_id = siglip_model_id or settings.siglip_model_id
        self._text_model_id = text_model_id or settings.text_model_id
        self._siglip_model = None
        self._siglip_processor = None
        self._text_model = None

    @property
    def device(self) -> str:
        if settings.embedding_device != "auto":
            return settings.embedding_device
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"

    @property
    def siglip_dim(self) -> int:
        if self._siglip_model is None:
            self._load_siglip()
        vision_config = getattr(self._siglip_model.config, "vision_config", None)
        if vision_config is not None:
            return vision_config.hidden_size
        return self._siglip_model.config.hidden_size

    def _load_siglip(self) -> None:
        # Чекпоинты google/siglip2-* имеют model_type "siglip" в конфиге и
        # совместимы с SiglipModel (через Siglip2Model transformers 5.x
        # переинициализирует часть весов — см. отчёт загрузки).
        self._siglip_processor = SiglipImageProcessor.from_pretrained(self._siglip_model_id)
        self._siglip_model = (
            SiglipModel.from_pretrained(self._siglip_model_id).to(self.device).eval()
        )

    def _load_text_model(self) -> None:
        # sentence-transformers 5.x удалил SentenceTransformer.from_pretrained —
        # загрузка через конструктор.
        model = SentenceTransformer(self._text_model_id)
        if hasattr(model, "to"):
            model = model.to(self.device)
        self._text_model = model

    def vectorize_photo(self, image: bytes | Image.Image) -> list[float]:
        """Эмбеддинг фото (SigLIP2), L2-нормализованный."""
        if self._siglip_model is None:
            self._load_siglip()

        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")

        inputs = self._siglip_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        kwargs = {}
        if inputs.get("pixel_mask") is not None:
            kwargs["pixel_mask"] = inputs["pixel_mask"].to(self.device)

        with torch.no_grad():
            if hasattr(self._siglip_model, "get_image_features"):
                out = self._siglip_model.get_image_features(
                    pixel_values=pixel_values, **kwargs
                )
                # transformers возвращает BaseModelOutputWithPooling
                features = out.pooler_output if not torch.is_tensor(out) else out
            else:
                features = self._siglip_model.vision_model(
                    pixel_values=pixel_values, **kwargs
                ).pooler_output

        vector = features.squeeze(0)
        vector = vector / vector.norm(p=2)
        return vector.cpu().tolist()

    def vectorize_text(self, text: str) -> list[float]:
        """Эмбеддинг текста (sentence-transformer), нормализованный."""
        if self._text_model is None:
            self._load_text_model()
        vector = self._text_model.encode(text, normalize_embeddings=True)
        return vector.tolist()
