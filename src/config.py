from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(value: str | Path) -> Path:
    """Разрешает относительный путь относительно корня проекта, а не cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


class Settings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_siglip2: str = "siglip2-vectors"
    qdrant_collection_ocr: str = "ocr-data-vectors"
    catalog_path: str = "data/embedings/wine_catalog.json"
    links_path: str = "data/embedings/wine_links.json"
    images_path: str = "data/images"
    database_url: str = (
        "postgresql+psycopg://wine:wine_password@localhost:5432/vino_svoe"
    )
    wine_base_url: str = "https://vino-svoe.ru"

    siglip_model_id: str = "google/siglip2-large-patch16-384"
    siglip_revision: str = "1b426889ea62b5a72bf9839009a1b184bfc9c178"
    text_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    text_model_revision: str = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
    embedding_device: str = "auto"
    ocr_detection_model_id: str = "PP-OCRv4_mobile_det"
    ocr_recognition_model_id: str = "cyrillic_PP-OCRv5_mobile_rec"
    ocr_max_side: int = 1280
    ocr_min_confidence: float = 0.55
    ocr_enabled: bool = True
    ocr_device: str = "auto"
    max_upload_size_bytes: int = 10 * 1024 * 1024

    yolo_model_path: str = "data/yolo_ann/runs/bottle_yolo26n/weights/best.pt"
    yolo_confidence: float = 0.25
    yolo_image_size: int = 640
    yolo_crop_margin: float = 0.08
    yolo_enabled: bool = True
    yolo_device: str = "auto"

    retrieval_candidate_k: int = 20
    retrieval_ocr_candidate_k: int = 2000
    retrieval_min_ocr_matches: int = 2
    retrieval_min_confidence: float = 0.60
    retrieval_ocr_only_confidence: float = 0.75
    retrieval_visual_ocr_confidence: float = 0.65
    retrieval_confidence_visual_floor: float = 0.45
    retrieval_confidence_visual_ceiling: float = 0.85
    retrieval_confidence_margin_scale: float = 0.05
    retrieval_confidence_lexical_scale: float = 4.0
    rrf_k: int = 60

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_prefix="APP_",
        extra="ignore",
    )


settings = Settings()
