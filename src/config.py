from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str
    port: int

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection_siglip2: str = "siglip2-vectors"
    qdrant_collection_ocr: str = "ocr-data-vectors"

    siglip_model_id: str = "google/siglip2-large-patch16-384"
    text_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    rrf_k: int = 60

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")


settings = Settings()
