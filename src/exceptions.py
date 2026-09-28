class QdrantCollectionNotFoundException(Exception):
    pass


class OCRNotAvailableError(Exception):
    """PaddleOCR не установлен (опциональная зависимость) или не может быть загружен."""
