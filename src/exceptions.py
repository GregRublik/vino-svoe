class QdrantCollectionNotFoundException(Exception):
    pass


class CatalogUnavailableError(Exception):
    """Каталог вин не удалось инициализировать или прочитать."""


class PhotoTooLargeError(Exception):
    """Загруженное изображение превышает ограничение размера."""


class InvalidImageError(Exception):
    """Загруженный файл не является поддерживаемым изображением."""


class OCRNotAvailableError(Exception):
    """PaddleOCR не установлен (опциональная зависимость) или не может быть загружен."""
