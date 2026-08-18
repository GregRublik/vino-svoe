import hashlib


def stable_int_id(filename: str) -> int:
    """Стабильный числовой id точки из имени файла (встроенный hash() — salted, не годится).

    Один и тот же id используется во всех коллекциях Qdrant (siglip2, ocr),
    чтобы Reciprocal Rank Fusion мог объединять результаты по id.
    """
    digest = hashlib.sha256(filename.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)
