"""Утилиты для подготовки списка изображений к индексации."""

import hashlib
from pathlib import Path


def deduplicate_files(files: list[Path]) -> tuple[list[Path], int]:
    """Удаляет из списка только побайтные копии, сохраняя первый файл.

    Разные фотографии одного товара не считаются дублями: их slug может быть
    одинаковым, но они могут давать разные визуальные признаки.
    """
    unique_files: list[Path] = []
    seen_digests: set[str] = set()

    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        unique_files.append(path)

    return unique_files, len(files) - len(unique_files)
