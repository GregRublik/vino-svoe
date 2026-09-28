"""Идемпотентная индексация каталога в обе коллекции Qdrant.

Запуск из корня репозитория:
    uv run python src/scripts/index_catalog.py
    uv run python src/scripts/index_catalog.py --rebuild --filenames-only
"""

import argparse
import asyncio
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from scripts.index_images import main as index_images
from scripts.index_ocr import main as index_ocr


async def main(rebuild: bool = False, filenames_only: bool = False) -> None:
    await index_images(rebuild=rebuild)
    await index_ocr(filenames_only=filenames_only, rebuild=rebuild)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--filenames-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(rebuild=args.rebuild, filenames_only=args.filenames_only))
