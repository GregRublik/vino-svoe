"""CLI-обёртка для импорта JSON-каталога в PostgreSQL."""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

from services.catalog_import import main

if __name__ == "__main__":
    import os

    force = os.getenv("APP_CATALOG_REIMPORT", "").lower() == "true"
    main(force=force)
