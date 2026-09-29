import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.v1.endpoints import pages, search
from config import resolve_project_path, settings
from depends import get_qdrant_client, get_wine_catalog_repository
from exceptions import (
    CatalogUnavailableError,
    InvalidImageError,
    PhotoTooLargeError,
    QdrantCollectionNotFoundException,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    dependency_overrides = application.dependency_overrides
    repository_factory = application.dependency_overrides.get(
        get_wine_catalog_repository,
        get_wine_catalog_repository,
    )
    catalog_repository = repository_factory()
    qdrant_factory = dependency_overrides.get(get_qdrant_client, get_qdrant_client)
    qdrant_client = qdrant_factory()
    try:
        await asyncio.to_thread(catalog_repository.initialize)
        yield
    finally:
        catalog_repository.close()
        await qdrant_client.close()
        get_wine_catalog_repository.cache_clear()
        get_qdrant_client.cache_clear()


app = FastAPI(lifespan=lifespan)

app.include_router(search.router)
app.include_router(pages.router)


@app.exception_handler(QdrantCollectionNotFoundException)
async def qdrant_collection_not_found_handler(
    request: Request, exc: QdrantCollectionNotFoundException
):
    return JSONResponse(
        status_code=503,
        content={"detail": "Векторная коллекция ещё не проиндексирована"},
    )


@app.exception_handler(CatalogUnavailableError)
async def catalog_unavailable_handler(request: Request, exc: CatalogUnavailableError):
    return JSONResponse(
        status_code=503,
        content={"detail": "Каталог вин временно недоступен"},
    )


@app.exception_handler(PhotoTooLargeError)
async def photo_too_large_handler(request: Request, exc: PhotoTooLargeError):
    return JSONResponse(
        status_code=413,
        content={"detail": "Размер изображения превышает допустимый лимит"},
    )


@app.exception_handler(InvalidImageError)
async def invalid_image_handler(request: Request, exc: InvalidImageError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Файл не является корректным изображением"},
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(request: Request, exc: Exception):
    """Не отдаёт клиенту traceback и внутренние пути файловой системы."""
    logger.exception("Непредвиденная ошибка при обработке запроса")
    return JSONResponse(
        status_code=500,
        content={"detail": "Не удалось обработать изображение"},
    )


app.mount(
    "/static",
    StaticFiles(directory=str(resolve_project_path("src/static"))),
    name="static",
)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
