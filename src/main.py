from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

from api.v1.endpoints import search

from config import settings
from exceptions import QdrantCollectionNotFoundException

app = FastAPI()

app.include_router(search.router)


@app.exception_handler(QdrantCollectionNotFoundException)
async def qdrant_collection_not_found_handler(
    request: Request, exc: QdrantCollectionNotFoundException
):
    return JSONResponse(
        status_code=503,
        content={"detail": "Векторная коллекция ещё не проиндексирована"},
    )


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
