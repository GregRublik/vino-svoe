from fastapi import FastAPI
import uvicorn

from api.v1.endpoints import search

from config import settings

app = FastAPI()

app.include_router(search.router)

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
