from fastapi import FastAPI
import uvicorn
from config import settings

app = FastAPI()

if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port)
