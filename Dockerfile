FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev --extra ocr --no-install-project

COPY src ./src

EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]

