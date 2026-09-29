FROM ghcr.io/astral-sh/uv:0.11.32 AS uv
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
    HF_HOME=/opt/model-cache/huggingface \
    HF_HUB_CACHE=/opt/model-cache/huggingface/hub \
    PADDLE_PDX_CACHE_HOME=/opt/model-cache/paddlex \
    YOLO_CONFIG_DIR=/opt/model-cache/ultralytics

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

# Базовый lock-файл остаётся переносимым. Для production GPU-образа заменяем
# только backend-веса на CUDA 12.6 wheels; профиль не зависит от модели GPU.
ARG PYTORCH_VERSION=2.8.0+cu126
ARG TORCHVISION_VERSION=0.23.0+cu126
ARG PADDLE_VERSION=3.3.0
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra ocr --no-install-project \
        --no-install-package torch \
        --no-install-package torchvision \
        --no-install-package paddlepaddle \
        --no-install-package cuda-bindings \
        --no-install-package cuda-toolkit \
        --no-install-package triton \
        --no-install-package nvidia-cublas \
        --no-install-package nvidia-cuda-cupti \
        --no-install-package nvidia-cuda-nvrtc \
        --no-install-package nvidia-cuda-runtime \
        --no-install-package nvidia-cudnn-cu13 \
        --no-install-package nvidia-cufft \
        --no-install-package nvidia-cufile \
        --no-install-package nvidia-curand \
        --no-install-package nvidia-cusolver \
        --no-install-package nvidia-cusparse \
        --no-install-package nvidia-cusparselt-cu13 \
        --no-install-package nvidia-nccl-cu13 \
        --no-install-package nvidia-nvjitlink \
        --no-install-package nvidia-nvshmem-cu13 \
        --no-install-package nvidia-nvtx \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 \
        --reinstall "torch==${PYTORCH_VERSION}" \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://download.pytorch.org/whl/cu126 \
        --reinstall "torchvision==${TORCHVISION_VERSION}" \
    && uv pip install --python /app/.venv/bin/python \
        --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
        --no-deps "paddlepaddle-gpu==${PADDLE_VERSION}"

# Скачиваем все внешние веса во время сборки. В runtime сеть для model hub
# отключается, поэтому первый запрос не инициирует загрузку моделей.
#
# Важный нюанс: на этапе build контейнер намеренно не получает GPU-драйвер
# хоста. Поэтому PaddleOCR временно заменяется CPU-wheel только для загрузки
# OCR-моделей, после чего в том же слое возвращается GPU-wheel. В финальном
# образе CPU-wheel не остаётся, а сборка не зависит от конкретной видеокарты
# или установленного на build-хосте NVIDIA runtime.
RUN --mount=type=cache,target=/root/.cache/uv <<'SH'
set -eux

uv pip uninstall --python /app/.venv/bin/python paddlepaddle-gpu
uv pip install --python /app/.venv/bin/python \
    --no-cache \
    --no-deps \
    --reinstall "paddlepaddle==${PADDLE_VERSION}"

/app/.venv/bin/python - <<'PY'
from paddleocr import PaddleOCR
from sentence_transformers import SentenceTransformer
from transformers import SiglipImageProcessor, SiglipModel

siglip_id = "google/siglip2-large-patch16-384"
siglip_revision = "1b426889ea62b5a72bf9839009a1b184bfc9c178"
text_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
text_revision = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"

SiglipImageProcessor.from_pretrained(siglip_id, revision=siglip_revision)
SiglipModel.from_pretrained(siglip_id, revision=siglip_revision)
SentenceTransformer(text_id, revision=text_revision)
PaddleOCR(
    text_detection_model_name="PP-OCRv4_mobile_det",
    text_recognition_model_name="cyrillic_PP-OCRv5_mobile_rec",
    enable_mkldnn=False,
    device="cpu",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
PY

uv pip uninstall --python /app/.venv/bin/python paddlepaddle
uv pip install --python /app/.venv/bin/python \
    --index-url https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --no-deps \
    --reinstall "paddlepaddle-gpu==${PADDLE_VERSION}"
SH

RUN mkdir -p /app/models /opt/model-cache/ultralytics
COPY models/bottle_yolo26n.pt /app/models/bottle_yolo26n.pt

# В production-образ попадают только runtime-модули приложения.
# CLI-скрипты индексации/обучения и датасеты сюда намеренно не копируются.
COPY src/api ./src/api
COPY src/db ./src/db
COPY src/migrations ./src/migrations
COPY src/repositories ./src/repositories
COPY src/schemas ./src/schemas
COPY src/services ./src/services
COPY src/utils ./src/utils
COPY src/templates ./src/templates
COPY src/static ./src/static
COPY src/config.py src/depends.py src/exceptions.py src/main.py ./src/
COPY alembic.ini docker-entrypoint.sh ./
RUN chmod +x /app/docker-entrypoint.sh

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

EXPOSE 8000

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["/app/.venv/bin/uvicorn", "main:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
