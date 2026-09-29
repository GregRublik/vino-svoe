from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_production_dockerfile_copies_only_runtime_source_tree():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "COPY src ./src" not in dockerfile
    assert "COPY models/bottle_yolo26n.pt /app/models/bottle_yolo26n.pt" in dockerfile
    assert "COPY src/services ./src/services" in dockerfile
    assert "COPY src/templates ./src/templates" in dockerfile
    assert "COPY src/static ./src/static" in dockerfile


def test_production_image_bakes_gpu_runtime_and_model_caches():
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")

    assert "download.pytorch.org/whl/cu126" in dockerfile
    assert "paddlepaddle-gpu" in dockerfile
    assert "HF_HUB_OFFLINE=1" in dockerfile
    assert "TRANSFORMERS_OFFLINE=1" in dockerfile
    assert "PaddleOCR(" in dockerfile


def test_catalog_import_is_a_runtime_module_not_a_cli_dependency():
    entrypoint = (PROJECT_ROOT / "docker-entrypoint.sh").read_text(encoding="utf-8")

    assert "python -m services.catalog_import" in entrypoint
    assert "python -m scripts.import_catalog" not in entrypoint


def test_docker_context_excludes_bytecode_and_local_data():
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert "src/scripts" in dockerignore
    assert "data" in dockerignore
    assert "**/__pycache__/" in dockerignore
    assert "**/*.pyc" in dockerignore
