"""Подготовка разметки CVAT и обучение YOLO26n на бутылках.

Запуск из корня репозитория:
    uv run python src/scripts/train_yolo.py --images-dir /path/to/images

Путь к исходным изображениям передаётся явно, поэтому скрипт не зависит от
названия папки или расположения конкретного проекта. Он делает воспроизводимое
train/val-разбиение и не изменяет исходную разметку CVAT.
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC_DIR))

REPO_ROOT = SRC_DIR.parent
DEFAULT_ANNOTATIONS_DIR = REPO_ROOT / "data" / "yolo_ann" / "obj_train_data"
DEFAULT_DATASET_DIR = REPO_ROOT / "data" / "yolo_ann" / "ultralytics_dataset"


def _image_for_stem(images_dir: Path, stem: str) -> Path:
    candidates = sorted(
        path
        for path in images_dir.glob(f"{stem}.*")
        if path.is_file() and path.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"Для разметки '{stem}.txt' не найдено исходное изображение в '{images_dir}'"
        )
    return candidates[0]


def _reset_generated_dir(dataset_dir: Path) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            directory = dataset_dir / kind / split
            directory.mkdir(parents=True, exist_ok=True)
            for path in directory.iterdir():
                if path.is_file() or path.is_symlink():
                    path.unlink()


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _write_dataset_yaml(dataset_dir: Path) -> Path:
    path = dataset_dir / "dataset.yaml"
    path.write_text(
        # YAML лежит внутри dataset_dir, поэтому относительный путь
        # делает датасет переносимым между хостом и Docker.
        "path: .\ntrain: images/train\nval: images/val\nnames:\n  0: bottle\n",
        encoding="utf-8",
    )
    return path


def prepare_dataset(
    images_dir: Path,
    annotations_dir: Path,
    dataset_dir: Path,
    val_ratio: float,
    seed: int,
) -> tuple[Path, int, int, int]:
    annotation_files = sorted(annotations_dir.glob("*.txt"))
    if not annotation_files:
        raise FileNotFoundError(f"В '{annotations_dir}' не найдена YOLO-разметка")

    pairs = [
        (image_path, annotation_path)
        for annotation_path in annotation_files
        for image_path in [_image_for_stem(images_dir, annotation_path.stem)]
    ]
    # Разбиение воспроизводимое, криптографическая случайность здесь не нужна.
    random.Random(seed).shuffle(pairs)  # nosec B311
    val_count = max(1, round(len(pairs) * val_ratio))
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    dataset_dir.mkdir(parents=True, exist_ok=True)
    _reset_generated_dir(dataset_dir)
    for split, split_pairs in (("train", train_pairs), ("val", val_pairs)):
        for image_path, annotation_path in split_pairs:
            _link_or_copy(image_path, dataset_dir / "images" / split / image_path.name)
            _link_or_copy(
                annotation_path,
                dataset_dir / "labels" / split / f"{image_path.stem}.txt",
            )

    dataset_yaml = _write_dataset_yaml(dataset_dir)
    box_count = sum(
        sum(
            1
            for line in annotation_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        for _, annotation_path in pairs
    )
    return dataset_yaml, len(train_pairs), len(val_pairs), box_count


def _resolve_device(value: str):
    if value != "auto":
        return value
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def main(args: argparse.Namespace) -> None:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Установите зависимости проекта командой: uv sync") from exc

    dataset_yaml, train_count, val_count, box_count = prepare_dataset(
        images_dir=args.images_dir,
        annotations_dir=args.annotations_dir,
        dataset_dir=args.dataset_dir,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(f"Разметка: {train_count} train, {val_count} val, всего рамок: {box_count}")

    model = YOLO(args.model)
    model.train(
        data=str(dataset_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=_resolve_device(args.device),
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        cache=False,
        patience=args.patience,
        pretrained=True,
        plots=True,
    )

    best_path = getattr(getattr(model, "trainer", None), "best", None)
    if best_path:
        print(f"Лучшая модель: {best_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=(
            Path(os.environ["YOLO_IMAGES_DIR"]).expanduser()
            if os.environ.get("YOLO_IMAGES_DIR")
            else None
        ),
        help="Каталог исходных изображений (или переменная YOLO_IMAGES_DIR)",
    )
    parser.add_argument("--annotations-dir", type=Path, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--project", type=Path, default=REPO_ROOT / "data" / "yolo_ann" / "runs"
    )
    parser.add_argument("--name", default="bottle_yolo26n")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.images_dir is None:
        parser.error("укажите --images-dir или задайте YOLO_IMAGES_DIR")
    main(args)
