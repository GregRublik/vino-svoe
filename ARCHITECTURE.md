# Архитектура

## Pipeline

1. `POST /search` или `POST /v1/eval/predict` принимает изображение.
2. `BottleDetectionService` запускает YOLO и выбирает одну наиболее уверенную
   бутылку. При отсутствии модели или детекции используется исходное фото.
3. `EmbeddingService` строит SigLIP2-вектор изображения. Устройство выбирается
   в порядке `CUDA -> MPS -> CPU`, если в настройках оставлено `auto`.
4. `QdrantRepository` ищет кандидатов в `siglip2-vectors`.
5. `OCRService` распознаёт текст, если OCR включён. Текст ищется в
   `ocr-data-vectors`, а лексический rerank уточняет near-duplicates.
6. `RetrievalService` объединяет визуальный и OCR-поиск через RRF и вычисляет
   диагностическую confidence.
7. Если confidence ниже настраиваемого порога, сервис возвращает кандидатов
   только как аналоги с `found=false`. Evaluation endpoint в этом случае не
   выдаёт случайный slug.
8. Для принятого результата API возвращает slug и данные, необходимые для
   перехода к карточке вина.

## Слои

- `api/` — HTTP-контракт, валидация файла и формат ответа evaluator;
- `services/` — inference и orchestration pipeline;
- `repositories/` — работа с Qdrant;
- `utils/` — нормализация OCR, slug/link, confidence и каталоговые профили;
- `scripts/` — идемпотентная индексация и обучение YOLO;
- `schemas/` — Pydantic-модели API.

## Расширение каталога

`src/scripts/index_catalog.py` последовательно обновляет визуальную и OCR
коллекции. Идентификатор точки стабилен и зависит от имени файла, поэтому
повторный запуск безопасен. Новая строка каталога без изображения всё равно
попадает в OCR-коллекцию как каталоговый профиль.

## Docker

Базовый `docker-compose.yml` запускает API и Qdrant. Файл
`docker-compose.gpu.yml` добавляет NVIDIA GPU reservation. Профиль PostgreSQL
сохранён в Compose как задел для будущего источника данных карточки, но текущий
поисковый pipeline к нему не подключается.
