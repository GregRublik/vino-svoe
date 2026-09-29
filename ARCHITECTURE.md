# Архитектура «Свое Вино»

## Назначение

Сервис принимает фотографию этикетки, подготавливает изображение, ищет
кандидатов в Qdrant по визуальному embedding, при наличии OCR объединяет
результаты и обогащает их карточками из PostgreSQL.

## Пайплайн запроса

```text
браузер / evaluator
        │ multipart photo/image
        ▼
FastAPI endpoints
        │ проверка расширения и top_k
        ▼
RetrievalService
        │ размер и формат → YOLO crop → SigLIP2
        │                     └→ OCR → текстовый embedding → lexical match
        ▼
QdrantRepository
        │ visual search + optional OCR search + RRF
        ▼
явное решение found по visual score / margin / OCR
        │
        ▼
WineCatalogRepository → PostgreSQL → card
        │
        ├── evaluator: {"slug": "..."}
        └── UI: подробный SearchResponse
```

Пустой список кандидатов считается безопасным результатом: подробный ответ
содержит `results=[]` и `found=false`, а evaluator возвращает `{"slug":null}`.

## Границы слоёв

- `src/api/v1/endpoints` — HTTP-контракты, валидация загрузки и преобразование
  результата в ответ. Endpoint не знает деталей Qdrant или SQL-запросов.
- `src/services/retrieval.py` — orchestration поиска и правила принятия
  `found`. Здесь нет агрегированного показателя качества: решение строится
  на отдельных признаках score, margin и лексических OCR-совпадениях.
- `src/services/ocr.py`, `embedding.py`, `bottle_detector.py` — адаптеры
  модельных компонентов. Веса загружаются лениво.
- `src/repositories/qdrant.py` — доступ к векторным коллекциям.
- `src/repositories/catalog.py` — импорт и чтение карточек через SQLAlchemy.
  Отсутствующие рейтинг Роскачества и рекомендация к подаче сохраняются как
  nullable.
- `src/schemas/search.py` — Pydantic-контракты API и внутренние DTO.
- `src/static` и `src/templates` — браузерный UI. Он использует только
  подробный endpoint и отдельно запрашивает сочетание после успешного поиска.
- `src/migrations/versions` — PostgreSQL-схема, применяемая Alembic.

## API-контракты

### Evaluator

`POST /v1/eval/predict`, `multipart/form-data`, поле `image`, необязательное
поле `top_k`. Это основной контракт для скрипта оценки кейсодержателя.

Успешный ответ содержит ровно один ключ:

```json
{"slug":"aligote-barrel-2024"}
```

Если поиск ничего не вернул:

```json
{"slug":null}
```

Для обратной совместимости локальных клиентов также поддерживается `POST
/search` с полем `photo`.

### UI search

`POST /search/details`, те же входные поля. Ответ `SearchResponse`:

```json
{
  "results": [
    {
      "id": 1,
      "score": 0.81,
      "content": {},
      "metadata": {},
      "link": "https://vino-svoe.ru/wines/aligote-barrel-2024",
      "card": {
        "slug": "aligote-barrel-2024",
        "name": "Алиготе Баррель, 2024",
        "roskachestvo_rating": null,
        "serving_recommendation": null
      }
    }
  ],
  "found": true,
  "margin": 0.02,
  "ocr_matches": 2
}
```

`card` может быть `null`, а новые поля карточки всегда nullable. Текущий
каталог не получает значения, которых нет в исходном JSON.

### Post-search pairing

`GET /pairing/{slug}` возвращает:

```json
{
  "slug": "aligote-barrel-2024",
  "recommendation": "Рыба и молодые сыры.",
  "rationale": "Рекомендация взята из карточки каталога."
}
```

Для карточки без рекомендации используется фиксированная таблица правил по
категории, цвету, сорту и названию вина. Неизвестный slug даёт `404`.

## Запуск

1. Создать `.env` на основе `.env.example`.
2. Запустить `docker compose up --build -d`.
3. Передать базу через Alembic: `uv run alembic upgrade head` при локальном
   запуске; в Docker это делает entrypoint.
4. Один раз построить Qdrant-коллекции из окружения с исходными изображениями:
   `uv run --extra ocr python src/scripts/index_catalog.py`.
5. Открыть API на `http://localhost:8000/`.

JSON-каталог используется только при импорте. Рабочие запросы и индексаторы
читают карточки из PostgreSQL.

## Проверки

```bash
uv run --extra dev pytest -q
uv run --extra dev ruff check src tests
uv run --extra dev mypy src
```

## Ограничения

- production Compose ориентирован на NVIDIA GPU и требует настроенный NVIDIA
  Container Toolkit;
- SLA и качество распознавания требуют отдельного benchmark-прогона на
  размеченном наборе изображений;
- текущий JSON может не содержать новых полей карточки; отсутствующая
  рекомендация к подаче сохраняется как `null`, и UI не показывает пустой блок;
- индексация Qdrant — отдельная тяжёлая операция и не запускается на каждый
  запрос API.
