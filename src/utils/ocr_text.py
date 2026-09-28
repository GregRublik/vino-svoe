from difflib import SequenceMatcher
from pathlib import Path
import re


OCR_CONFUSABLES = str.maketrans(
    {
        "A": "А",
        "B": "В",
        "C": "С",
        "E": "Е",
        "H": "Н",
        "K": "К",
        "M": "М",
        "O": "О",
        "P": "Р",
        "R": "Л",
        "T": "Т",
        "X": "Х",
        "Y": "У",
        "a": "а",
        "b": "в",
        "c": "с",
        "e": "е",
        "h": "н",
        "k": "к",
        "m": "м",
        "o": "о",
        "p": "р",
        "r": "л",
        "t": "т",
        "x": "х",
        "y": "у",
    }
)

OCR_STOP_WORDS = {
    "БЕЛАЯ",
    "БЕЛОЕ",
    "БЕЛЫЙ",
    "ВИНО",
    "ВИНОДЕЛЬНЯ",
    "ГОДА",
    "ГОДУРОЖАЯ",
    "КРАСНОЕ",
    "ПОЛУСЛАДКОЕ",
    "ПОЛУСУХОЕ",
    "РОЗОВОЕ",
    "СЛАДКОЕ",
    "СЕМЕЙНАЯ",
    "СУХОЕ",
}

LATIN_DIGRAPHS = (
    ("shch", "щ"),
    ("sch", "щ"),
    ("yo", "ё"),
    ("yu", "ю"),
    ("ya", "я"),
    ("zh", "ж"),
    ("kh", "х"),
    ("ts", "ц"),
    ("ch", "ч"),
    ("sh", "ш"),
)

LATIN_LETTERS = str.maketrans(
    {
        "a": "а",
        "b": "б",
        "c": "к",
        "d": "д",
        "e": "е",
        "f": "ф",
        "g": "г",
        "h": "х",
        "i": "и",
        "j": "й",
        "k": "к",
        "l": "л",
        "m": "м",
        "n": "н",
        "o": "о",
        "p": "п",
        "q": "к",
        "r": "р",
        "s": "с",
        "t": "т",
        "u": "у",
        "v": "в",
        "w": "в",
        "x": "х",
        "y": "ы",
        "z": "з",
    }
)


CATALOG_FIELDS = (
    "Название вина",
    "Slug",
    "Винодельня",
    "Категория",
    "Цвет",
    "Регион",
    "Сорт винограда",
)


def _normalized_tokens(text: str) -> set[str]:
    normalized = text.translate(OCR_CONFUSABLES).upper().replace("Ё", "Е")
    raw_tokens = re.findall(r"[А-ЯA-Z0-9]+", normalized)
    tokens = {
        token
        for token in raw_tokens
        if (
            len(token) >= 5 or (token.isdigit() and len(token) >= 4)
        )
        and token not in OCR_STOP_WORDS
    }

    # OCR often merges a compound name ("ПиноHуар"), while the catalog keeps
    # its words separate ("Пино Нуар"). Keep adjacent-word compounds in both
    # representations so the lexical stage can compare them consistently.
    for left, right in zip(raw_tokens, raw_tokens[1:]):
        if (
            left not in OCR_STOP_WORDS
            and right not in OCR_STOP_WORDS
            and not left.isdigit()
            and not right.isdigit()
            and 3 <= len(left) <= 4
            and 3 <= len(right) <= 4
        ):
            tokens.add(f"{left}{right}")

    return tokens


def _ocr_tokens(text: str) -> set[str]:
    return _normalized_tokens(text)


def _transliterate_token(token: str) -> str:
    """Переводит slug-токен в кириллицу для сравнения с OCR."""
    value = token.lower()
    for source, target in LATIN_DIGRAPHS:
        value = value.replace(source, target)
    if value.endswith("yy"):
        value = f"{value[:-2]}ый"
    return value.translate(LATIN_LETTERS)


def _catalog_tokens(text: str) -> set[str]:
    tokens = _ocr_tokens(text)
    for token in re.findall(r"[A-Za-z]+", text):
        transliterated = _transliterate_token(token)
        if len(transliterated) >= 5:
            tokens.add(transliterated.upper().replace("Ё", "Е"))
    return tokens


def ocr_catalog_match_score(query: str, catalog_text: str) -> int:
    """Считает совпавшие значимые слова OCR и профиля товара."""
    query_tokens = _ocr_tokens(query)
    catalog_tokens = _catalog_tokens(catalog_text)
    score = 0
    for query_token in query_tokens:
        if query_token in catalog_tokens:
            score += 1
            continue
        if any(
            len(query_token) >= 5
            and abs(len(query_token) - len(catalog_token)) <= 2
            and SequenceMatcher(None, query_token, catalog_token).ratio() >= 0.80
            for catalog_token in catalog_tokens
        ):
            score += 1
    return score


def _result_catalog_text(result) -> str:
    parts: list[str] = []
    for payload in (result.metadata or {}, result.content or {}):
        for field in ("text", "filename", "slug"):
            value = payload.get(field)
            if value and str(value) not in parts:
                parts.append(str(value))
    return "\n".join(parts)


def ocr_result_match_score(result, query: str) -> int:
    return ocr_catalog_match_score(query, _result_catalog_text(result))


def rank_ocr_results(results: list, query: str) -> list:
    """Возвращает только OCR-кандидатов с лексическим совпадением, по силе совпадения."""
    scored = []
    for index, result in enumerate(results):
        score = ocr_result_match_score(result, query)
        if score:
            scored.append((score, -index, result))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    return [result for _, _, result in scored]


def build_catalog_text(record: dict | None) -> str:
    """Собирает короткий текстовый профиль вина из каталога."""
    if not record:
        return ""

    values = [
        str(record[field]).strip()
        for field in CATALOG_FIELDS
        if record.get(field) and str(record[field]).strip()
    ]
    return "\n".join(values)


def build_index_text(
    ocr_text: str,
    filename: str,
    catalog_text: str = "",
) -> str:
    """Формирует текст для векторизации OCR-коллекции."""
    parts = [text.strip() for text in (catalog_text, ocr_text) if text.strip()]
    if parts:
        return "\n".join(parts)
    return Path(filename).stem.replace("_", " ").strip()
