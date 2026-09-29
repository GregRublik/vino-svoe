import re
from difflib import SequenceMatcher
from itertools import pairwise
from pathlib import Path

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
        "R": "Р",
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
        "r": "р",
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
        if (len(token) >= 5 or (token.isdigit() and len(token) >= 4))
        and token not in OCR_STOP_WORDS
    }

    # Для отдельных OCR-ошибок вроде ``Mepre`` сохраняем дополнительный
    # вариант: латинская ``r`` иногда распознаётся вместо кириллической ``л``.
    # Это не меняет основное исправление R/r -> Р/р и не превращает каждую
    # латинскую букву R в Л.
    for source_token in re.findall(r"[А-Яа-яA-Za-z0-9]+", text):
        if not any(char in "Rr" for char in source_token):
            continue
        translated = source_token.translate(OCR_CONFUSABLES).upper().replace("Ё", "Е")
        for index, char in enumerate(source_token):
            if char not in "Rr":
                continue
            variant_chars = list(translated)
            variant_chars[index] = "Л"
            variant = "".join(variant_chars)
            if len(variant) >= 5 and variant not in OCR_STOP_WORDS:
                tokens.add(variant)

    # OCR often merges a compound name ("ПиноHуар"), while the catalog keeps
    # its words separate ("Пино Нуар"). Keep adjacent-word compounds in both
    # representations so the lexical stage can compare them consistently.
    for left, right in pairwise(raw_tokens):
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
    exact_matches = query_tokens & catalog_tokens
    score = len(exact_matches)
    for query_token in query_tokens - exact_matches:
        if any(
            len(query_token) >= 5
            and abs(len(query_token) - len(catalog_token)) <= 2
            and SequenceMatcher(None, query_token, catalog_token).ratio() >= 0.80
            for catalog_token in catalog_tokens
        ):
            # Альтернативная OCR-нормализация одного слова не должна
            # считаться вторым совпадением, если основной вариант уже совпал.
            if any(
                SequenceMatcher(None, query_token, exact_token).ratio() >= 0.80
                for exact_token in exact_matches
            ):
                continue
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
