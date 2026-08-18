from pathlib import Path


def build_index_text(ocr_text: str, filename: str) -> str:
    """Текст для векторизации в OCR-коллекцию.

    Приоритет — распознанный OCR-текст фото (совпадает по модальности
    с запросом: OCR текста пользовательского фото). Если текст пустой,
    фолбэк на имя файла (название вина и описание цвета).
    """
    if ocr_text.strip():
        return ocr_text
    return Path(filename).stem.replace("_", " ").strip()
