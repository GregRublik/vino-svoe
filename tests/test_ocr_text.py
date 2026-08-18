from utils.ocr_text import build_index_text


def test_non_empty_ocr_text_wins():
    assert (
        build_index_text("Абрау-Дюрсо\nБрют", "Вино_белое_брют_Абрау.webp")
        == "Абрау-Дюрсо\nБрют"
    )


def test_empty_ocr_text_falls_back_to_filename():
    assert (
        build_index_text("", "Вино_красное_сухое_Мерло,_цвет_рубиновый.webp")
        == "Вино красное сухое Мерло, цвет рубиновый"
    )


def test_whitespace_ocr_text_falls_back_to_filename():
    assert (
        build_index_text("  \n\t ", "Вино_белое_сухое_Рислинг.webp")
        == "Вино белое сухое Рислинг"
    )


def test_fallback_handles_file_without_extension():
    assert build_index_text("", "Вино_розовое_брют") == "Вино розовое брют"


def test_fallback_keeps_unicode_and_punctuation():
    assert build_index_text("", "Кагор_«Южнобережный»,_2020.webp") == "Кагор «Южнобережный», 2020"
