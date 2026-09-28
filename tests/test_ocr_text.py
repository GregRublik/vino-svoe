from schemas.search import SearchResult
from utils.ocr_text import (
    build_catalog_text,
    build_index_text,
    ocr_catalog_match_score,
    ocr_result_match_score,
    rank_ocr_results,
)


def test_catalog_profile_contains_product_identity():
    assert build_catalog_text(
        {
            "Название вина": "Абрау-Дюрсо Брют",
            "Slug": "abrau-dyurso-bryut",
            "Винодельня": "Абрау-Дюрсо",
            "Категория": "Белое",
        }
    ) == "Абрау-Дюрсо Брют\nabrau-dyurso-bryut\nАбрау-Дюрсо\nБелое"


def test_catalog_profile_is_used_before_filename_fallback():
    assert build_index_text(
        "",
        "fallback.webp",
        "Абрау-Дюрсо Брют\nabrau-dyurso-bryut",
    ) == "Абрау-Дюрсо Брют\nabrau-dyurso-bryut"


def test_catalog_profile_and_ocr_are_combined():
    assert build_index_text("Абрау-Дюрсо", "fallback.webp", "Брют") == "Брют\nАбрау-Дюрсо"


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


def test_ocr_catalog_match_handles_latin_lookalikes():
    assert ocr_catalog_match_score("MACCAHAPA МYCKATEAЬ", "Мускатель белый\nМассандра") == 2


def test_ocr_catalog_match_handles_latin_slug():
    assert ocr_catalog_match_score(
        "MACCAHAPA МYCKATEAЬ",
        "massandra-muskatel-belyy-beloe-sladkoe-16",
    ) == 2


def test_ocr_catalog_match_handles_compound_name_and_year():
    assert ocr_catalog_match_score(
        "ТАБИЯ\nПиноHуар\nполусухое\n2025",
        "Пино Нуар, 2025\npino-nuar-2025\nВинодельня Братьев Мельниковых",
    ) >= 2


def test_ocr_catalog_match_handles_litavshchuk_merlo_ocr_errors():
    assert ocr_catalog_match_score(
        "Семейная винодельня\nЛитавщуков\nMepre",
        "Мерло Литавщук\nmerlo-litavshhuk\n"
        "Литавщук. Litavshchuk vineyards & winery",
    ) >= 2


def test_generic_family_winery_label_does_not_beat_exact_winery():
    query = "Семейная винодельня\nЛитавщуков\nMepre"
    expected = SearchResult(
        id=1,
        score=0.7,
        content={
            "text": "Мерло Литавщук\nmerlo-litavshhuk\n"
            "Литавщук. Litavshchuk vineyards & winery"
        },
        metadata={},
    )
    misleading = SearchResult(
        id=2,
        score=0.8,
        content={
            "text": "Каберне Фран - Мерло - Красностоп\n"
            "Семейная винодельня Михаила Колесникова"
        },
        metadata={},
    )

    assert ocr_result_match_score(expected, query) > ocr_result_match_score(
        misleading, query
    )
    assert rank_ocr_results([misleading, expected], query)[0].id == expected.id


def test_ocr_results_are_lexically_reranked():
    generic = SearchResult(
        id=1,
        score=0.9,
        content={"text": "Массандра Мускат"},
        metadata={"text": "Массандра Мускат"},
    )
    exact = SearchResult(
        id=2,
        score=0.8,
        content={"text": "Мускатель белый Массандра"},
        metadata={"text": "Мускатель белый Массандра"},
    )

    ranked = rank_ocr_results([generic, exact], "MACCAHAPA МYCKATEAЬ")

    assert [result.id for result in ranked] == [2, 1]


def test_ocr_result_match_uses_filename_when_text_is_missing():
    result = SearchResult(
        id=3,
        score=0.8,
        content={"filename": "massandra-muskatel-belyy-beloe-sladkoe-16.webp"},
        metadata={"filename": "massandra-muskatel-belyy-beloe-sladkoe-16.webp"},
    )

    assert ocr_result_match_score(result, "MACCAHAPA МYCKATEAЬ") == 2


def test_generic_winery_label_does_not_beat_product_identity():
    query = "ТАБИЯ\nВИНОДЕЛЬНЯ\n2024\nЦИТРОН\nCYXOE"
    expected = SearchResult(
        id=1,
        score=0.6,
        content={
            "text": (
                "Цитрон\nczitronnyj-magaracha\nТабия\nБелое\n"
                "Светло-золотистый\nКубань\nЦитронный Магарача"
            )
        },
        metadata={},
    )
    misleading = SearchResult(
        id=2,
        score=0.7,
        content={
            "text": (
                "Рислинг, 2024\nrisling-2024\n"
                "Винодельня Братьев Мельниковых\nБелое\n"
                "Бледно-лимонный\nКрым\nРислинг"
            )
        },
        metadata={},
    )

    assert ocr_result_match_score(expected, query) == 2
    assert ocr_result_match_score(misleading, query) == 1
    assert rank_ocr_results([expected, misleading], query)[0].id == expected.id
