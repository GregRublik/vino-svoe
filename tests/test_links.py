from utils.links import (
    build_wine_link,
    extract_wine_slug,
    slug_from_filename,
    slug_from_link,
)


def test_build_wine_link_from_slug() -> None:
    assert build_wine_link("abrau-dyurso") == "https://vino-svoe.ru/wines/abrau-dyurso"


def test_slug_can_be_extracted_from_filename_and_link() -> None:
    assert slug_from_filename("abrau-dyurso.webp") == "abrau-dyurso"
    assert slug_from_link("https://vino-svoe.ru/wines/abrau-dyurso/") == "abrau-dyurso"


def test_extract_wine_slug_prefers_payload_slug() -> None:
    assert extract_wine_slug(
        content={"slug": "payload-slug", "filename": "filename-slug.webp"},
        link="https://vino-svoe.ru/wines/link-slug",
    ) == "payload-slug"


def test_extract_wine_slug_falls_back_to_link_and_filename() -> None:
    assert extract_wine_slug(link="https://vino-svoe.ru/wines/link-slug") == "link-slug"
    assert extract_wine_slug(content={"filename": "filename-slug.webp"}) == "filename-slug"
