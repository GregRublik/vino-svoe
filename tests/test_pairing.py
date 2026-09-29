from schemas.search import WineCard
from services.pairing import build_pairing


def test_pairing_uses_catalog_recommendation_when_available():
    card = WineCard(
        slug="catalog-wine",
        serving_recommendation="Паста с морепродуктами.",
    )

    result = build_pairing(card)

    assert result.recommendation == "Паста с морепродуктами."
    assert result.rationale == "Рекомендация взята из карточки каталога."


def test_pairing_fallback_is_deterministic_for_red_wine():
    card = WineCard(slug="red-wine", category="Красное")

    first = build_pairing(card)
    second = build_pairing(card)

    assert first == second
    assert first.recommendation == (
        "Говядина на гриле, мясные закуски и выдержанные сыры."
    )
