"""Детерминированные гастрономические рекомендации для карточки вина."""

from schemas.search import PairingResponse, WineCard


def _card_profile(card: WineCard) -> str:
    return " ".join(
        value
        for value in (card.category, card.color, card.grape_variety, card.name)
        if value
    ).lower()


def build_pairing(card: WineCard) -> PairingResponse:
    """Возвращает рекомендацию без внешних сервисов и случайности."""
    if card.serving_recommendation:
        return PairingResponse(
            slug=card.slug,
            recommendation=card.serving_recommendation,
            rationale="Рекомендация взята из карточки каталога.",
        )

    profile = _card_profile(card)
    if any(token in profile for token in ("игрист", "брют")):
        recommendation = "Морепродукты, устрицы и лёгкие закуски."
        rationale = "Свежесть и формат игристого лучше всего поддерживают лёгкие блюда."
    elif any(token in profile for token in ("красн", "саперави", "каберне", "сира")):
        recommendation = "Говядина на гриле, мясные закуски и выдержанные сыры."
        rationale = "Плотный профиль красного вина выдерживает насыщенные блюда."
    elif "розов" in profile:
        recommendation = "Лосось, лёгкие салаты и мягкие сыры."
        rationale = "Розовое вино хорошо сочетается с лёгкими блюдами и рыбой."
    elif any(token in profile for token in ("бел", "шардоне", "рислинг", "совин")):
        recommendation = "Рыба, морепродукты и молодые мягкие сыры."
        rationale = "Свежий белый профиль не перегружает деликатные блюда."
    else:
        recommendation = "Мягкие сыры, овощные закуски и белое мясо."
        rationale = "Универсальное сочетание для вина без достаточных стилевых данных."

    return PairingResponse(
        slug=card.slug,
        recommendation=recommendation,
        rationale=rationale,
    )
