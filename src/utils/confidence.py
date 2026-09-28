"""Расчёт устойчивой уверенности результата поиска."""


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def calculate_match_confidence(
    visual_score: float | None,
    visual_margin: float | None,
    lexical_matches: int = 0,
    *,
    visual_floor: float = 0.45,
    visual_ceiling: float = 0.85,
    margin_scale: float = 0.05,
    lexical_scale: float = 4.0,
    visual_weight: float = 0.55,
    margin_weight: float = 0.15,
    lexical_weight: float = 0.30,
) -> float:
    """Объединяет независимые признаки совпадения в диапазоне 0..1.

    Cosine-score разных поисковых каналов нельзя напрямую называть
    вероятностью. Здесь учитываются визуальное сходство, отрыв от второго
    кандидата и подтверждённые OCR-совпадения. Если канал недоступен,
    оставшиеся веса нормализуются.
    """
    components: list[tuple[float, float]] = []

    if visual_score is not None:
        score_range = max(visual_ceiling - visual_floor, 1e-6)
        visual_quality = _clamp((visual_score - visual_floor) / score_range)
        components.append((visual_quality, visual_weight))

    if visual_margin is not None:
        margin_quality = _clamp(visual_margin / max(margin_scale, 1e-6))
        components.append((margin_quality, margin_weight))

    if lexical_matches > 0:
        lexical_quality = _clamp(lexical_matches / max(lexical_scale, 1e-6))
        components.append((lexical_quality, lexical_weight))

    if not components:
        return 0.0

    total_weight = sum(weight for _, weight in components)
    confidence = sum(value * weight for value, weight in components) / total_weight
    return round(_clamp(confidence), 4)

