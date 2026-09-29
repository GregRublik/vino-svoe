from schemas.search import SearchResult


def result_group_key(result: SearchResult) -> tuple[str, object]:
    """Возвращает ключ товара: slug, а при его отсутствии — id точки."""
    for payload in (result.content, result.metadata):
        slug = payload.get("slug") if isinstance(payload, dict) else None
        if isinstance(slug, str) and slug.strip():
            return ("slug", slug.strip().strip("/"))
    return ("id", result.id)


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
) -> list[SearchResult]:
    """
    Reciprocal Rank Fusion: score(d) = sum(1 / (k + rank)) по всем спискам,
    где документ присутствует. Документ на 1-м ранге в двух списках
    получит больший score, чем документ на 1-м ранге только в одном.

    score в результатах заменяется на fused-значение, сортировка — по убыванию.
    """
    fused: dict[tuple[str, object], tuple[float, SearchResult]] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            score = 1.0 / (k + rank)
            key = result_group_key(result)
            if key in fused:
                prev_score, prev_result = fused[key]
                fused[key] = (prev_score + score, prev_result)
            else:
                fused[key] = (score, result)

    ordered = sorted(fused.items(), key=lambda item: item[1][0], reverse=True)
    return [
        result.model_copy(update={"score": fused_score})
        for _, (fused_score, result) in ordered
    ]
