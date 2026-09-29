from schemas.search import SearchResult, SearchResultId


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
    fused: dict[SearchResultId, tuple[float, SearchResult]] = {}

    for ranked in ranked_lists:
        for rank, result in enumerate(ranked, start=1):
            score = 1.0 / (k + rank)
            if result.id in fused:
                prev_score, prev_result = fused[result.id]
                fused[result.id] = (prev_score + score, prev_result)
            else:
                fused[result.id] = (score, result)

    ordered = sorted(fused.items(), key=lambda item: item[1][0], reverse=True)
    return [
        result.model_copy(update={"score": fused_score})
        for _, (fused_score, result) in ordered
    ]
