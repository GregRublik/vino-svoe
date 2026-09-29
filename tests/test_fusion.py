import pytest

from schemas.search import SearchResult
from utils.fusion import reciprocal_rank_fusion


def make_result(doc_id: int, score: float = 0.9) -> SearchResult:
    return SearchResult(
        id=doc_id,
        score=score,
        content={"link": f"https://vino-svoe.ru/wines/{doc_id}"},
        metadata={"filename": f"wine_{doc_id}.webp"},
        link=f"https://vino-svoe.ru/wines/{doc_id}",
    )


def make_slug_result(doc_id: int, slug: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        id=doc_id,
        score=score,
        content={"slug": slug},
        metadata={"filename": f"wine_{doc_id}.webp"},
    )


def test_single_list_passthrough_order():
    results = [make_result(1), make_result(2), make_result(3)]
    fused = reciprocal_rank_fusion([results])
    assert [r.id for r in fused] == [1, 2, 3]
    # score заменён на fused-значение (убывает по рангу)
    assert fused[0].score > fused[1].score > fused[2].score


def test_shared_doc_ranks_above_docs_in_one_list():
    shared = make_result(1)
    list_a = [shared, make_result(2)]
    list_b = [shared, make_result(3)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert fused[0].id == 1
    # 2*(1/(60+1)) > 1/(60+2)
    assert fused[0].score == pytest.approx(2 / 61)


def test_same_slug_from_different_points_is_fused_once():
    first = make_slug_result(1, "massandra-muskat")
    second = make_slug_result(2, "massandra-muskat")

    fused = reciprocal_rank_fusion([[first], [second]])

    assert len(fused) == 1
    assert fused[0].score == pytest.approx(2 / 61)


def test_doc_in_one_list_only_still_ranked():
    list_a = [make_result(1)]
    list_b = [make_result(2)]
    fused = reciprocal_rank_fusion([list_a, list_b])
    assert sorted(r.id for r in fused) == [1, 2]
    assert fused[0].score == fused[1].score  # оба на 1-м ранге своих списков


def test_empty_inputs():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_custom_k_changes_scores_but_not_order():
    results = [make_result(1), make_result(2)]
    default = reciprocal_rank_fusion([results])
    custom = reciprocal_rank_fusion([results], k=1)
    assert [r.id for r in default] == [r.id for r in custom] == [1, 2]
    assert default[0].score != custom[0].score
    # k=1: 1/2 и 1/3
    assert custom[0].score == pytest.approx(1 / 2)
    assert custom[1].score == pytest.approx(1 / 3)


def test_ties_broken_by_input_order():
    # при k=1: A на 1-м ранге списка 1 (1/2), B на 1-м ранге списка 2 (1/2) — ничья,
    # порядок определяется порядком первого появления во входных списках
    list_a = [make_result(1)]
    list_b = [make_result(2)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=1)
    assert [r.id for r in fused] == [1, 2]


def test_fused_score_replaces_original():
    results = [make_result(1, score=0.99)]
    fused = reciprocal_rank_fusion([results])
    assert fused[0].score != 0.99
    assert fused[0].score == pytest.approx(1 / 61)
    # остальные поля не тронуты
    assert fused[0].link == "https://vino-svoe.ru/wines/1"
    assert fused[0].metadata == {"filename": "wine_1.webp"}
