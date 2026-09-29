import pytest

from utils.confidence import calculate_match_confidence


def test_visual_and_ocr_evidence_produce_high_confidence():
    confidence = calculate_match_confidence(
        visual_score=0.78,
        visual_margin=0.08,
        lexical_matches=4,
    )

    assert confidence == pytest.approx(0.901, abs=0.01)


def test_visual_only_weak_match_stays_low():
    confidence = calculate_match_confidence(
        visual_score=0.695,
        visual_margin=0.025,
        lexical_matches=0,
    )

    assert confidence < 0.60


def test_ocr_can_confirm_candidate_without_visual_score():
    confidence = calculate_match_confidence(
        visual_score=None,
        visual_margin=None,
        lexical_matches=4,
    )

    assert confidence == pytest.approx(1.0)
