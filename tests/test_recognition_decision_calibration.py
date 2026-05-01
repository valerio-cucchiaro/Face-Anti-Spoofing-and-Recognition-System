import pytest

from src.evaluation.recognition_metrics import evaluate_recognition_decisions
from src.evaluation.recognition_threshold_calibration import (
    calibrate_reliability_threshold,
)
from src.recognition.recognition_decision import (
    build_open_set_decisions_from_pairwise,
    build_ranked_identity_distances,
    compute_margin_score,
)


def _pairwise_fixture():
    return {
        "available": True,
        "identity_columns": ["alice", "bob"],
        "rows": [
            {
                "probe_index": 1,
                "split": "validation_known",
                "true_identity": "alice",
                "is_known": True,
                "best_identity": "alice",
                "best_distance": 0.10,
                "second_best_identity": "bob",
                "second_best_distance": 0.80,
                "margin": 0.70,
                "true_identity_rank": 1,
                "distance__alice": 0.10,
                "distance__bob": 0.80,
            },
            {
                "probe_index": 2,
                "split": "validation_known",
                "true_identity": "bob",
                "is_known": True,
                "best_identity": "alice",
                "best_distance": 0.12,
                "second_best_identity": "bob",
                "second_best_distance": 0.13,
                "margin": 0.01,
                "true_identity_rank": 2,
                "distance__alice": 0.12,
                "distance__bob": 0.13,
            },
            {
                "probe_index": 3,
                "split": "validation_unknown",
                "true_identity": "carol",
                "is_known": False,
                "best_identity": "alice",
                "best_distance": 0.90,
                "second_best_identity": "bob",
                "second_best_distance": 1.00,
                "margin": 0.10,
                "true_identity_rank": None,
                "distance__alice": 0.90,
                "distance__bob": 1.00,
            },
        ],
    }


def test_margin_decision_rejects_ambiguous_known_match():
    pairwise_analysis = _pairwise_fixture()
    decisions = build_open_set_decisions_from_pairwise(
        pairwise_analysis,
        distance_threshold=0.50,
        reliability_threshold=0.50,
        reliability_mode="margin",
    )

    assert decisions[0]["accepted"] is True
    assert decisions[0]["final_identity"] == "alice"
    assert decisions[1]["accepted"] is False
    assert decisions[1]["final_identity"] == "Unknown"
    assert decisions[2]["accepted"] is False

    metrics = evaluate_recognition_decisions(decisions)
    assert metrics["correct_identification"] == 1
    assert metrics["false_reject"] == 1
    assert metrics["true_reject"] == 1
    assert metrics["false_accept"] == 0


def test_calibrate_reliability_threshold_uses_validation_decisions():
    pairwise_analysis = _pairwise_fixture()
    calibration = calibrate_reliability_threshold(
        pairwise_analysis,
        distance_threshold=0.50,
        reliability_mode="margin",
    )

    assert calibration["reliability_mode"] == "margin"
    assert calibration["reliability_threshold"] is not None
    assert calibration["candidate_count"] == 3
    assert calibration["metrics"]["false_accept"] == 0


def test_ranked_identity_distances_drive_margin_score():
    row = _pairwise_fixture()["rows"][0]
    ranked_distances = build_ranked_identity_distances(row, ["alice", "bob"])

    assert ranked_distances == [("alice", 0.10), ("bob", 0.80)]
    assert compute_margin_score(ranked_distances) == pytest.approx(0.70)
