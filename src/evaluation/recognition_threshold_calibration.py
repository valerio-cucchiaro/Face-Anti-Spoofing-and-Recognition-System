"""
Threshold calibration helpers for recognition evaluation.

Calibration is evaluation-side logic: it chooses thresholds from a
validation split, while recognition metrics only score final decisions.
"""

import numpy as np

from src.evaluation.recognition_metrics import evaluate_recognition_decisions
from src.recognition.recognition_decision import (
    DEFAULT_RELIABILITY_MODE,
    build_open_set_decisions_from_pairwise,
    build_ranked_identity_distances,
    compute_reliability_score,
)


def _unique_threshold_grid(values):
    values = [
        float(value)
        for value in values
        if value is not None and np.isfinite(float(value))
    ]
    if not values:
        return []

    unique_values = np.unique(np.asarray(values, dtype=np.float64))
    if unique_values.size == 1:
        return [float(unique_values[0])]

    return [float(value) for value in unique_values.tolist()]


def _score_calibration_metrics(metrics):
    return (
        metrics["FAR"]
        + metrics["misidentification_rate"]
        + metrics["FRR"],
        metrics["FAR"],
        metrics["misidentification_rate"],
        metrics["FRR"],
    )


def collect_validation_reliability_scores(
    pairwise_analysis,
    reliability_mode=DEFAULT_RELIABILITY_MODE,
):
    """
    Collect one reliability score per validation probe.
    """

    if not pairwise_analysis or not pairwise_analysis.get("available"):
        return []

    identity_columns = pairwise_analysis.get("identity_columns", [])
    scores = []
    for row in pairwise_analysis.get("rows", []):
        ranked_distances = build_ranked_identity_distances(row, identity_columns)
        score = compute_reliability_score(ranked_distances, mode=reliability_mode)
        if score is not None:
            scores.append(score)

    return scores


def calibrate_reliability_threshold(
    pairwise_analysis,
    distance_threshold,
    reliability_mode=DEFAULT_RELIABILITY_MODE,
):
    """
    Select a reliability threshold from a validation pairwise analysis.
    """

    candidate_thresholds = _unique_threshold_grid(
        collect_validation_reliability_scores(
            pairwise_analysis,
            reliability_mode=reliability_mode,
        )
    )

    if not candidate_thresholds:
        return {
            "reliability_mode": reliability_mode,
            "reliability_threshold": None,
            "threshold_rule": "unavailable",
            "candidate_count": 0,
            "metrics": None,
        }

    scored_candidates = []
    for threshold in candidate_thresholds:
        decisions = build_open_set_decisions_from_pairwise(
            pairwise_analysis,
            distance_threshold=distance_threshold,
            reliability_threshold=threshold,
            reliability_mode=reliability_mode,
        )
        metrics = evaluate_recognition_decisions(decisions)
        scored_candidates.append(
            {
                "reliability_threshold": threshold,
                "metrics": metrics,
                "score": _score_calibration_metrics(metrics),
            }
        )

    selected = min(
        scored_candidates,
        key=lambda candidate: (
            candidate["score"],
            -candidate["reliability_threshold"],
        ),
    )

    return {
        "reliability_mode": reliability_mode,
        "reliability_threshold": selected["reliability_threshold"],
        "threshold_rule": "min_far_misidentification_frr",
        "candidate_count": len(candidate_thresholds),
        "metrics": selected["metrics"],
    }


def calibrate_open_set_reliability_thresholds(
    pairwise_analysis,
    distance_threshold,
    reliability_modes=None,
):
    """
    Calibrate one or more reliability thresholds on validation data.
    """

    selected_modes = reliability_modes or [DEFAULT_RELIABILITY_MODE]
    calibrations = {}

    for mode in selected_modes:
        calibrations[mode] = calibrate_reliability_threshold(
            pairwise_analysis,
            distance_threshold=distance_threshold,
            reliability_mode=mode,
        )

    return calibrations
