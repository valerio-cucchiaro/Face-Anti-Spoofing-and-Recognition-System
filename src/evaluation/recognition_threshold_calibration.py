"""
Threshold calibration helpers for recognition evaluation.

Calibration is evaluation-side logic: it chooses thresholds from a
validation split, while recognition metrics only score final decisions.
The official open-set distance calibration follows the course/report
protocol: sweep candidate thresholds, constrain false accepts and wrong
known accepts when possible, and preserve detection/identification
instead of choosing a trivial reject-all operating point.
"""

import numpy as np

from src.evaluation.recognition_metrics import (
    evaluate_recognition_decisions,
    sweep_recognition_thresholds,
)
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


def _score_distance_threshold_metrics(metrics, misidentification_weight):
    """
    Rank fallback distance-threshold candidates by open-set error cost.
    """

    return (
        metrics["FAR"]
        + metrics["FRR"]
        + (misidentification_weight * metrics["misidentification_rate"]),
        metrics["FAR"],
        metrics["misidentification_rate"],
        metrics["FRR"],
    )


def calibrate_open_set_distance_threshold(
    validation_records,
    misidentification_weight=2.0,
    target_far=0.10,
    target_misidentification=0.10,
    minimum_dir=0.50,
    relative_dir_floor=0.80,
):
    """
    Select the LBPH distance threshold from open-set validation records.

    LBPH confidence is a distance: lower scores are better, and increasing
    the threshold accepts more probes. Open-set identification must both
    detect enrolled users and identify them correctly, so the primary
    rule keeps thresholds that satisfy explicit DIR/FAR/MisID constraints:
    accept enough enrolled users, reject impostors, and avoid accepting a
    known probe as the wrong enrolled identity.

    If the validation set is too difficult for those absolute constraints,
    the fallback preserves a fraction of the best observed DIR before
    minimizing the open-set error penalty. This keeps the fallback from
    selecting a low-distance threshold that obtains low FAR only by
    rejecting almost every enrolled probe.
    """

    records = list(validation_records)
    has_known = any(record.get("is_known") is True for record in records)
    has_unknown = any(record.get("is_known") is False for record in records)

    if not records or not has_known or not has_unknown:
        return {
            "threshold": None,
            "threshold_rule": "unavailable",
            "candidate_count": 0,
            "metrics": None,
        }

    sweep_results = sweep_recognition_thresholds(records)
    candidates = sweep_results.get("per_threshold", [])
    if not candidates:
        return {
            "threshold": None,
            "threshold_rule": "unavailable",
            "candidate_count": 0,
            "metrics": None,
        }

    constrained_candidates = [
        metrics
        for metrics in candidates
        if (
            metrics["DIR"] >= minimum_dir
            and metrics["FAR"] <= target_far
            and metrics["misidentification_rate"] <= target_misidentification
        )
    ]

    if constrained_candidates:
        # Course-consistent operating point: among thresholds that satisfy
        # security constraints, keep the best detection/identification rate.
        selected = max(
            constrained_candidates,
            key=lambda metrics: (
                metrics["DIR"],
                -metrics["FAR"],
                -metrics["misidentification_rate"],
                -metrics["FRR"],
                -metrics["threshold"],
            ),
        )
        threshold_rule = "max_dir_with_far_misidentification_constraints"
    else:
        best_dir = max(metrics["DIR"] for metrics in candidates)
        minimum_relative_dir = best_dir * relative_dir_floor
        relative_candidates = [
            metrics
            for metrics in candidates
            if metrics["DIR"] >= minimum_relative_dir
        ]
        # Difficult datasets may have overlapping score distributions. In
        # that case, keep recognition usable relative to the best available
        # DIR, then minimize FAR/FRR/MisID within that usable region.
        selected = min(
            relative_candidates,
            key=lambda metrics: (
                _score_distance_threshold_metrics(
                    metrics,
                    misidentification_weight,
                ),
                -metrics["DIR"],
                metrics["FAR"],
                metrics["misidentification_rate"],
                metrics["threshold"],
            ),
        )
        threshold_rule = "relative_dir_floor_min_far_frr_misidentification_penalty"

    return {
        "threshold": selected["threshold"],
        "threshold_rule": threshold_rule,
        "candidate_count": len(candidates),
        "misidentification_weight": misidentification_weight,
        "target_far": target_far,
        "target_misidentification": target_misidentification,
        "minimum_dir": minimum_dir,
        "relative_dir_floor": relative_dir_floor,
        "metrics": selected,
    }


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
