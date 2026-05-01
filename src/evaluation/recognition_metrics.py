"""
Module: recognition_metrics

Description:
Pure recognition evaluation helpers used by the project evaluators.

This module intentionally contains metric computation and export logic
only:
- single-threshold open-set recognition metrics
- threshold sweep
- EER estimation
- low/mid/high threshold summaries
- optional all gallery vs all probe pairwise analysis
- optional offline system response reliability analysis
- export of report-ready JSON / CSV artifacts

It does not load models, datasets, or print project diagnostics.
"""

import csv
import json
from pathlib import Path

import numpy as np


def _safe_div(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _as_float_array(values):
    return np.asarray(values, dtype=np.float64).reshape(-1)


def _as_int_array(values):
    return np.asarray(values, dtype=np.int64).reshape(-1)


def _build_threshold_grid(records):
    """
    Build a deterministic threshold grid from the observed score range.
    """

    scores = _as_float_array([record["score"] for record in records])
    if scores.size == 0:
        return np.array([], dtype=np.float64)

    unique_scores = np.unique(scores)

    if unique_scores.size == 1:
        base = float(unique_scores[0])
        epsilon = max(abs(base) * 1e-6, 1e-9)
        return np.array(
            [base - epsilon, base, base + epsilon],
            dtype=np.float64,
        )

    lower = np.nextafter(float(unique_scores[0]), -np.inf)
    upper = np.nextafter(float(unique_scores[-1]), np.inf)
    return np.unique(np.concatenate(([lower], unique_scores, [upper]))).astype(
        np.float64
    )


def _pick_threshold_result(per_threshold, target_threshold):
    """
    Return the sweep row closest to the requested threshold.
    """

    if not per_threshold or target_threshold is None:
        return None

    return min(
        per_threshold,
        key=lambda item: abs(float(item["threshold"]) - float(target_threshold)),
    )


def _build_summary_thresholds(thresholds):
    """
    Select representative low/mid/high thresholds from the sweep grid.
    """

    if thresholds.size == 0:
        return {"low": None, "mid": None, "high": None}

    return {
        "low": float(np.quantile(thresholds, 0.25)),
        "mid": float(np.quantile(thresholds, 0.50)),
        "high": float(np.quantile(thresholds, 0.75)),
    }


def _validate_records(records):
    """
    Validate the normalized record schema expected by the recognition metrics.

    The evaluator should pass one flat record per probe sample with the
    ``is_known``, ``true_id``, ``predicted_id``, and ``score`` keys.
    """

    required_keys = {"is_known", "true_id", "predicted_id", "score"}

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Recognition record #{index} must be a dictionary.")

        missing_keys = required_keys.difference(record)
        if missing_keys:
            missing_list = ", ".join(sorted(missing_keys))
            raise ValueError(
                f"Recognition record #{index} is missing required keys: {missing_list}"
            )

        if not isinstance(record["is_known"], bool):
            raise ValueError(
                f"Recognition record #{index} has an invalid 'is_known' flag."
            )

        try:
            float(record["score"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Recognition record #{index} has a non-numeric 'score'."
            ) from exc

        if record["is_known"]:
            if record["true_id"] is None:
                raise ValueError(
                    f"Recognition record #{index} is known but missing 'true_id'."
                )
            if record["predicted_id"] is None:
                raise ValueError(
                    f"Recognition record #{index} is known but missing 'predicted_id'."
                )


def evaluate_recognition_at_threshold(records, threshold):
    """
    Evaluate open-set recognition metrics at one threshold.

    The project convention is:
    - score < threshold -> accepted identity
    - score >= threshold -> rejected as Unknown
    """

    normalized_records = list(records)
    _validate_records(normalized_records)
    scores = np.array([float(record["score"]) for record in normalized_records])

    if len(normalized_records) != scores.size:
        raise ValueError("Recognition records are malformed.")

    total_known = 0
    total_unknown = 0
    correct_identification = 0
    false_reject = 0
    misidentification = 0
    false_accept = 0
    true_reject = 0

    for record in normalized_records:
        is_known = bool(record.get("is_known", False))
        true_id = record.get("true_id")
        predicted_id = record.get("predicted_id")
        score = float(record["score"])
        accepted = score < threshold

        if is_known:
            total_known += 1
            if accepted:
                if predicted_id == true_id:
                    correct_identification += 1
                else:
                    misidentification += 1
            else:
                false_reject += 1
        else:
            total_unknown += 1
            if accepted:
                false_accept += 1
            else:
                true_reject += 1

    dir_rate = _safe_div(correct_identification, total_known)
    far = _safe_div(false_accept, total_unknown)
    frr = _safe_div(false_reject, total_known)
    gar = 1.0 - frr
    misidentification_rate = _safe_div(misidentification, total_known)
    return {
        "threshold": float(threshold),
        "total_known": total_known,
        "total_unknown": total_unknown,
        "correct_identification": correct_identification,
        "false_reject": false_reject,
        "misidentification": misidentification,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "DIR": dir_rate,
        "FAR": far,
        "FRR": frr,
        "GAR": gar,
        "misidentification_rate": misidentification_rate,
    }


def evaluate_recognition_decisions(records):
    """
    Evaluate open-set recognition metrics from already finalized decisions.

    Decision records are expected to include:
    - ``is_known``
    - ``true_id`` for known probes
    - ``final_identity``
    - ``accepted``

    This lets distance-only, distance+margin, distance+SRR, and future
    runtime-like decision policies share the same metric computation.
    """

    total_known = 0
    total_unknown = 0
    correct_identification = 0
    false_reject = 0
    misidentification = 0
    false_accept = 0
    true_reject = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Recognition decision #{index} must be a dictionary.")

        is_known = bool(record.get("is_known", False))
        accepted = bool(record.get("accepted", False))
        true_id = record.get("true_id")
        final_identity = record.get("final_identity")

        if is_known:
            total_known += 1
            if accepted:
                if final_identity == true_id:
                    correct_identification += 1
                else:
                    misidentification += 1
            else:
                false_reject += 1
        else:
            total_unknown += 1
            if accepted:
                false_accept += 1
            else:
                true_reject += 1

    frr = _safe_div(false_reject, total_known)
    return {
        "total_known": total_known,
        "total_unknown": total_unknown,
        "correct_identification": correct_identification,
        "false_reject": false_reject,
        "misidentification": misidentification,
        "false_accept": false_accept,
        "true_reject": true_reject,
        "DIR": _safe_div(correct_identification, total_known),
        "FAR": _safe_div(false_accept, total_unknown),
        "FRR": frr,
        "GAR": 1.0 - frr,
        "misidentification_rate": _safe_div(misidentification, total_known),
    }


def sweep_recognition_thresholds(records, thresholds=None):
    """
    Evaluate recognition metrics over a threshold sweep.
    """

    normalized_records = list(records)
    if thresholds is None:
        thresholds = _build_threshold_grid(normalized_records)
    else:
        thresholds = np.unique(_as_float_array(thresholds))

    per_threshold = [
        evaluate_recognition_at_threshold(normalized_records, threshold)
        for threshold in thresholds
    ]

    roc_points = [
        {
            "threshold": item["threshold"],
            "far": item["FAR"],
            "gar": item["GAR"],
        }
        for item in per_threshold
    ]
    det_points = [
        {
            "threshold": item["threshold"],
            "far": item["FAR"],
            "frr": item["FRR"],
        }
        for item in per_threshold
    ]

    far_values = np.array([item["FAR"] for item in per_threshold], dtype=np.float64)
    frr_values = np.array([item["FRR"] for item in per_threshold], dtype=np.float64)
    eer_gap = np.abs(far_values - frr_values)

    if eer_gap.size == 0:
        eer_value = None
        eer_threshold = None
    else:
        eer_index = int(np.argmin(eer_gap))
        eer_value = float((far_values[eer_index] + frr_values[eer_index]) / 2.0)
        eer_threshold = float(per_threshold[eer_index]["threshold"])

    return {
        "thresholds": [float(value) for value in thresholds.tolist()],
        "per_threshold": per_threshold,
        "roc_points": roc_points,
        "det_points": det_points,
        "eer_value": eer_value,
        "eer_threshold": eer_threshold,
        "eer_gap": float(eer_gap.min()) if eer_gap.size else None,
        "summary_thresholds": _build_summary_thresholds(thresholds),
    }


def compute_recognition_eer(sweep_results):
    """
    Return the EER information already derived from the sweep.
    """

    if not sweep_results:
        return {"eer_value": None, "eer_threshold": None}

    return {
        "eer_value": sweep_results.get("eer_value"),
        "eer_threshold": sweep_results.get("eer_threshold"),
    }


def build_recognition_threshold_summary(
    sweep_results,
    low_t=None,
    mid_t=None,
    high_t=None,
):
    """
    Pick representative rows from the sweep for low / mid / high thresholds.
    """

    if not sweep_results:
        return {
            "low": None,
            "mid": None,
            "high": None,
            "selected_thresholds": {
                "low": low_t,
                "mid": mid_t,
                "high": high_t,
            },
        }

    per_threshold = sweep_results.get("per_threshold", [])
    summary_thresholds = sweep_results.get("summary_thresholds", {})

    if low_t is None:
        low_t = summary_thresholds.get("low")
    if mid_t is None:
        mid_t = summary_thresholds.get("mid")
    if high_t is None:
        high_t = summary_thresholds.get("high")

    return {
        "low": _pick_threshold_result(per_threshold, low_t),
        "mid": _pick_threshold_result(per_threshold, mid_t),
        "high": _pick_threshold_result(per_threshold, high_t),
        "selected_thresholds": {
            "low": low_t,
            "mid": mid_t,
            "high": high_t,
        },
    }


def _default_recognition_metrics_root():
    return Path(__file__).resolve().parents[2] / "logs" / "metrics" / "recognition"


def _json_ready_report(report):
    """
    Convert the in-memory recognition report into a JSON-serializable object.
    """

    pairwise_analysis = report.get("pairwise_analysis")
    pairwise_json = None
    if pairwise_analysis is not None:
        pairwise_json = {
            key: value
            for key, value in pairwise_analysis.items()
            if key != "rows"
        }

    doddington_analysis = report.get("doddington_analysis")
    srr_analysis = report.get("srr_analysis")
    reliability_gated_report = report.get("reliability_gated_report")

    return {
        "selected_threshold": report.get("selected_threshold"),
        "threshold_source": report.get("threshold_source"),
        "evaluation_scope": report.get("evaluation_scope"),
        "selected_metrics": report.get("selected_metrics"),
        "eer_results": report.get("eer_results"),
        "threshold_summary": report.get("threshold_summary"),
        "pairwise_analysis": pairwise_json,
        "doddington_analysis": doddington_analysis,
        "srr_analysis": srr_analysis,
        "reliability_gated_report": reliability_gated_report,
        "sweep_results": {
            "thresholds": report.get("sweep_results", {}).get("thresholds", []),
            "roc_points": report.get("sweep_results", {}).get("roc_points", []),
            "det_points": report.get("sweep_results", {}).get("det_points", []),
        },
    }


def _write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _chi_square_alt_distance_matrix(gallery_matrix, probe_histogram):
    """
    Compute LBPH-style pairwise distances between one probe histogram and
    a matrix of gallery histograms.

    The formula mirrors the confidence used by LBPH in OpenCV's default
    "alt" chi-square distance.
    """
    epsilon = 1e-10
    difference = gallery_matrix - probe_histogram
    denominator = gallery_matrix + probe_histogram + epsilon
    return 2.0 * np.sum((difference * difference) / denominator, axis=1)


def _safe_numeric(value, default):
    """
    Return ``default`` when ``value`` is missing, otherwise coerce it to float.
    """

    if value is None:
        return float(default)
    return float(value)


def _quantile_or_default(values, quantile, default=None):
    """
    Return a quantile from a non-empty sequence, otherwise ``default``.
    """

    if not values:
        return default

    return float(np.quantile(np.asarray(values, dtype=np.float64), quantile))


def build_cms_curve(rank_values, max_rank=None):
    """
    Build a CMS/CMC curve from one list of true identity ranks.

    The curve is computed as the cumulative probability of the rank
    being less than or equal to k for every k in the selected range.
    """

    normalized_ranks = _as_int_array(rank_values)
    if normalized_ranks.size == 0:
        return {
            "available": False,
            "max_rank": 0,
            "points": [],
        }

    if max_rank is None:
        max_rank = int(np.max(normalized_ranks))

    max_rank = int(max(0, max_rank))
    if max_rank == 0:
        return {
            "available": False,
            "max_rank": 0,
            "points": [],
        }

    points = []
    for rank in range(1, max_rank + 1):
        hits = int(np.sum(normalized_ranks <= rank))
        points.append(
            {
                "rank": rank,
                "cms": _safe_div(hits, normalized_ranks.size),
                "hit_count": hits,
                "total": int(normalized_ranks.size),
            }
        )

    return {
        "available": True,
        "max_rank": max_rank,
        "points": points,
    }


def _select_pairwise_preview_rows(rows, limit=6):
    """
    Select a compact set of representative pairwise rows for the report.

    The preview is intentionally small so the report can show a visual
    summary of the matrix without repeating the full probe-by-template
    export.
    """

    if not rows or limit <= 0:
        return []

    known_rows = [row for row in rows if row.get("is_known")]
    unknown_rows = [row for row in rows if not row.get("is_known")]

    candidates = []

    if known_rows:
        known_rows_sorted = sorted(
            known_rows,
            key=lambda row: (
                _safe_numeric(row.get("true_identity_rank"), np.inf),
                _safe_numeric(row.get("margin"), np.inf),
            ),
        )
        candidates.extend(
            [
                ("best_rank", known_rows_sorted[0]),
                ("median_rank", known_rows_sorted[len(known_rows_sorted) // 2]),
                (
                    "hardest_rank",
                    known_rows_sorted[-1],
                ),
                (
                    "lowest_margin",
                    min(
                        known_rows,
                        key=lambda row: (
                            _safe_numeric(row.get("margin"), np.inf),
                            _safe_numeric(row.get("true_identity_rank"), np.inf),
                        ),
                    ),
                ),
                (
                    "highest_margin",
                    max(
                        known_rows,
                        key=lambda row: (
                            _safe_numeric(row.get("margin"), -np.inf),
                            -_safe_numeric(row.get("true_identity_rank"), -np.inf),
                        ),
                    ),
                ),
            ]
        )

    if unknown_rows:
        candidates.extend(
            [
                (
                    "unknown_best_distance",
                    min(
                        unknown_rows,
                        key=lambda row: (
                            _safe_numeric(row.get("best_distance"), np.inf),
                            _safe_numeric(row.get("margin"), np.inf),
                        ),
                    ),
                ),
                (
                    "unknown_low_margin",
                    min(
                        unknown_rows,
                        key=lambda row: (
                            _safe_numeric(row.get("margin"), np.inf),
                            _safe_numeric(row.get("best_distance"), np.inf),
                        ),
                    ),
                ),
            ]
        )

    preview_rows = []
    seen_probe_indices = set()
    compact_fields = (
        "probe_index",
        "split",
        "true_identity",
        "is_known",
        "best_identity",
        "best_distance",
        "second_best_identity",
        "second_best_distance",
        "margin",
        "true_identity_rank",
        "true_identity_distance",
        "accepted_at_threshold",
        "final_identity",
        "decision_correct",
    )

    for example_type, row in candidates:
        probe_index = row.get("probe_index")
        if probe_index in seen_probe_indices:
            continue
        seen_probe_indices.add(probe_index)
        preview_row = {"example_type": example_type}
        for field in compact_fields:
            preview_row[field] = row.get(field)
        preview_rows.append(preview_row)
        if len(preview_rows) >= limit:
            break

    return preview_rows


def _build_distance_order_from_row(row, identity_columns):
    """
    Rebuild the ordered identity-distance list for one pairwise row.
    """

    ordered_distances = []
    for identity_name in identity_columns:
        distance_key = f"distance__{identity_name}"
        if distance_key not in row:
            continue
        ordered_distances.append((identity_name, float(row[distance_key])))

    ordered_distances.sort(key=lambda item: (item[1], item[0]))
    return ordered_distances


def _build_threshold_grid_from_values(values):
    """
    Build a deterministic threshold grid from a list of scalar values.
    """

    scores = _as_float_array(values)
    if scores.size == 0:
        return np.array([], dtype=np.float64)

    unique_scores = np.unique(scores)

    if unique_scores.size == 1:
        base = float(unique_scores[0])
        epsilon = max(abs(base) * 1e-6, 1e-9)
        return np.array([base - epsilon, base, base + epsilon], dtype=np.float64)

    lower = np.nextafter(float(unique_scores[0]), -np.inf)
    upper = np.nextafter(float(unique_scores[-1]), np.inf)
    return np.unique(np.concatenate(([lower], unique_scores, [upper]))).astype(
        np.float64
    )


def _evaluate_binary_threshold(scores, positive_labels, threshold):
    """
    Evaluate a binary threshold where higher scores indicate reliability.
    """

    score_array = _as_float_array(scores)
    positive_array = np.asarray(positive_labels, dtype=bool).reshape(-1)

    if score_array.size != positive_array.size:
        raise ValueError("Scores and labels must have the same length.")

    predicted_positive = score_array >= float(threshold)

    true_positive = int(np.sum(predicted_positive & positive_array))
    false_positive = int(np.sum(predicted_positive & ~positive_array))
    false_negative = int(np.sum(~predicted_positive & positive_array))
    true_negative = int(np.sum(~predicted_positive & ~positive_array))

    positive_total = int(np.sum(positive_array))
    negative_total = int(score_array.size - positive_total)

    precision = _safe_div(true_positive, true_positive + false_positive)
    recall = _safe_div(true_positive, positive_total)
    false_accept_rate = _safe_div(false_positive, negative_total)
    false_reject_rate = _safe_div(false_negative, positive_total)
    balanced_accuracy = 1.0 - 0.5 * (false_accept_rate + false_reject_rate)
    f1_denominator = 2 * true_positive + false_positive + false_negative
    f1_score = _safe_div(2 * true_positive, f1_denominator)

    return {
        "threshold": float(threshold),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "precision": precision,
        "recall": recall,
        "false_accept_rate": false_accept_rate,
        "false_reject_rate": false_reject_rate,
        "balanced_accuracy": balanced_accuracy,
        "f1_score": f1_score,
    }


def _build_binary_threshold_sweep(scores, positive_labels):
    """
    Evaluate a reliability score over a deterministic threshold grid.
    """

    score_array = _as_float_array(scores)
    positive_array = np.asarray(positive_labels, dtype=bool).reshape(-1)

    if score_array.size != positive_array.size:
        raise ValueError("Scores and labels must have the same length.")

    thresholds = _build_threshold_grid_from_values(score_array)
    per_threshold = [
        _evaluate_binary_threshold(score_array, positive_array, threshold)
        for threshold in thresholds
    ]

    if not per_threshold:
        return {
            "thresholds": [],
            "per_threshold": [],
            "selected_threshold": None,
            "selected_threshold_row": None,
            "summary_thresholds": {"low": None, "mid": None, "high": None},
        }

    selected_index = min(
        range(len(per_threshold)),
        key=lambda index: (
            per_threshold[index]["false_accept_rate"]
            + per_threshold[index]["false_reject_rate"],
            abs(
                per_threshold[index]["false_accept_rate"]
                - per_threshold[index]["false_reject_rate"]
            ),
            -per_threshold[index]["threshold"],
        ),
    )

    threshold_values = np.asarray(thresholds, dtype=np.float64)
    summary_thresholds = _build_summary_thresholds(threshold_values)

    return {
        "thresholds": [float(value) for value in threshold_values.tolist()],
        "per_threshold": per_threshold,
        "selected_threshold": float(per_threshold[selected_index]["threshold"]),
        "selected_threshold_row": per_threshold[selected_index],
        "summary_thresholds": summary_thresholds,
    }


def _srr_relative_distance(sorted_distances):
    """
    Relative-distance SRR variant.
    """

    if not sorted_distances:
        return None

    if len(sorted_distances) == 1:
        return 1.0

    first_distance = float(sorted_distances[0][1])
    second_distance = float(sorted_distances[1][1])
    last_distance = float(sorted_distances[-1][1])

    denominator = max(last_distance - first_distance, 1e-12)
    score = (second_distance - first_distance) / denominator
    return float(np.clip(score, 0.0, 1.0))


def _srr_density_ratio(sorted_distances):
    """
    Density-ratio SRR variant.
    """

    if not sorted_distances:
        return None

    first_distance = float(sorted_distances[0][1])
    radius = 2.0 * first_distance
    neighbourhood_size = sum(1 for _, distance in sorted_distances if distance < radius)
    return float(1.0 - _safe_div(neighbourhood_size, len(sorted_distances)))


def _srr_adaptive_density_ratio(sorted_distances):
    """
    Adaptive density-ratio SRR variant.

    The cloud radius is expanded when the first two candidates are close
    together, which makes ambiguous responses score lower.
    """

    if not sorted_distances:
        return None

    first_distance = float(sorted_distances[0][1])
    if len(sorted_distances) == 1:
        return 1.0

    second_distance = float(sorted_distances[1][1])
    gap = max(second_distance - first_distance, 0.0)
    adaptive_radius = 2.0 * first_distance * (1.0 + 1.0 / (1.0 + gap))
    neighbourhood_size = sum(
        1 for _, distance in sorted_distances if distance < adaptive_radius
    )
    return float(1.0 - _safe_div(neighbourhood_size, len(sorted_distances)))


def build_system_response_reliability_analysis(
    pairwise_analysis,
    threshold=None,
    top_n=3,
):
    """
    Build an offline System Response Reliability summary.

    The SRR is computed as a descriptive confidence measure over the
    ordered gallery distances of each probe. The selected variant uses a
    data-driven threshold calibrated on the same evaluation split.
    """

    if not pairwise_analysis or not pairwise_analysis.get("available"):
        return {
            "available": False,
            "reason": "pairwise_unavailable",
            "threshold": threshold,
            "selected_variant": "adaptive_density_ratio",
            "variants": {},
            "per_probe": [],
            "summary_counts": {},
        }

    rows = pairwise_analysis.get("rows", [])
    identity_columns = list(pairwise_analysis.get("identity_columns", []))
    if not rows or not identity_columns:
        return {
            "available": False,
            "reason": "no_pairwise_rows",
            "threshold": threshold,
            "selected_variant": "adaptive_density_ratio",
            "variants": {},
            "per_probe": [],
            "summary_counts": {},
        }

    if threshold is None:
        threshold = pairwise_analysis.get("threshold")

    variant_functions = {
        "relative_distance": _srr_relative_distance,
        "density_ratio": _srr_density_ratio,
        "adaptive_density_ratio": _srr_adaptive_density_ratio,
    }

    per_probe_rows = []
    variant_scores = {name: [] for name in variant_functions}
    reliability_labels = []

    for row in rows:
        sorted_distances = _build_distance_order_from_row(row, identity_columns)
        selected_variant_scores = {}
        for variant_name, variant_fn in variant_functions.items():
            score = variant_fn(sorted_distances)
            if score is None:
                score = 0.0
            score = float(np.clip(score, 0.0, 1.0))
            variant_scores[variant_name].append(score)
            selected_variant_scores[variant_name] = score

        reliability_label = bool(row.get("decision_correct"))
        if row.get("decision_correct") is None:
            reliability_label = bool(
                row.get("is_known") and row.get("true_identity_rank") == 1
            )
        reliability_labels.append(reliability_label)

        per_probe_rows.append(
            {
                "probe_index": row.get("probe_index"),
                "split": row.get("split"),
                "true_identity": row.get("true_identity"),
                "is_known": row.get("is_known"),
                "best_identity": row.get("best_identity"),
                "best_distance": row.get("best_distance"),
                "second_best_identity": row.get("second_best_identity"),
                "second_best_distance": row.get("second_best_distance"),
                "margin": row.get("margin"),
                "true_identity_rank": row.get("true_identity_rank"),
                "decision_correct": row.get("decision_correct"),
                "relative_distance": selected_variant_scores["relative_distance"],
                "density_ratio": selected_variant_scores["density_ratio"],
                "adaptive_density_ratio": selected_variant_scores[
                    "adaptive_density_ratio"
                ],
            }
        )

    variant_reports = {}
    for variant_name, scores in variant_scores.items():
        sweep = _build_binary_threshold_sweep(scores, reliability_labels)
        selected_row = sweep.get("selected_threshold_row") or {}
        score_array = _as_float_array(scores)
        if score_array.size > 0:
            stats = {
                "count": int(score_array.size),
                "mean": float(np.mean(score_array)),
                "std": float(np.std(score_array)) if score_array.size > 1 else 0.0,
                "min": float(np.min(score_array)),
                "max": float(np.max(score_array)),
            }
        else:
            stats = None

        variant_reports[variant_name] = {
            "scores": scores,
            "stats": stats,
            "threshold": sweep.get("selected_threshold"),
            "threshold_row": selected_row,
            "threshold_sweep": {
                "thresholds": sweep.get("thresholds", []),
                "per_threshold": sweep.get("per_threshold", []),
                "summary_thresholds": sweep.get("summary_thresholds", {}),
            },
        }

    selected_variant = "adaptive_density_ratio"
    selected_scores = variant_scores[selected_variant]
    selected_threshold = variant_reports[selected_variant]["threshold"]
    selected_threshold_row = variant_reports[selected_variant]["threshold_row"]

    selected_per_probe = []
    for row, selected_score, reliability_label in zip(
        per_probe_rows,
        selected_scores,
        reliability_labels,
    ):
        selected_reliable = bool(
            selected_threshold is not None and selected_score >= selected_threshold
        )
        selected_per_probe.append(
            {
                **row,
                "selected_variant": selected_variant,
                "selected_score": float(selected_score),
                "selected_threshold": selected_threshold,
                "selected_reliable": selected_reliable,
                "reliability_label": reliability_label,
                "reliability_gap": (
                    float(selected_score - selected_threshold)
                    if selected_threshold is not None
                    else None
                ),
            }
        )

    selected_correct = sum(
        1 for row in selected_per_probe if row["selected_reliable"] and row["reliability_label"]
    )
    selected_incorrect = sum(
        1
        for row in selected_per_probe
        if row["selected_reliable"] and not row["reliability_label"]
    )
    selected_false_rejects = sum(
        1
        for row in selected_per_probe
        if not row["selected_reliable"] and row["reliability_label"]
    )
    selected_true_rejects = sum(
        1
        for row in selected_per_probe
        if not row["selected_reliable"] and not row["reliability_label"]
    )

    summary_counts = {
        "total_probes": len(selected_per_probe),
        "known_probes": sum(1 for row in selected_per_probe if row["is_known"]),
        "unknown_probes": sum(1 for row in selected_per_probe if not row["is_known"]),
        "reliable_responses": sum(1 for row in selected_per_probe if row["selected_reliable"]),
        "unreliable_responses": sum(
            1 for row in selected_per_probe if not row["selected_reliable"]
        ),
        "reliable_correct": selected_correct,
        "reliable_incorrect": selected_incorrect,
        "unreliable_correct": selected_true_rejects,
        "unreliable_incorrect": selected_false_rejects,
        "threshold": float(threshold) if threshold is not None else None,
        "selected_variant": selected_variant,
    }

    return {
        "available": True,
        "reason": None,
        "threshold": float(threshold) if threshold is not None else None,
        "selected_variant": selected_variant,
        "selected_threshold": selected_threshold,
        "selected_threshold_row": selected_threshold_row,
        "variants": variant_reports,
        "per_probe": selected_per_probe,
        "summary_counts": summary_counts,
        "top_n": int(top_n),
    }


def build_doddington_zoo_analysis(pairwise_analysis, threshold=None, top_n=3):
    """
    Build a lightweight Doddington-style per-identity summary.

    The summary is intentionally descriptive rather than prescriptive: it
    highlights identities that behave like sheep, goats, lambs, or wolves
    under the current recognition setup without changing the base metrics.
    """

    if not pairwise_analysis or not pairwise_analysis.get("available"):
        return {
            "available": False,
            "reason": "pairwise_unavailable",
            "threshold": threshold,
            "per_identity": [],
            "categories": {},
        }

    rows = pairwise_analysis.get("rows", [])
    identity_columns = list(pairwise_analysis.get("identity_columns", []))
    if not rows or not identity_columns:
        return {
            "available": False,
            "reason": "no_pairwise_rows",
            "threshold": threshold,
            "per_identity": [],
            "categories": {},
        }

    if threshold is None:
        threshold = pairwise_analysis.get("threshold")

    total_probes = len(rows)
    total_known_probes = sum(1 for row in rows if row.get("is_known"))
    total_unknown_probes = total_probes - total_known_probes

    identity_rows = []
    for identity_name in identity_columns:
        known_rows = [
            row
            for row in rows
            if row.get("is_known") and row.get("true_identity") == identity_name
        ]
        other_rows = [
            row for row in rows if row.get("true_identity") != identity_name
        ]
        accepted_rows = [
            row
            for row in rows
            if row.get("accepted_at_threshold")
            and row.get("final_identity") == identity_name
        ]
        lure_rows = [
            row
            for row in rows
            if row.get("best_identity") == identity_name
            and row.get("true_identity") != identity_name
        ]

        known_count = len(known_rows)
        other_count = len(other_rows)
        correct_rank_1 = sum(1 for row in known_rows if row.get("true_identity_rank") == 1)
        correct_rank_3 = sum(
            1
            for row in known_rows
            if row.get("true_identity_rank") is not None
            and row.get("true_identity_rank") <= 3
        )
        correct_rank_5 = sum(
            1
            for row in known_rows
            if row.get("true_identity_rank") is not None
            and row.get("true_identity_rank") <= 5
        )

        genuine_accepts = sum(
            1
            for row in known_rows
            if row.get("accepted_at_threshold")
            and row.get("final_identity") == identity_name
        )
        false_rejects = sum(
            1
            for row in known_rows
            if row.get("accepted_at_threshold") is not None
            and row.get("final_identity") == "Unknown"
        )
        misidentifications_out = sum(
            1
            for row in known_rows
            if row.get("accepted_at_threshold")
            and row.get("final_identity") not in (identity_name, "Unknown")
        )

        impostor_accepts = sum(
            1 for row in accepted_rows if row.get("true_identity") != identity_name
        )
        unknown_accepts = sum(1 for row in accepted_rows if not row.get("is_known"))
        wrong_known_accepts = sum(
            1
            for row in accepted_rows
            if row.get("is_known") and row.get("true_identity") != identity_name
        )
        best_match_from_others = len(lure_rows)

        sheep_score = (
            2.0 * _safe_div(correct_rank_1, known_count)
            + _safe_div(correct_rank_3, known_count)
            + _safe_div(correct_rank_5, known_count)
            + _safe_div(genuine_accepts, known_count)
            - _safe_div(false_rejects, known_count)
            - _safe_div(misidentifications_out, known_count)
        )
        goat_score = (
            _safe_div(false_rejects, known_count)
            + _safe_div(misidentifications_out, known_count)
            + (1.0 - _safe_div(correct_rank_1, known_count))
            + (1.0 - _safe_div(genuine_accepts, known_count))
        )
        lamb_score = (
            2.0 * _safe_div(impostor_accepts, other_count)
            + _safe_div(unknown_accepts, total_unknown_probes)
            + _safe_div(wrong_known_accepts, len(rows))
        )
        wolf_score = (
            _safe_div(best_match_from_others, other_count)
            + _safe_div(impostor_accepts, other_count)
            + _safe_div(wrong_known_accepts, len(rows))
        )

        row = {
            "identity": identity_name,
            "known_probe_count": known_count,
            "other_probe_count": other_count,
            "genuine_accepts": genuine_accepts if threshold is not None else None,
            "false_rejects": false_rejects if threshold is not None else None,
            "misidentifications_out": misidentifications_out
            if threshold is not None
            else None,
            "impostor_accepts": impostor_accepts if threshold is not None else None,
            "unknown_accepts": unknown_accepts if threshold is not None else None,
            "wrong_known_accepts": wrong_known_accepts
            if threshold is not None
            else None,
            "best_match_from_others": best_match_from_others,
            "rank_1_hits": correct_rank_1,
            "rank_3_hits": correct_rank_3,
            "rank_5_hits": correct_rank_5,
            "rank_1_accuracy": _safe_div(correct_rank_1, known_count),
            "rank_3_accuracy": _safe_div(correct_rank_3, known_count),
            "rank_5_accuracy": _safe_div(correct_rank_5, known_count),
            "genuine_accept_rate": _safe_div(genuine_accepts, known_count),
            "genuine_reject_rate": _safe_div(false_rejects, known_count),
            "misidentification_rate": _safe_div(misidentifications_out, known_count),
            "impostor_accept_rate": _safe_div(impostor_accepts, other_count),
            "unknown_accept_rate": _safe_div(unknown_accepts, total_unknown_probes),
            "lure_rate": _safe_div(best_match_from_others, other_count),
            "scores": {
                "sheep": float(sheep_score),
                "goat": float(goat_score),
                "lamb": float(lamb_score),
                "wolf": float(wolf_score),
            },
        }

        identity_rows.append(row)

    for row in identity_rows:
        score_items = sorted(
            row["scores"].items(),
            key=lambda item: (-item[1], item[0]),
        )
        dominant_tag, dominant_score = score_items[0]
        second_tag, second_score = score_items[1]
        row["dominant_tag"] = dominant_tag
        row["dominant_score"] = float(dominant_score)
        row["second_tag"] = second_tag
        row["second_score"] = float(second_score)
        row["score_gap"] = float(dominant_score - second_score)

    categories = {}
    for category_name in ("sheep", "goat", "lamb", "wolf"):
        category_rows = [
            row for row in identity_rows if row["dominant_tag"] == category_name
        ]
        category_rows = sorted(
            category_rows,
            key=lambda row: (-row["dominant_score"], -row["score_gap"], row["identity"]),
        )
        categories[category_name] = {
            "count": len(category_rows),
            "identities": [row["identity"] for row in category_rows],
            "top": [
                {
                    "identity": row["identity"],
                    "score": row["dominant_score"],
                    "score_gap": row["score_gap"],
                }
                for row in category_rows[:top_n]
            ],
        }

    summary_counts = {
        "total_probes": total_probes,
        "known_probes": total_known_probes,
        "unknown_probes": total_unknown_probes,
        "identity_count": len(identity_columns),
        "threshold": float(threshold) if threshold is not None else None,
    }

    return {
        "available": True,
        "reason": None,
        "threshold": float(threshold) if threshold is not None else None,
        "summary_counts": summary_counts,
        "per_identity": identity_rows,
        "categories": categories,
    }


def compute_all_gallery_vs_all_probe_analysis(
    probe_records,
    probe_histograms,
    gallery_histograms,
    gallery_labels,
    label_map,
    threshold=None,
    top_k_values=(1, 3, 5),
    max_total_comparisons=1000000,
    evaluation_scope="probe",
):
    """
    Build an all gallery vs all evaluation-sample distance analysis.

    The analysis uses the LBPH chi-square alternative distance so the
    pairwise matrix stays consistent with the recognizer confidence.
    Distances are aggregated per gallery identity using the minimum
    template distance, which matches the operational "best template wins"
    behavior of LBPH.

    The returned structure is evaluation-only and does not alter the
    runtime prediction path.
    """

    normalized_probe_records = list(probe_records)
    normalized_probe_histograms = [
        np.asarray(histogram, dtype=np.float64).reshape(-1)
        for histogram in probe_histograms
    ]
    normalized_gallery_histograms = [
        np.asarray(histogram, dtype=np.float64).reshape(-1)
        for histogram in gallery_histograms
    ]
    normalized_gallery_labels = _as_int_array(gallery_labels)

    if len(normalized_probe_records) != len(normalized_probe_histograms):
        raise ValueError(
            "Probe records and probe histograms must have the same length."
        )

    if not normalized_gallery_histograms or normalized_gallery_labels.size == 0:
        return {
            "available": False,
            "reason": "no_gallery_templates",
            "protocol": f"all gallery vs all {evaluation_scope}",
            "threshold": threshold,
            "rows": [],
            "identity_columns": [],
        }

    if len(normalized_gallery_histograms) != normalized_gallery_labels.size:
        raise ValueError(
            "Gallery histograms and gallery labels must have the same length."
        )

    comparison_count = len(normalized_probe_records) * len(normalized_gallery_histograms)
    if comparison_count > max_total_comparisons:
        return {
            "available": False,
            "reason": "matrix_too_large",
            "protocol": f"all gallery vs all {evaluation_scope}",
            "threshold": threshold,
            "probe_count": len(normalized_probe_records),
            "gallery_template_count": len(normalized_gallery_histograms),
            "comparison_count": comparison_count,
            "max_total_comparisons": max_total_comparisons,
            "rows": [],
            "identity_columns": [],
        }

    gallery_matrix = np.vstack(normalized_gallery_histograms).astype(np.float64)
    gallery_identity_names = [
        label_map[int(label)] for label in normalized_gallery_labels.tolist()
    ]

    identity_to_template_indices = {}
    for template_index, identity_name in enumerate(gallery_identity_names):
        identity_to_template_indices.setdefault(identity_name, []).append(template_index)

    identity_columns = sorted(identity_to_template_indices)
    top_k_values = sorted({int(value) for value in top_k_values if int(value) > 0})
    if not top_k_values:
        top_k_values = [1]

    rows = []
    known_probe_count = 0
    unknown_probe_count = 0
    known_rank_hits = {k: 0 for k in top_k_values}
    known_rank_values = []
    best_distances = []
    margins = []

    open_set_counts = {
        "correct_identification": 0,
        "false_reject": 0,
        "misidentification": 0,
        "false_accept": 0,
        "true_reject": 0,
    }

    for probe_index, (probe_record, probe_histogram) in enumerate(
        zip(normalized_probe_records, normalized_probe_histograms),
        start=1,
    ):
        true_identity = probe_record.get("true_identity")
        split_name = probe_record.get("split", "probe")
        is_known = bool(probe_record.get("is_known", "unknown" not in split_name))

        probe_distances = _chi_square_alt_distance_matrix(gallery_matrix, probe_histogram)

        identity_distance_map = {}
        for identity_name in identity_columns:
            template_indices = identity_to_template_indices[identity_name]
            identity_distance_map[identity_name] = float(
                np.min(probe_distances[template_indices])
            )

        ordered_identity_distances = sorted(
            identity_distance_map.items(),
            key=lambda item: (item[1], item[0]),
        )

        best_identity, best_distance = ordered_identity_distances[0]
        second_best_identity = None
        second_best_distance = None
        if len(ordered_identity_distances) > 1:
            second_best_identity, second_best_distance = ordered_identity_distances[1]

        margin = None
        if second_best_distance is not None:
            margin = float(second_best_distance - best_distance)

        true_identity_rank = None
        true_identity_distance = None
        if is_known and true_identity in identity_distance_map:
            true_identity_distance = float(identity_distance_map[true_identity])
            for rank_index, (identity_name, _) in enumerate(
                ordered_identity_distances,
                start=1,
            ):
                if identity_name == true_identity:
                    true_identity_rank = rank_index
                    break
            if true_identity_rank is not None:
                known_rank_values.append(true_identity_rank)
                for k in top_k_values:
                    if true_identity_rank <= k:
                        known_rank_hits[k] += 1

        if is_known:
            known_probe_count += 1
        else:
            unknown_probe_count += 1

        accepted = threshold is not None and best_distance < threshold
        final_identity = best_identity if accepted else "Unknown"

        if threshold is not None:
            if is_known:
                if accepted and final_identity == true_identity:
                    open_set_counts["correct_identification"] += 1
                elif accepted:
                    open_set_counts["misidentification"] += 1
                else:
                    open_set_counts["false_reject"] += 1
            else:
                if accepted:
                    open_set_counts["false_accept"] += 1
                else:
                    open_set_counts["true_reject"] += 1

        row = {
            "probe_index": probe_index,
            "split": split_name,
            "true_identity": true_identity,
            "is_known": is_known,
            "best_identity": best_identity,
            "best_distance": best_distance,
            "second_best_identity": second_best_identity,
            "second_best_distance": second_best_distance,
            "margin": margin,
            "true_identity_rank": true_identity_rank,
            "true_identity_distance": true_identity_distance,
        }

        if threshold is not None:
            row["accepted_at_threshold"] = accepted
            row["final_identity"] = final_identity
            row["decision_correct"] = (
                (is_known and accepted and final_identity == true_identity)
                or (not is_known and not accepted)
            )

        for identity_name in identity_columns:
            row[f"distance__{identity_name}"] = identity_distance_map[identity_name]

        rows.append(row)
        best_distances.append(best_distance)
        if margin is not None:
            margins.append(margin)

    known_rank_summary = {}
    for k in top_k_values:
        known_rank_summary[f"rank_{k}_accuracy"] = _safe_div(
            known_rank_hits[k],
            known_probe_count,
        )

    if known_rank_values:
        known_rank_array = np.asarray(known_rank_values, dtype=np.float64)
        mean_true_rank = float(np.mean(known_rank_array))
        median_true_rank = float(np.median(known_rank_array))
    else:
        mean_true_rank = None
        median_true_rank = None

    best_distance_array = np.asarray(best_distances, dtype=np.float64)
    if best_distance_array.size > 0:
        best_distance_stats = {
            "count": int(best_distance_array.size),
            "mean": float(np.mean(best_distance_array)),
            "std": float(np.std(best_distance_array)) if best_distance_array.size > 1 else 0.0,
            "min": float(np.min(best_distance_array)),
            "max": float(np.max(best_distance_array)),
        }
    else:
        best_distance_stats = None

    if margins:
        margin_array = np.asarray(margins, dtype=np.float64)
        margin_stats = {
            "count": int(margin_array.size),
            "mean": float(np.mean(margin_array)),
            "std": float(np.std(margin_array)) if margin_array.size > 1 else 0.0,
            "min": float(np.min(margin_array)),
            "max": float(np.max(margin_array)),
        }
    else:
        margin_stats = None

    if threshold is not None:
        total_known = known_probe_count
        total_unknown = unknown_probe_count
        correct_identification = open_set_counts["correct_identification"]
        false_reject = open_set_counts["false_reject"]
        misidentification = open_set_counts["misidentification"]
        false_accept = open_set_counts["false_accept"]
        true_reject = open_set_counts["true_reject"]

        open_set_summary = {
            "threshold": float(threshold),
            "total_known": total_known,
            "total_unknown": total_unknown,
            "correct_identification": correct_identification,
            "false_reject": false_reject,
            "misidentification": misidentification,
            "false_accept": false_accept,
            "true_reject": true_reject,
            "DIR": _safe_div(correct_identification, total_known),
            "FAR": _safe_div(false_accept, total_unknown),
            "FRR": _safe_div(false_reject, total_known),
            "GAR": 1.0 - _safe_div(false_reject, total_known),
            "misidentification_rate": _safe_div(misidentification, total_known),
        }
    else:
        open_set_summary = None

    cms_curve = build_cms_curve(known_rank_values, max_rank=min(10, len(identity_columns)))
    preview_rows = _select_pairwise_preview_rows(rows, limit=6)

    return {
        "available": True,
        "reason": None,
        "protocol": f"all gallery vs all {evaluation_scope}",
        "threshold": threshold,
        "probe_count": len(normalized_probe_records),
        "known_probe_count": known_probe_count,
        "unknown_probe_count": unknown_probe_count,
        "gallery_template_count": len(normalized_gallery_histograms),
        "gallery_identity_count": len(identity_columns),
        "comparison_count": comparison_count,
        "identity_columns": identity_columns,
        "rank_summary": known_rank_summary,
        "mean_true_rank": mean_true_rank,
        "median_true_rank": median_true_rank,
        "best_distance_stats": best_distance_stats,
        "margin_stats": margin_stats,
        "open_set_summary": open_set_summary,
        "cms_curve": cms_curve,
        "preview_rows": preview_rows,
        "rows": rows,
    }


def export_recognition_metrics_report(
    report,
    output_stem,
    output_root=None,
    evaluation_scope="probe",
):
    """
    Export report-ready recognition metrics as JSON and CSV artifacts.

    The evaluator uses this after computing the official recognition
    metrics so the report data is stored on disk in a stable location.
    When the report includes pairwise analysis, the pairwise CSV is
    exported alongside the standard summary artifacts.
    """

    if output_root is None:
        output_root = _default_recognition_metrics_root()
    else:
        output_root = Path(output_root)

    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / f"{output_stem}_metrics_report.json"
    summary_csv_path = output_root / f"{output_stem}_threshold_summary.csv"
    roc_csv_path = output_root / f"{output_stem}_roc_points.csv"
    det_csv_path = output_root / f"{output_stem}_det_points.csv"
    pairwise_tag = f"all_gallery_vs_all_{evaluation_scope}"
    pairwise_csv_path = output_root / f"{output_stem}_{pairwise_tag}_matrix.csv"
    doddington_csv_path = output_root / f"{output_stem}_doddington_zoo.csv"
    srr_csv_path = output_root / f"{output_stem}_system_response_reliability.csv"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(_json_ready_report(report), file, indent=2)

    summary_rows = []
    threshold_summary = report.get("threshold_summary", {})
    selected_thresholds = threshold_summary.get("selected_thresholds", {})
    for label in ("low", "mid", "high"):
        row = threshold_summary.get(label)
        if row is None:
            continue
        summary_rows.append(
            {
                "label": label,
                "threshold": row["threshold"],
                "selected_threshold": selected_thresholds.get(label),
                "official_threshold": report.get("selected_threshold"),
                "threshold_source": report.get("threshold_source"),
                "evaluation_scope": report.get("evaluation_scope"),
                "dir": row["DIR"],
                "far": row["FAR"],
                "frr": row["FRR"],
                "gar": row["GAR"],
                "misidentification_rate": row["misidentification_rate"],
            }
        )

    _write_csv(
        summary_csv_path,
        summary_rows,
        [
            "label",
            "threshold",
            "selected_threshold",
            "official_threshold",
            "threshold_source",
            "evaluation_scope",
            "dir",
            "far",
            "frr",
            "gar",
            "misidentification_rate",
        ],
    )

    _write_csv(
        roc_csv_path,
        report.get("sweep_results", {}).get("roc_points", []),
        ["threshold", "far", "gar"],
    )
    _write_csv(
        det_csv_path,
        report.get("sweep_results", {}).get("det_points", []),
        ["threshold", "far", "frr"],
    )

    pairwise_analysis = report.get("pairwise_analysis", {})
    pairwise_rows = pairwise_analysis.get("rows", []) if pairwise_analysis else []
    pairwise_csv_exported = False
    cms_csv_exported = False
    preview_csv_exported = False
    doddington_csv_exported = False
    srr_csv_exported = False
    if pairwise_analysis.get("available") and pairwise_rows:
        identity_columns = pairwise_analysis.get("identity_columns", [])
        pairwise_fieldnames = [
            "probe_index",
            "split",
            "true_identity",
            "is_known",
            "best_identity",
            "best_distance",
            "second_best_identity",
            "second_best_distance",
            "margin",
            "true_identity_rank",
            "true_identity_distance",
            "accepted_at_threshold",
            "final_identity",
            "decision_correct",
        ]
        pairwise_fieldnames.extend(f"distance__{identity}" for identity in identity_columns)
        _write_csv(pairwise_csv_path, pairwise_rows, pairwise_fieldnames)
        pairwise_csv_exported = True

    cms_curve = pairwise_analysis.get("cms_curve", {}) if pairwise_analysis else {}
    cms_points = cms_curve.get("points", []) if cms_curve else []
    cms_csv_path = output_root / f"{output_stem}_{pairwise_tag}_cms_points.csv"
    if cms_curve.get("available") and cms_points:
        _write_csv(
            cms_csv_path,
            cms_points,
            ["rank", "cms", "hit_count", "total"],
        )
        cms_csv_exported = True

    preview_rows = pairwise_analysis.get("preview_rows", []) if pairwise_analysis else []
    preview_csv_path = output_root / f"{output_stem}_{pairwise_tag}_preview.csv"
    if pairwise_analysis.get("available") and preview_rows:
        _write_csv(
            preview_csv_path,
            preview_rows,
            [
                "example_type",
                "probe_index",
                "split",
                "true_identity",
                "is_known",
                "best_identity",
                "best_distance",
                "second_best_identity",
                "second_best_distance",
                "margin",
                "true_identity_rank",
                "true_identity_distance",
                "accepted_at_threshold",
                "final_identity",
                "decision_correct",
            ],
        )
        preview_csv_exported = True

    doddington_analysis = report.get("doddington_analysis") or {}
    doddington_rows = (
        doddington_analysis.get("per_identity", []) if doddington_analysis else []
    )
    if doddington_analysis.get("available") and doddington_rows:
        _write_csv(
            doddington_csv_path,
            [
                {
                    "identity": row["identity"],
                    "dominant_tag": row.get("dominant_tag"),
                    "dominant_score": row.get("dominant_score"),
                    "second_tag": row.get("second_tag"),
                    "second_score": row.get("second_score"),
                    "score_gap": row.get("score_gap"),
                    "sheep_score": row.get("scores", {}).get("sheep"),
                    "goat_score": row.get("scores", {}).get("goat"),
                    "lamb_score": row.get("scores", {}).get("lamb"),
                    "wolf_score": row.get("scores", {}).get("wolf"),
                    "rank_1_accuracy": row.get("rank_1_accuracy"),
                    "rank_3_accuracy": row.get("rank_3_accuracy"),
                    "rank_5_accuracy": row.get("rank_5_accuracy"),
                    "genuine_accept_rate": row.get("genuine_accept_rate"),
                    "genuine_reject_rate": row.get("genuine_reject_rate"),
                    "misidentification_rate": row.get("misidentification_rate"),
                    "impostor_accept_rate": row.get("impostor_accept_rate"),
                    "unknown_accept_rate": row.get("unknown_accept_rate"),
                    "lure_rate": row.get("lure_rate"),
                    "genuine_accepts": row.get("genuine_accepts"),
                    "false_rejects": row.get("false_rejects"),
                    "misidentifications_out": row.get("misidentifications_out"),
                    "impostor_accepts": row.get("impostor_accepts"),
                    "unknown_accepts": row.get("unknown_accepts"),
                    "wrong_known_accepts": row.get("wrong_known_accepts"),
                    "best_match_from_others": row.get("best_match_from_others"),
                }
                for row in doddington_rows
            ],
            [
                "identity",
                "dominant_tag",
                "dominant_score",
                "second_tag",
                "second_score",
                "score_gap",
                "sheep_score",
                "goat_score",
                "lamb_score",
                "wolf_score",
                "rank_1_accuracy",
                "rank_3_accuracy",
                "rank_5_accuracy",
                "genuine_accept_rate",
                "genuine_reject_rate",
                "misidentification_rate",
                "impostor_accept_rate",
                "unknown_accept_rate",
                "lure_rate",
                "genuine_accepts",
                "false_rejects",
                "misidentifications_out",
                "impostor_accepts",
                "unknown_accepts",
                "wrong_known_accepts",
                "best_match_from_others",
            ],
        )
        doddington_csv_exported = True

    srr_analysis = report.get("srr_analysis") or {}
    srr_rows = srr_analysis.get("per_probe", []) if srr_analysis else []
    if srr_analysis.get("available") and srr_rows:
        _write_csv(
            srr_csv_path,
            srr_rows,
            [
                "probe_index",
                "split",
                "true_identity",
                "is_known",
                "best_identity",
                "best_distance",
                "second_best_identity",
                "second_best_distance",
                "margin",
                "true_identity_rank",
                "decision_correct",
                "relative_distance",
                "density_ratio",
                "adaptive_density_ratio",
                "selected_variant",
                "selected_score",
                "selected_threshold",
                "selected_reliable",
                "reliability_label",
                "reliability_gap",
            ],
        )
        srr_csv_exported = True

    export_paths = {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
        "roc_csv": str(roc_csv_path),
        "det_csv": str(det_csv_path),
    }
    if pairwise_csv_exported:
        export_paths[f"{pairwise_tag}_csv"] = str(pairwise_csv_path)
    if cms_csv_exported:
        export_paths[f"{pairwise_tag}_cms_csv"] = str(cms_csv_path)
    if preview_csv_exported:
        export_paths[f"{pairwise_tag}_preview_csv"] = str(preview_csv_path)
    if doddington_csv_exported:
        export_paths["doddington_zoo_csv"] = str(doddington_csv_path)
    if srr_csv_exported:
        export_paths["system_response_reliability_csv"] = str(srr_csv_path)

    return export_paths
