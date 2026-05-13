"""
Module: pad_metrics

Description:
Pure PAD evaluation helpers used by the project evaluators.

This module intentionally contains metric computation and export logic
only:
- single-threshold PAD metrics
- threshold sweep
- EER estimation
- low/mid/high threshold summaries
- export of report-ready JSON / CSV artifacts

PAD metrics here are frame-level metrics: each sample is classified by
comparing its live score against one threshold. They do not simulate the
webcam runtime state machine, which applies temporal smoothing and
hysteresis over multiple usable frames before exposing a final PAD state.

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
    return np.asarray(values, dtype=np.int32).reshape(-1)


def _build_threshold_grid(y_score):
    """
    Build a deterministic threshold grid from the observed score range.
    """

    scores = _as_float_array(y_score)
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


def evaluate_pad_at_threshold(y_true, y_score, threshold):
    """
    Evaluate PAD metrics at one threshold.

    The project convention is:
    - score >= threshold -> live
    - score < threshold -> spoof

    This is the offline/frame-level PAD decision used for APCER/BPCER,
    FAR/FRR, and threshold sweeps. Runtime webcam decisions add a
    temporal layer on top of classifier scores, so these metrics describe
    the base PAD classifier rather than the final smoothed webcam state.
    """

    y_true = _as_int_array(y_true)
    y_score = _as_float_array(y_score)

    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same length.")

    y_pred = (y_score >= threshold).astype(np.int32)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = int(y_true.size)
    live_total = int(np.sum(y_true == 1))
    spoof_total = int(np.sum(y_true == 0))

    far = _safe_div(fp, fp + tn)
    frr = _safe_div(fn, fn + tp)
    gar = 1.0 - frr
    apcer = far
    bpcer = frr
    acer = (apcer + bpcer) / 2.0
    return {
        "threshold": float(threshold),
        "total": total,
        "live_total": live_total,
        "spoof_total": spoof_total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "far": far,
        "frr": frr,
        "gar": gar,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def sweep_pad_thresholds(y_true, y_score, thresholds=None):
    """
    Evaluate PAD metrics over a threshold sweep and collect ROC / DET data.
    """

    y_true = _as_int_array(y_true)
    y_score = _as_float_array(y_score)

    if y_true.size != y_score.size:
        raise ValueError("y_true and y_score must have the same length.")

    if thresholds is None:
        thresholds = _build_threshold_grid(y_score)
    else:
        thresholds = np.unique(_as_float_array(thresholds))

    per_threshold = [
        evaluate_pad_at_threshold(y_true, y_score, threshold)
        for threshold in thresholds
    ]

    roc_points = [
        {
            "threshold": item["threshold"],
            "far": item["far"],
            "gar": item["gar"],
        }
        for item in per_threshold
    ]
    det_points = [
        {
            "threshold": item["threshold"],
            "far": item["far"],
            "frr": item["frr"],
        }
        for item in per_threshold
    ]

    far_values = np.array([item["far"] for item in per_threshold], dtype=np.float64)
    frr_values = np.array([item["frr"] for item in per_threshold], dtype=np.float64)
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


def compute_pad_eer(sweep_results):
    """
    Return the EER information already derived from the sweep.
    """

    if not sweep_results:
        return {"eer_value": None, "eer_threshold": None}

    return {
        "eer_value": sweep_results.get("eer_value"),
        "eer_threshold": sweep_results.get("eer_threshold"),
    }


def build_pad_threshold_summary(sweep_results, low_t=None, mid_t=None, high_t=None):
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


def _default_pad_metrics_root():
    return Path(__file__).resolve().parents[2] / "logs" / "metrics" / "pad"


def _pad_official_root(output_root):
    return Path(output_root) / "official"


def _pad_curves_root(output_root):
    return Path(output_root) / "curves"


def _json_ready_report(report):
    """
    Convert the in-memory PAD report into a JSON-serializable object.
    """

    return {
        "selected_threshold": report.get("selected_threshold"),
        "selected_metrics": report.get("selected_metrics"),
        "eer_results": report.get("eer_results"),
        "threshold_summary": report.get("threshold_summary"),
        "export_paths": report.get("export_paths"),
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


def export_pad_metrics_report(report, output_stem, output_root=None, export_curves=False):
    """
    Export report-ready PAD metrics as JSON and CSV artifacts.

    The evaluator uses this after computing the official PAD metrics so
    the report data is stored on disk in a stable location.
    """

    if output_root is None:
        output_root = _default_pad_metrics_root()
    else:
        output_root = Path(output_root)

    official_root = _pad_official_root(output_root)
    curves_root = _pad_curves_root(output_root)
    official_root.mkdir(parents=True, exist_ok=True)

    json_path = official_root / f"{output_stem}_metrics_report.json"
    summary_csv_path = official_root / f"{output_stem}_threshold_summary.csv"
    roc_csv_path = curves_root / f"{output_stem}_roc_points.csv"
    det_csv_path = curves_root / f"{output_stem}_det_points.csv"

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
                "far": row["far"],
                "frr": row["frr"],
                "gar": row["gar"],
                "apcer": row["apcer"],
                "bpcer": row["bpcer"],
                "acer": row["acer"],
            }
        )

    _write_csv(
        summary_csv_path,
        summary_rows,
        [
            "label",
            "threshold",
            "selected_threshold",
            "far",
            "frr",
            "gar",
            "apcer",
            "bpcer",
            "acer",
        ],
    )

    export_paths = {
        "json": str(json_path),
        "summary_csv": str(summary_csv_path),
    }
    if export_curves:
        curves_root.mkdir(parents=True, exist_ok=True)
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
        export_paths["roc_csv"] = str(roc_csv_path)
        export_paths["det_csv"] = str(det_csv_path)

    report["export_paths"] = export_paths
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(_json_ready_report(report), file, indent=2)

    return export_paths
