"""
Module: recognition_model_evaluator

Description:
This module evaluates the trained LBPH face recognition model on one
selected probe dataset or on all supported probe datasets
sequentially.

Evaluation uses the single recognition model currently stored on disk,
which is overwritten every time a new training run is executed. For
selected evaluation pairs, it also stores an offline threshold
reference derived from the current score distributions.

Besides the global accuracy summary, the evaluator also reports:
- score statistics for genuine and impostor predictions
- the worst-performing identities in the current probe split
- a small threshold analysis that approximates runtime accept/reject
  behavior for a few candidate thresholds
- an optional all gallery vs all probe distance analysis for reports

Offline threshold updates are intentionally stricter than plain
evaluation: a threshold is persisted only when the selected run is an
approved reference pair and contains both genuine and impostor scores,
so fallback thresholds from overly easy runs are never promoted to the
stored evaluation reference.
"""

import cv2
import json
import pickle
from pathlib import Path
import statistics
import time

import numpy as np

from src.preprocessing.face_preprocessing import get_recognition_preprocessing_config
from src.preprocessing.pose_statistics import (
    format_pose_summary,
    merge_pose_summaries,
)
from src.recognition.face_recognizer import DEFAULT_PREDICTION_THRESHOLD
from src.recognition.recognition_dataset_loader import (
    load_probe_dataset,
    normalize_identity_name,
    should_namespace_identities,
)
from src.recognition.recognition_metrics import (
    compute_all_gallery_vs_all_probe_analysis,
    build_recognition_threshold_summary,
    build_doddington_zoo_analysis,
    build_system_response_reliability_analysis,
    compute_recognition_eer,
    export_recognition_metrics_report,
    evaluate_recognition_at_threshold,
    sweep_recognition_thresholds,
)


# ----- CONSTANTS ----- #
RECOGNITION_MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models" / "recognition"
MODEL_PATH = RECOGNITION_MODELS_ROOT / "lbph_model.xml"
LABEL_MAP_PATH = RECOGNITION_MODELS_ROOT / "label_map.pkl"
RECOGNITION_CONFIG_PATH = RECOGNITION_MODELS_ROOT / "recognition_config.json"
EVAL_MODE = "custom"
PROBE_PROGRESS_STEP = 100
DEBUG_LABEL_MAP_PREVIEW_LIMIT = 10
DEBUG_SAMPLE_PREVIEW_LIMIT = 10
PAIRWISE_MAX_TOTAL_COMPARISONS = 1500000
PAIRWISE_TOP_K_VALUES = (1, 3, 5)


def _build_lfw_sample_suffix(lfw_sample_ratio):
    """
    Build a filename suffix that reflects the LFW sampling level.
    """
    if lfw_sample_ratio is None or lfw_sample_ratio >= 1.0:
        return ""

    ratio_tag = f"{lfw_sample_ratio:.3f}".rstrip("0").rstrip(".")
    ratio_tag = ratio_tag.replace(".", "p")
    return f"_lfw_r{ratio_tag}"


def load_trained_recognition_model():
    """
    Load the trained recognition model and label map from disk.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Recognition model not found: {MODEL_PATH}")

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Recognition label map not found: {LABEL_MAP_PATH}")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_PATH))

    with open(LABEL_MAP_PATH, "rb") as file:
        label_map = pickle.load(file)

    return recognizer, label_map


def save_recognition_config(config_data):
    """
    Save the recognition evaluation configuration to disk as JSON.
    """
    RECOGNITION_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(RECOGNITION_CONFIG_PATH, "w", encoding="utf-8") as file:
        json.dump(config_data, file, indent=4)


def load_evaluation_threshold_config():
    """
    Load the currently saved offline threshold configuration, if any.

    This is used only for reporting purposes inside the evaluator so the
    current run can be compared against the last persisted offline
    threshold reference.
    """
    if not RECOGNITION_CONFIG_PATH.exists():
        return None

    with open(RECOGNITION_CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def should_save_evaluation_threshold(
    train_mode,
    eval_mode,
    allow_config_update=False,
):
    """
    Decide whether the current evaluation should update the offline threshold.

    The stored offline threshold is updated only for the approved
    reference pairs used by the project. The run must also contain both
    genuine and impostor scores and must not rely on the fallback
    threshold rule.
    """
    valid_pairs = {
        ("custom", "custom"),
        ("orl", "orl"),
        ("custom_orl", "orl"),
    }
    return allow_config_update and (train_mode, eval_mode) in valid_pairs


def _extract_lbph_params(recognizer):
    """
    Read the LBPH hyperparameters from a trained recognizer.
    """
    return {
        "radius": recognizer.getRadius(),
        "neighbors": recognizer.getNeighbors(),
        "grid_x": recognizer.getGridX(),
        "grid_y": recognizer.getGridY(),
    }


def _extract_lbph_histogram(face_image, lbph_params):
    """
    Extract one LBPH histogram using the same parameters as the trained model.

    This helper is used only for the offline pairwise report and leaves
    the runtime recognition path untouched.
    """
    temp_recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=lbph_params["radius"],
        neighbors=lbph_params["neighbors"],
        grid_x=lbph_params["grid_x"],
        grid_y=lbph_params["grid_y"],
    )
    temp_recognizer.train([face_image], np.array([0], dtype=np.int32))
    return np.asarray(temp_recognizer.getHistograms()[0], dtype=np.float64).reshape(-1)


def build_all_gallery_vs_all_probe_analysis(
    probe_records,
    recognizer,
    label_map,
    threshold=DEFAULT_PREDICTION_THRESHOLD,
):
    """
    Build the all gallery vs all probe analysis for the current recognizer.

    The helper re-extracts probe histograms with the trained LBPH
    configuration so the pairwise report stays comparable to the stored
    gallery templates.
    """
    gallery_histograms = [
        np.asarray(histogram, dtype=np.float64).reshape(-1)
        for histogram in recognizer.getHistograms()
    ]
    gallery_labels = np.asarray(recognizer.getLabels()).reshape(-1).astype(int)

    comparison_count = len(probe_records) * len(gallery_histograms)
    if comparison_count > PAIRWISE_MAX_TOTAL_COMPARISONS:
        return {
            "available": False,
            "reason": "matrix_too_large",
            "protocol": "all gallery vs all probe",
            "threshold": threshold,
            "probe_count": len(probe_records),
            "gallery_template_count": len(gallery_histograms),
            "comparison_count": comparison_count,
            "max_total_comparisons": PAIRWISE_MAX_TOTAL_COMPARISONS,
            "rows": [],
            "identity_columns": [],
        }

    lbph_params = _extract_lbph_params(recognizer)
    pairwise_probe_records = []
    pairwise_probe_histograms = []

    for record in probe_records:
        true_identity = normalize_identity_name(record["identity"])
        split_name = record["split"]
        pairwise_probe_records.append(
            {
                "true_identity": true_identity,
                "split": split_name,
                "is_known": split_name != "probe_unknown",
            }
        )
        pairwise_probe_histograms.append(
            _extract_lbph_histogram(record["image"], lbph_params)
        )

    return compute_all_gallery_vs_all_probe_analysis(
        pairwise_probe_records,
        pairwise_probe_histograms,
        gallery_histograms,
        gallery_labels,
        label_map,
        threshold=threshold,
        top_k_values=PAIRWISE_TOP_K_VALUES,
        max_total_comparisons=PAIRWISE_MAX_TOTAL_COMPARISONS,
    )


def compute_score_statistics(scores):
    """
    Compute compact descriptive statistics for a list of confidence scores.
    """
    if not scores:
        return None

    return {
        "count": len(scores),
        "mean": statistics.fmean(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "min": min(scores),
        "max": max(scores),
    }


def _format_optional_float(value):
    """
    Format a possibly missing floating-point value for diagnostics.
    """
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _format_optional_int(value):
    """
    Format a possibly missing integer value for diagnostics.
    """
    if value is None:
        return "n/a"
    return str(int(value))


def _lookup_cms_point(cms_points, rank):
    """
    Return the CMS value for one requested rank, if available.
    """
    for point in cms_points or []:
        if int(point.get("rank", 0)) == int(rank):
            return point.get("cms")
    return None


def _format_optional_list(values):
    """
    Format a possibly empty list for compact diagnostics.
    """
    if not values:
        return "n/a"
    return ", ".join(str(value) for value in values)


def compute_suggested_threshold(genuine_scores, impostor_scores):
    """
    Compute an offline threshold suggestion from the evaluation score
    distributions.

    When both genuine and impostor scores are available, use the midpoint
    between the worst genuine score and the best impostor score. This is
    less sensitive to single-score outliers than the previous fixed-margin
    rule. If impostor scores are not available, fall back to a conservative
    genuine-only estimate.
    """
    if not genuine_scores:
        return None, "unavailable"

    max_genuine = max(genuine_scores)

    if impostor_scores:
        min_impostor = min(impostor_scores)
        return (max_genuine + min_impostor) / 2, "midpoint_max_genuine_min_impostor"

    return max_genuine + 5, "max_genuine_plus_5_fallback"


def compute_open_set_threshold(known_scores, unknown_scores):
    """
    Compute a threshold suggestion for an open-set evaluation.

    The goal is to separate accepted known identities from rejected
    unknown identities using the LBPH confidence score, where lower is
    better.
    """
    if not known_scores:
        return None, "unavailable"

    max_known = max(known_scores)

    if unknown_scores:
        min_unknown = min(unknown_scores)
        return (max_known + min_unknown) / 2, "midpoint_max_known_min_unknown"

    return max_known + 5, "max_known_plus_5_fallback"


def compute_identity_accuracy(records):
    """
    Aggregate probe outcomes by true identity.

    The returned dictionary is keyed by the ground-truth identity so the
    report highlights which enrolled identities are hardest to recognize.
    """
    identity_stats = {}

    for record in records:
        identity_name = record["true_identity"]
        current_stats = identity_stats.setdefault(
            identity_name,
            {"samples": 0, "correct": 0},
        )
        current_stats["samples"] += 1
        current_stats["correct"] += int(record["is_correct"])

    for current_stats in identity_stats.values():
        current_stats["accuracy"] = current_stats["correct"] / current_stats["samples"]

    return identity_stats


def evaluate_threshold_candidates(records, candidates):
    """
    Evaluate a few threshold candidates on the collected records.

    For LBPH, lower confidence is better:
    - confidence < threshold  -> accepted as the predicted identity
    - confidence >= threshold -> rejected as Unknown
    """
    threshold_results = []

    for candidate in candidates:
        threshold = candidate["threshold"]

        if threshold is None:
            continue

        accepted_genuine = 0
        rejected_genuine = 0
        accepted_impostor = 0
        rejected_impostor = 0

        for record in records:
            is_accepted = record["confidence"] < threshold

            if record["is_correct"]:
                if is_accepted:
                    accepted_genuine += 1
                else:
                    rejected_genuine += 1
            else:
                if is_accepted:
                    accepted_impostor += 1
                else:
                    rejected_impostor += 1

        threshold_results.append(
            {
                "label": candidate["label"],
                "threshold": threshold,
                "accepted_genuine": accepted_genuine,
                "rejected_genuine": rejected_genuine,
                "accepted_impostor": accepted_impostor,
                "rejected_impostor": rejected_impostor,
            }
        )

    return threshold_results


def evaluate_open_set_threshold_candidates(records, candidates):
    """
    Evaluate a few thresholds on an open-set probe collection.

    The records are expected to include:
    - ``split``: ``probe_known`` or ``probe_unknown`` for the custom dataset
      (other probe splits are treated as known)
    - ``identity``: ground-truth identity
    - ``predicted_identity``: best-match identity produced by LBPH
    - ``confidence``: LBPH confidence score
    """
    threshold_results = []

    for candidate in candidates:
        threshold = candidate["threshold"]

        if threshold is None:
            continue

        known_accepted_correctly = 0
        known_rejected_as_unknown = 0
        known_accepted_wrongly_as_other_known = 0
        unknown_rejected_correctly = 0
        unknown_accepted_wrongly_as_known = 0

        for record in records:
            is_accepted = record["confidence"] < threshold
            final_identity = (
                record["predicted_identity"] if is_accepted else "Unknown"
            )
            is_unknown_split = record["split"] == "probe_unknown"

            if is_unknown_split:
                if final_identity == "Unknown":
                    unknown_rejected_correctly += 1
                else:
                    unknown_accepted_wrongly_as_known += 1
            else:
                if final_identity == record["identity"]:
                    known_accepted_correctly += 1
                elif final_identity == "Unknown":
                    known_rejected_as_unknown += 1
                else:
                    known_accepted_wrongly_as_other_known += 1

        threshold_results.append(
            {
                "label": candidate["label"],
                "threshold": threshold,
                "known_accepted_correctly": known_accepted_correctly,
                "known_rejected_as_unknown": known_rejected_as_unknown,
                "known_accepted_wrongly_as_other_known": known_accepted_wrongly_as_other_known,
                "unknown_rejected_correctly": unknown_rejected_correctly,
                "unknown_accepted_wrongly_as_known": unknown_accepted_wrongly_as_known,
            }
        )

    return threshold_results


def print_score_statistics(name, stats):
    """
    Print a compact score-statistics block.
    """
    if stats is None:
        print(f"{name} score stats: unavailable")
        return

    print(
        f"{name} score stats: "
        f"count={stats['count']} | "
        f"mean={stats['mean']:.4f} | "
        f"std={stats['std']:.4f} | "
        f"min={stats['min']:.4f} | "
        f"max={stats['max']:.4f}"
    )


def print_identity_breakdown(identity_stats, limit=5):
    """
    Print the worst-performing identities in the current evaluation.
    """
    if not identity_stats:
        print("Worst identities: unavailable")
        return

    print(f"Worst identities (top {min(limit, len(identity_stats))}):")

    sorted_items = sorted(
        identity_stats.items(),
        key=lambda item: (item[1]["accuracy"], -item[1]["samples"], item[0]),
    )

    for identity_name, current_stats in sorted_items[:limit]:
        print(
            f"- {identity_name}: "
            f"{current_stats['correct']}/{current_stats['samples']} correct | "
            f"accuracy={current_stats['accuracy']:.4f}"
        )


def print_threshold_analysis(threshold_results):
    """
    Print a compact accept/reject analysis for candidate thresholds.
    """
    if not threshold_results:
        print("Threshold analysis: unavailable")
        return

    print("Threshold analysis:")

    for result in threshold_results:
        print(
            f"- {result['label']}: "
            f"threshold={result['threshold']:.4f} | "
            f"accepted_genuine={result['accepted_genuine']} | "
            f"rejected_genuine={result['rejected_genuine']} | "
            f"accepted_impostor={result['accepted_impostor']} | "
            f"rejected_impostor={result['rejected_impostor']}"
        )


def print_open_set_threshold_analysis(threshold_results):
    """
    Print a compact open-set accept/reject analysis for candidate thresholds.
    """
    if not threshold_results:
        print("Open-set threshold analysis: unavailable")
        return

    print("Open-set threshold analysis:")

    for result in threshold_results:
        print(
            f"- {result['label']}: "
            f"threshold={result['threshold']:.4f} | "
            f"known_ok={result['known_accepted_correctly']} | "
            f"known_reject={result['known_rejected_as_unknown']} | "
            f"known_wrong_accept={result['known_accepted_wrongly_as_other_known']} | "
            f"unknown_ok={result['unknown_rejected_correctly']} | "
            f"unknown_wrong_accept={result['unknown_accepted_wrongly_as_known']}"
        )


def print_doddington_zoo_analysis(doddington_analysis, limit=3):
    """
    Print a compact Doddington-style summary for recognition.
    """
    if not doddington_analysis or not doddington_analysis.get("available"):
        print("Doddington zoo: unavailable")
        return

    summary_counts = doddington_analysis.get("summary_counts", {})
    categories = doddington_analysis.get("categories", {})
    print(
        "Doddington zoo:",
        (
            f"identities={summary_counts.get('identity_count', 0)} | "
            f"known_probes={summary_counts.get('known_probes', 0)} | "
            f"unknown_probes={summary_counts.get('unknown_probes', 0)}"
        ),
    )

    for category_name in ("sheep", "goat", "lamb", "wolf"):
        category = categories.get(category_name, {})
        identities = category.get("identities", [])
        top = category.get("top", [])
        print(
            f"- {category_name}:",
            (
                f"count={category.get('count', 0)} | "
                f"identities={_format_optional_list(identities[:limit])}"
            ),
        )
        if top:
            top_text = ", ".join(
                f"{row['identity']}({row['score']:.4f}, gap={row.get('score_gap', 0.0):.4f})"
                for row in top[:limit]
            )
            print(f"  top={top_text}")


def print_system_response_reliability_analysis(srr_analysis, limit=3):
    """
    Print a compact offline SRR summary.
    """
    if not srr_analysis or not srr_analysis.get("available"):
        print("System response reliability: unavailable")
        return

    def _sort_value(value, default):
        if value is None:
            return float(default)
        return float(value)

    summary_counts = srr_analysis.get("summary_counts", {})
    selected_variant = srr_analysis.get("selected_variant", "adaptive_density_ratio")
    selected_threshold = srr_analysis.get("selected_threshold")
    variants = srr_analysis.get("variants", {})
    selected_variant_report = variants.get(selected_variant, {})
    selected_stats = selected_variant_report.get("stats", {})

    print(
        "System response reliability:",
        (
            f"variant={selected_variant} | "
            f"phi_k={_format_optional_float(selected_threshold)} | "
            f"reliable={summary_counts.get('reliable_responses', 0)} | "
            f"unreliable={summary_counts.get('unreliable_responses', 0)} | "
            f"mean_phi={_format_optional_float(selected_stats.get('mean'))}"
        ),
    )

    selected_per_probe = srr_analysis.get("per_probe", [])
    if not selected_per_probe:
        return

    top_rows = sorted(
        selected_per_probe,
        key=lambda row: (
            -_sort_value(row.get("selected_score"), -np.inf),
            _sort_value(row.get("reliability_gap"), -np.inf),
            row.get("probe_index"),
        ),
    )
    low_rows = sorted(
        selected_per_probe,
        key=lambda row: (
            _sort_value(row.get("selected_score"), np.inf),
            _sort_value(row.get("reliability_gap"), np.inf),
            row.get("probe_index"),
        ),
    )

    print(f"SRR top {min(limit, len(top_rows))}:")
    for row in top_rows[:limit]:
        print(
            "-",
            f"probe={row.get('probe_index')}",
            f"true={row.get('true_identity')}",
            f"score={_format_optional_float(row.get('selected_score'))}",
            f"gap={_format_optional_float(row.get('reliability_gap'))}",
            f"reliable={row.get('selected_reliable')}",
        )

    print(f"SRR lowest {min(limit, len(low_rows))}:")
    for row in low_rows[:limit]:
        print(
            "-",
            f"probe={row.get('probe_index')}",
            f"true={row.get('true_identity')}",
            f"score={_format_optional_float(row.get('selected_score'))}",
            f"gap={_format_optional_float(row.get('reliability_gap'))}",
            f"reliable={row.get('selected_reliable')}",
        )


def compute_recognition_evaluation_metrics(
    records,
    selected_threshold,
    output_stem=None,
):
    """
    Compute the official recognition metrics used by the project reports.

    The evaluator keeps the loading and logging responsibilities, while
    this function delegates the actual metric computation to
    ``recognition_metrics.py``.
    """

    selected_metrics = evaluate_recognition_at_threshold(records, selected_threshold)
    sweep_results = sweep_recognition_thresholds(records)
    eer_results = compute_recognition_eer(sweep_results)
    threshold_summary = build_recognition_threshold_summary(sweep_results)

    report = {
        "selected_threshold": selected_threshold,
        "selected_metrics": selected_metrics,
        "sweep_results": sweep_results,
        "eer_results": eer_results,
        "threshold_summary": threshold_summary,
    }

    if output_stem is not None:
        report["export_paths"] = export_recognition_metrics_report(report, output_stem)

    return report


def log_recognition_evaluation_diagnostics(
    mode_label,
    load_elapsed,
    metrics_report,
    known_stats=None,
    unknown_stats=None,
    identity_stats=None,
    threshold_results=None,
    saved_evaluation_threshold=None,
    saved_evaluation_threshold_source=None,
    dataset_counts=None,
    total_count=None,
    correct_count=None,
    genuine_count=None,
    impostor_count=None,
    known_count=None,
    unknown_count=None,
    evaluation_elapsed=None,
    open_set=False,
    pairwise_analysis=None,
    pose_summary=None,
):
    """
    Print the project diagnostics for recognition.

    The evaluator keeps the human-readable logs here, while the official
    project metrics are produced by ``recognition_metrics.py``.
    """

    print(f"=== RECOGNITION DIAGNOSTICS: {mode_label} ===")
    selected_metrics = metrics_report["selected_metrics"]
    threshold_summary = metrics_report["threshold_summary"]
    eer_results = metrics_report["eer_results"]

    if open_set:
        print("Probe samples:", total_count if total_count is not None else 0)
        print(f"Probe loading completed in {load_elapsed:.2f}s")
        if known_count is not None:
            print("probe_known samples:", known_count)
        if unknown_count is not None:
            print("probe_unknown samples:", unknown_count)
        if pose_summary is not None:
            print(format_pose_summary(pose_summary, title="Probe pose summary:"))
        print(f"Using runtime threshold: {selected_metrics['threshold']:.2f}")
        print("Known accepted correctly:", selected_metrics["correct_identification"])
        print("Known rejected as unknown:", selected_metrics["false_reject"])
        print(
            "Known accepted wrongly as other known:",
            selected_metrics["misidentification"],
        )
        print("Unknown rejected correctly:", selected_metrics["true_reject"])
        print("Unknown accepted wrongly as known:", selected_metrics["false_accept"])
    else:
        print("Probe samples:", total_count if total_count is not None else 0)
        print(f"Probe loading completed in {load_elapsed:.2f}s")
        if dataset_counts is not None:
            for dataset_name, sample_count in dataset_counts.items():
                print(f"{dataset_name} probe samples: {sample_count}")
        if pose_summary is not None:
            print(format_pose_summary(pose_summary, title="Probe pose summary:"))
        if correct_count is not None and total_count is not None:
            print("Correct:", correct_count)
            print("Accuracy:", correct_count / total_count if total_count else 0.0)
        if genuine_count is not None:
            print("Genuine matches:", genuine_count)
        if impostor_count is not None:
            print("Impostor matches:", impostor_count)

    print(
        "Recognition official metrics:",
        (
            f"threshold={selected_metrics['threshold']:.4f} | "
            f"DIR={selected_metrics['DIR']:.4f} | "
            f"FAR={selected_metrics['FAR']:.4f} | "
            f"FRR={selected_metrics['FRR']:.4f} | "
            f"GAR={selected_metrics['GAR']:.4f} | "
            f"misidentification={selected_metrics['misidentification_rate']:.4f}"
        ),
    )

    if eer_results.get("eer_value") is not None:
        print(
            "Recognition EER:",
            (
                f"eer={eer_results['eer_value']:.4f} | "
                f"eer_threshold={eer_results['eer_threshold']:.4f}"
            ),
        )

    summary_rows = []
    for label in ("low", "mid", "high"):
        row = threshold_summary.get(label)
        if row is None:
            continue
        summary_rows.append(
            (
                f"{label}={row['threshold']:.4f} | "
                f"DIR={row['DIR']:.4f} | "
                f"FAR={row['FAR']:.4f} | "
                f"FRR={row['FRR']:.4f} | "
                f"GAR={row['GAR']:.4f} | "
                f"misidentification={row['misidentification_rate']:.4f}"
            )
        )

    if summary_rows:
        print("Recognition threshold summary:")
        for row in summary_rows:
            print(f"- {row}")

    if known_stats is not None:
        print("Mean known confidence:", known_stats["mean"])
    if unknown_stats is not None:
        print("Mean unknown confidence:", unknown_stats["mean"])
    if identity_stats is not None:
        print_identity_breakdown(identity_stats, limit=5)
    if saved_evaluation_threshold is not None:
        print(
            "Saved evaluation threshold source:",
            saved_evaluation_threshold_source or "unknown",
        )
    if threshold_results is not None:
        if open_set:
            print_open_set_threshold_analysis(threshold_results)
        else:
            print_threshold_analysis(threshold_results)

    if pairwise_analysis is not None:
        if pairwise_analysis.get("available"):
            rank_summary = pairwise_analysis.get("rank_summary", {})
            rank_1 = rank_summary.get("rank_1_accuracy")
            rank_3 = rank_summary.get("rank_3_accuracy")
            rank_5 = rank_summary.get("rank_5_accuracy")
            cms_curve = pairwise_analysis.get("cms_curve", {})
            cms_points = cms_curve.get("points", []) if cms_curve else []
            cms_1 = _lookup_cms_point(cms_points, 1)
            cms_3 = _lookup_cms_point(cms_points, 3)
            cms_5 = _lookup_cms_point(cms_points, 5)
            cms_10 = _lookup_cms_point(cms_points, 10)
            mean_margin = pairwise_analysis.get("margin_stats", {}).get("mean")
            print(
                "All gallery vs all probe:",
                (
                    f"probes={pairwise_analysis.get('probe_count', 0)} | "
                    f"gallery_templates={pairwise_analysis.get('gallery_template_count', 0)} | "
                    f"rank1={_format_optional_float(rank_1)} | "
                    f"rank3={_format_optional_float(rank_3)} | "
                    f"rank5={_format_optional_float(rank_5)} | "
                    f"cms1={_format_optional_float(cms_1)} | "
                    f"cms3={_format_optional_float(cms_3)} | "
                    f"cms5={_format_optional_float(cms_5)} | "
                    f"cms10={_format_optional_float(cms_10)} | "
                    f"mean_margin={_format_optional_float(mean_margin)}"
                ),
            )
            preview_rows = pairwise_analysis.get("preview_rows", [])
            if preview_rows:
                print("Pairwise preview:")
                for preview_row in preview_rows[:3]:
                    print(
                        "-",
                        f"{preview_row.get('example_type')}:",
                        f"probe={preview_row.get('probe_index')}",
                        f"true={preview_row.get('true_identity')}",
                        f"rank={_format_optional_int(preview_row.get('true_identity_rank'))}",
                        f"margin={_format_optional_float(preview_row.get('margin'))}",
                        f"best={preview_row.get('best_identity')}({_format_optional_float(preview_row.get('best_distance'))})",
                    )
        else:
            print(
                "All gallery vs all probe:",
                f"unavailable ({pairwise_analysis.get('reason', 'unknown')})",
            )

    doddington_analysis = metrics_report.get("doddington_analysis")
    if doddington_analysis is not None:
        print_doddington_zoo_analysis(doddington_analysis)

    srr_analysis = metrics_report.get("srr_analysis")
    if srr_analysis is not None:
        print_system_response_reliability_analysis(srr_analysis)

    if evaluation_elapsed is not None:
        print(f"Evaluation completed in {evaluation_elapsed:.2f}s")
    export_paths = metrics_report.get("export_paths")
    if export_paths:
        print("Recognition metric exports:")
        print(f"- json: {export_paths['json']}")
        print(f"- summary_csv: {export_paths['summary_csv']}")
        print(f"- roc_csv: {export_paths['roc_csv']}")
        print(f"- det_csv: {export_paths['det_csv']}")
        if "all_gallery_vs_all_probe_csv" in export_paths:
            print(f"- all_gallery_vs_all_probe_csv: {export_paths['all_gallery_vs_all_probe_csv']}")
        if "all_gallery_vs_all_probe_cms_csv" in export_paths:
            print(f"- all_gallery_vs_all_probe_cms_csv: {export_paths['all_gallery_vs_all_probe_cms_csv']}")
        if "all_gallery_vs_all_probe_preview_csv" in export_paths:
            print(f"- all_gallery_vs_all_probe_preview_csv: {export_paths['all_gallery_vs_all_probe_preview_csv']}")
        if "system_response_reliability_csv" in export_paths:
            print(f"- system_response_reliability_csv: {export_paths['system_response_reliability_csv']}")
    print()


def evaluate_custom_open_set_dataset(
    train_mode,
    recognizer,
    label_map,
    allow_config_update=False,
    output_stem=None,
    lfw_sample_ratio=None,
):
    """
    Evaluate the custom recognition dataset as an open-set problem.

    The custom gallery/probe layout is now split into ``probe_known`` and
    ``probe_unknown`` so the evaluation can report acceptance and rejection
    behavior separately for known and unknown identities.
    """
    print("=== RECOGNITION EVALUATION MODE: CUSTOM OPEN SET ===")
    preprocessing_config = get_recognition_preprocessing_config()
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )
    if lfw_sample_ratio is not None and lfw_sample_ratio < 1.0:
        print(f"LFW sample ratio: {lfw_sample_ratio:.3f}")

    load_start_time = time.perf_counter()
    known_records, known_dataset_counts, known_pose_summary = load_probe_dataset(
        "custom",
        split="probe_known",
        namespace_identities=False,
        collect_pose_metrics=True,
    )
    unknown_records, unknown_dataset_counts, unknown_pose_summary = load_probe_dataset(
        "custom",
        split="probe_unknown",
        namespace_identities=False,
        collect_pose_metrics=True,
    )
    pose_summary = merge_pose_summaries(known_pose_summary, unknown_pose_summary)
    load_elapsed = time.perf_counter() - load_start_time

    probe_records = known_records + unknown_records
    if not probe_records:
        print("No valid probe images found for evaluation.")
        print()
        return

    print("Label map preview:")
    for label_index, identity_name in sorted(label_map.items())[:DEBUG_LABEL_MAP_PREVIEW_LIMIT]:
        print(f"- {label_index} -> {identity_name}")

    known_scores = []
    unknown_scores = []
    normalized_records = []
    evaluation_records = []

    for index, record in enumerate(probe_records, start=1):
        label, confidence = recognizer.predict(record["image"])
        predicted_name = normalize_identity_name(label_map[label])
        true_name = normalize_identity_name(record["identity"])
        is_known_split = record["split"] == "probe_known"

        if is_known_split:
            known_scores.append(confidence)
        else:
            unknown_scores.append(confidence)

        evaluation_records.append(
            {
                "split": record["split"],
                "identity": true_name,
                "predicted_label": label,
                "predicted_identity": predicted_name,
                "confidence": confidence,
                "is_known_split": is_known_split,
            }
        )
        normalized_records.append(
            {
                "is_known": is_known_split,
                "true_id": true_name if is_known_split else None,
                "predicted_id": predicted_name,
                "score": confidence,
            }
        )

        if index <= DEBUG_SAMPLE_PREVIEW_LIMIT:
            print(
                "Sample preview:",
                f"split={record['split']}",
                f"true_raw={record['identity']}",
                f"true_norm={true_name}",
                f"label={label}",
                f"predicted_norm={predicted_name}",
                f"confidence={confidence:.4f}",
            )

        if index % PROBE_PROGRESS_STEP == 0:
            print(f"Evaluated {index}/{len(probe_records)} probe images")

    evaluation_threshold = DEFAULT_PREDICTION_THRESHOLD
    report_stem = output_stem or f"{train_mode}_custom_open_set{_build_lfw_sample_suffix(lfw_sample_ratio)}"

    metrics_report = compute_recognition_evaluation_metrics(
        normalized_records,
        evaluation_threshold,
        output_stem=None,
    )
    pairwise_analysis = build_all_gallery_vs_all_probe_analysis(
        probe_records,
        recognizer,
        label_map,
        threshold=evaluation_threshold,
    )
    metrics_report["pairwise_analysis"] = pairwise_analysis
    metrics_report["doddington_analysis"] = build_doddington_zoo_analysis(
        pairwise_analysis,
        threshold=evaluation_threshold,
    )
    metrics_report["srr_analysis"] = build_system_response_reliability_analysis(
        pairwise_analysis,
        threshold=evaluation_threshold,
    )
    metrics_report["export_paths"] = export_recognition_metrics_report(
        metrics_report,
        report_stem,
    )

    known_stats = compute_score_statistics(known_scores)
    unknown_stats = compute_score_statistics(unknown_scores)

    open_set_threshold, open_set_threshold_rule = compute_open_set_threshold(
        known_scores,
        unknown_scores,
    )

    if open_set_threshold is not None:
        print("Suggested open-set threshold:", open_set_threshold)
        print("Suggested open-set threshold rule:", open_set_threshold_rule)

    current_evaluation_config = load_evaluation_threshold_config()
    saved_evaluation_threshold = None
    saved_evaluation_threshold_source = None
    if current_evaluation_config is not None:
        saved_evaluation_threshold = current_evaluation_config.get("unknown_threshold")
        saved_evaluation_threshold_source = current_evaluation_config.get("threshold_source")

    conservative_threshold = None
    if open_set_threshold is not None and unknown_scores:
        conservative_threshold = max(0.0, open_set_threshold - 5.0)

    threshold_candidates = [
        {
            "label": "runtime_threshold",
            "threshold": evaluation_threshold,
        },
        {
            "label": "saved_evaluation_threshold",
            "threshold": saved_evaluation_threshold,
        },
        {
            "label": "suggested_open_set_threshold",
            "threshold": open_set_threshold,
        },
        {
            "label": "conservative_threshold",
            "threshold": conservative_threshold,
        },
    ]
    threshold_results = evaluate_open_set_threshold_candidates(
        evaluation_records,
        threshold_candidates,
    )

    log_recognition_evaluation_diagnostics(
        "CUSTOM OPEN SET",
        load_elapsed,
        metrics_report,
        known_stats=known_stats,
        unknown_stats=unknown_stats,
        threshold_results=threshold_results,
        saved_evaluation_threshold=saved_evaluation_threshold,
        saved_evaluation_threshold_source=saved_evaluation_threshold_source,
        total_count=len(probe_records),
        known_count=sum(known_dataset_counts.values()),
        unknown_count=sum(unknown_dataset_counts.values()),
        evaluation_elapsed=time.perf_counter() - load_start_time,
        open_set=True,
        pairwise_analysis=pairwise_analysis,
        pose_summary=pose_summary,
    )

    return


def evaluate_single_recognition_dataset(
    train_mode,
    eval_mode,
    recognizer,
    label_map,
    namespace_identities=False,
    allow_config_update=False,
    output_stem=None,
    lfw_sample_ratio=None,
):
    """
    Evaluate the trained recognition model on one selected probe dataset.

    The function computes the global summary, richer score statistics,
    a per-identity breakdown, a small threshold analysis, and optionally
    updates the stored offline threshold reference on disk.
    """
    if eval_mode == "custom":
        evaluate_custom_open_set_dataset(
            train_mode,
            recognizer,
            label_map,
            allow_config_update=allow_config_update,
            output_stem=output_stem,
            lfw_sample_ratio=lfw_sample_ratio,
        )
        return

    print(f"=== RECOGNITION EVALUATION MODE: {eval_mode.upper()} ===")
    preprocessing_config = get_recognition_preprocessing_config()
    print(
        "Recognition preprocessing:",
        f"normalization={preprocessing_config['normalization_mode']},",
        f"padding_ratio={preprocessing_config['crop_padding_ratio']:.3f}",
    )
    if eval_mode == "lfw" and lfw_sample_ratio is not None and lfw_sample_ratio < 1.0:
        print(f"LFW sample ratio: {lfw_sample_ratio:.3f}")

    load_start_time = time.perf_counter()
    probe_records, dataset_counts, pose_summary = load_probe_dataset(
        eval_mode,
        namespace_identities=namespace_identities,
        collect_pose_metrics=True,
        lfw_sample_ratio=lfw_sample_ratio if lfw_sample_ratio is not None else 1.0,
    )
    load_elapsed = time.perf_counter() - load_start_time

    if not probe_records:
        print("No valid probe images found for evaluation.")
        print()
        return

    genuine_scores = []
    impostor_scores = []
    normalized_records = []
    evaluation_records = []
    correct = 0
    total = 0
    evaluation_start_time = time.perf_counter()

    for index, record in enumerate(probe_records, start=1):
        label, confidence = recognizer.predict(record["image"])
        predicted_name = normalize_identity_name(label_map[label])
        true_name = normalize_identity_name(record["identity"])

        if predicted_name == true_name:
            genuine_scores.append(confidence)
            correct += 1
        else:
            impostor_scores.append(confidence)

        evaluation_records.append(
            {
                "true_identity": true_name,
                "predicted_identity": predicted_name,
                "confidence": confidence,
                "is_correct": predicted_name == true_name,
            }
        )
        normalized_records.append(
            {
                "is_known": True,
                "true_id": true_name,
                "predicted_id": predicted_name,
                "score": confidence,
            }
        )

        total += 1

        if index % PROBE_PROGRESS_STEP == 0:
            print(f"Evaluated {index}/{len(probe_records)} probe images")

    genuine_stats = compute_score_statistics(genuine_scores)
    impostor_stats = compute_score_statistics(impostor_scores)
    identity_stats = compute_identity_accuracy(evaluation_records)
    report_stem = output_stem or f"{train_mode}_{eval_mode}{_build_lfw_sample_suffix(lfw_sample_ratio)}"

    metrics_report = compute_recognition_evaluation_metrics(
        normalized_records,
        DEFAULT_PREDICTION_THRESHOLD,
        output_stem=None,
    )
    pairwise_analysis = build_all_gallery_vs_all_probe_analysis(
        probe_records,
        recognizer,
        label_map,
        threshold=DEFAULT_PREDICTION_THRESHOLD,
    )
    metrics_report["pairwise_analysis"] = pairwise_analysis
    metrics_report["doddington_analysis"] = build_doddington_zoo_analysis(
        pairwise_analysis,
        threshold=DEFAULT_PREDICTION_THRESHOLD,
    )
    metrics_report["srr_analysis"] = build_system_response_reliability_analysis(
        pairwise_analysis,
        threshold=DEFAULT_PREDICTION_THRESHOLD,
    )
    metrics_report["export_paths"] = export_recognition_metrics_report(
        metrics_report,
        report_stem,
    )

    current_evaluation_config = load_evaluation_threshold_config()
    saved_evaluation_threshold = None
    saved_evaluation_threshold_source = None
    if current_evaluation_config is not None:
        saved_evaluation_threshold = current_evaluation_config.get("unknown_threshold")
        saved_evaluation_threshold_source = current_evaluation_config.get("threshold_source")

    suggested_threshold = None
    suggested_threshold_rule = None

    suggested_threshold, suggested_threshold_rule = compute_suggested_threshold(
        genuine_scores,
        impostor_scores,
    )

    conservative_threshold = None
    if suggested_threshold is not None and impostor_scores:
        conservative_threshold = max(0.0, suggested_threshold - 5.0)

    threshold_candidates = [
        {
            "label": "saved_evaluation_threshold",
            "threshold": saved_evaluation_threshold,
        },
        {
            "label": "suggested_threshold",
            "threshold": suggested_threshold,
        },
        {
            "label": "conservative_threshold",
            "threshold": conservative_threshold,
        },
    ]
    threshold_results = evaluate_threshold_candidates(
        evaluation_records,
        threshold_candidates,
    )

    has_both_classes = bool(genuine_scores) and bool(impostor_scores)
    non_fallback_rule = suggested_threshold_rule != "max_genuine_plus_5_fallback"
    can_update_evaluation_threshold = should_save_evaluation_threshold(
        train_mode,
        eval_mode,
        allow_config_update=allow_config_update,
    )

    if (
        can_update_evaluation_threshold
        and suggested_threshold is not None
        and has_both_classes
        and non_fallback_rule
    ):
        config_data = {
            "unknown_threshold": suggested_threshold,
            "threshold_source": f"{train_mode}_to_{eval_mode}",
            "train_mode": train_mode,
            "eval_mode": eval_mode,
            "genuine_matches": len(genuine_scores),
            "impostor_matches": len(impostor_scores),
            "mean_genuine_confidence": genuine_stats["mean"] if genuine_stats else None,
            "mean_impostor_confidence": impostor_stats["mean"] if impostor_stats else None,
            "suggested_threshold_rule": suggested_threshold_rule,
        }
        save_recognition_config(config_data)
        saved_evaluation_threshold = suggested_threshold
        saved_evaluation_threshold_source = f"{train_mode}_to_{eval_mode}"
        print(f"Recognition evaluation config saved to: {RECOGNITION_CONFIG_PATH}")
    elif allow_config_update:
        if not can_update_evaluation_threshold:
            print(
                "Evaluation threshold not updated: "
                "train/eval pair is not allowed to define the stored reference."
            )
        elif not has_both_classes:
            print(
                "Evaluation threshold not updated: evaluation does not contain both "
                "genuine and impostor scores."
            )
        elif not non_fallback_rule:
            print(
                "Evaluation threshold not updated: fallback threshold rule is not "
                "allowed for the stored reference."
            )

    log_recognition_evaluation_diagnostics(
        f"{train_mode.upper()} -> {eval_mode.upper()}",
        load_elapsed,
        metrics_report,
        known_stats=genuine_stats,
        unknown_stats=impostor_stats,
        identity_stats=identity_stats,
        threshold_results=threshold_results,
        saved_evaluation_threshold=saved_evaluation_threshold,
        saved_evaluation_threshold_source=saved_evaluation_threshold_source,
        dataset_counts=dataset_counts,
        total_count=total,
        correct_count=correct,
        genuine_count=len(genuine_scores),
        impostor_count=len(impostor_scores),
        evaluation_elapsed=time.perf_counter() - evaluation_start_time,
        open_set=False,
        pairwise_analysis=pairwise_analysis,
        pose_summary=pose_summary,
    )


def evaluate_recognition_model(train_mode=None, eval_mode=None, lfw_sample_ratio=None):
    """
    Evaluate the recognition model on one dataset or on all supported
    datasets sequentially.

    The standalone defaults match the current project reference setup:
    `custom -> custom`.
    """
    selected_train_mode = train_mode or "custom"
    selected_eval_mode = eval_mode or EVAL_MODE
    selected_lfw_sample_ratio = lfw_sample_ratio
    namespace_identities = should_namespace_identities(selected_train_mode)
    total_start_time = time.perf_counter()

    recognizer, label_map = load_trained_recognition_model()

    if selected_eval_mode == "all":
        for mode in ("custom", "orl", "lfw"):
            evaluate_single_recognition_dataset(
                selected_train_mode,
                mode,
                recognizer,
                label_map,
                namespace_identities=namespace_identities,
                allow_config_update=False,
                lfw_sample_ratio=selected_lfw_sample_ratio,
            )
        print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")
        return

    evaluate_single_recognition_dataset(
        selected_train_mode,
        selected_eval_mode,
        recognizer,
        label_map,
        namespace_identities=namespace_identities,
        allow_config_update=True,
        lfw_sample_ratio=selected_lfw_sample_ratio,
    )
    print(f"Total evaluation time: {time.perf_counter() - total_start_time:.2f}s")


if __name__ == "__main__":
    evaluate_recognition_model()
