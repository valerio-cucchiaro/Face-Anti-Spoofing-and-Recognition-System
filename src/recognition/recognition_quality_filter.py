"""
Module: recognition_quality_filter

Description:
Quality diagnostics for recognition samples.

The module records whether a sample would be considered suspicious under
conservative biometric-quality checks. The dataset loader can use this in
report-only mode or in a separate probe-filtered protocol.
"""

import csv
from collections import Counter
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RECOGNITION_QUALITY_METRICS_DIR = PROJECT_ROOT / "logs" / "metrics" / "recognition"
RECOGNITION_QUALITY_OFFICIAL_DIR = RECOGNITION_QUALITY_METRICS_DIR / "official"
RECOGNITION_QUALITY_DEBUG_METRICS_DIR = RECOGNITION_QUALITY_METRICS_DIR / "debug"
RECOGNITION_QUALITY_DEBUG_ROOT = PROJECT_ROOT / "logs" / "debug" / "recognition_quality"
RECOGNITION_QUALITY_DEBUG_LIMIT = 20
_QUALITY_REPORT_INITIALIZED_PATHS = set()
_QUALITY_SPLIT_SUMMARY_INITIALIZED_PATHS = set()
_QUALITY_DEBUG_VALUES_INITIALIZED_PATHS = set()
_QUALITY_DEBUG_COUNTS = {}

# Conservative quality thresholds. They are intentionally broad so report-only
# runs can identify severe cases without changing the protocol.
RECOGNITION_MIN_BLUR_SCORE = 20.0
RECOGNITION_MIN_MEAN_INTENSITY = 35.0
RECOGNITION_MIN_CONTRAST_STD = 18.0
RECOGNITION_MAX_DARK_RATIO = 0.75
RECOGNITION_MIN_EDGE_DENSITY = 0.015
RECOGNITION_MIN_YAW_SCORE = 0.60
RECOGNITION_MIN_PITCH_SCORE = 0.45
RECOGNITION_MAX_ABS_ROLL_DEGREES = 25.0
RECOGNITION_MIN_SP_SCORE = 0.60

QUALITY_FIELDNAMES = [
    "dataset",
    "split",
    "identity",
    "sample_index",
    "relative_path",
    "decision",
    "reason",
    "all_reasons",
    "blur_score",
    "mean_intensity",
    "contrast_std",
    "dark_ratio",
    "bright_ratio",
    "edge_density",
    "visibility_score",
    "yaw_score",
    "pitch_score",
    "roll_degrees",
    "sp_score",
    "width",
    "height",
]
DEBUG_FIELDNAMES = [
    "decision",
    "reason",
    "all_reasons",
    "dataset",
    "split",
    "identity",
    "sample_index",
    "relative_path",
    "debug_image_path",
    "blur_score",
    "mean_intensity",
    "contrast_std",
    "dark_ratio",
    "bright_ratio",
    "edge_density",
    "visibility_score",
    "yaw_score",
    "pitch_score",
    "roll_degrees",
    "sp_score",
    "width",
    "height",
]
QUALITY_REASONS = [
    "valid",
    "invalid_crop",
    "too_dark",
    "low_contrast",
    "poor_visibility",
    "too_blurry",
    "low_detail",
    "too_much_yaw",
    "too_much_pitch",
    "too_much_roll",
    "poor_spatial_pose",
    "landmarks_missing",
]
QUALITY_SPLIT_SUMMARY_FIELDNAMES = [
    "dataset",
    "split",
    "quality_filter_mode",
    "total",
    "kept_samples",
    "valid_samples",
    "warning_samples",
    "rejected_samples",
    "rejected_percent",
    "top_reject_reason",
    "identities_before",
    "identities_after",
] + [f"reason_{reason}" for reason in QUALITY_REASONS]
QUALITY_FILTER_MODES = (
    "none",
    "report_only",
    "probe_only",
)
WARNING_ONLY_REASONS = {
    "too_much_pitch",
    "too_much_roll",
    "poor_spatial_pose",
}
HARD_REJECT_REASONS = {
    reason
    for reason in QUALITY_REASONS
    if reason not in WARNING_ONLY_REASONS and reason != "valid"
}


def _sanitize_filename_part(value):
    text = str(value).replace("\\", "_").replace("/", "_").replace(":", "_")
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in text)


def _relative_project_path(path):
    """Return a readable project-relative path for CSV exports."""

    path = Path(path)
    try:
        return path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_quality_output_path(root_dir, output_stem, default_name, stem_suffix):
    """Resolve one optional stem-aware quality export path."""

    if output_stem is None:
        return root_dir / default_name

    return root_dir / f"{output_stem}_{stem_suffix}"


def _as_grayscale(image):
    if image is None or image.size == 0:
        return None

    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def _metric_value(pose_metrics, key):
    if pose_metrics is None:
        return None
    return pose_metrics.get(key)


def _compute_edge_density(gray_image):
    edges = cv2.Canny(gray_image, 50, 150)
    return float((edges > 0).mean())


def _collect_quality_reasons(
    mean_intensity,
    contrast_std,
    dark_ratio,
    blur_score,
    edge_density,
    pose_metrics,
    yaw_score,
    pitch_score,
    roll_degrees,
    sp_score,
):
    """Return all quality reasons triggered by one sample."""

    reasons = []
    if mean_intensity < RECOGNITION_MIN_MEAN_INTENSITY:
        reasons.append("too_dark")
    if contrast_std < RECOGNITION_MIN_CONTRAST_STD:
        reasons.append("low_contrast")
    if dark_ratio > RECOGNITION_MAX_DARK_RATIO:
        reasons.append("poor_visibility")
    if blur_score < RECOGNITION_MIN_BLUR_SCORE:
        reasons.append("too_blurry")
    if edge_density < RECOGNITION_MIN_EDGE_DENSITY:
        reasons.append("low_detail")
    if pose_metrics is None:
        reasons.append("landmarks_missing")
    else:
        if yaw_score is not None and yaw_score < RECOGNITION_MIN_YAW_SCORE:
            reasons.append("too_much_yaw")
        if pitch_score is not None and pitch_score < RECOGNITION_MIN_PITCH_SCORE:
            reasons.append("too_much_pitch")
        if (
            roll_degrees is not None
            and abs(float(roll_degrees)) > RECOGNITION_MAX_ABS_ROLL_DEGREES
        ):
            reasons.append("too_much_roll")
        if sp_score is not None and sp_score < RECOGNITION_MIN_SP_SCORE:
            reasons.append("poor_spatial_pose")

    return reasons


def _quality_decision_from_reasons(reasons):
    """Return valid, warning, or reject from the triggered reasons."""

    if not reasons:
        return "valid"

    if any(reason in HARD_REJECT_REASONS for reason in reasons):
        return "reject"

    return "warning"


def assess_recognition_sample_quality(image, pose_metrics=None):
    """
    Return the quality decision and measured values for one recognition sample.
    """

    gray_image = _as_grayscale(image)
    if gray_image is None:
        return {
            "decision": "reject",
            "reason": "invalid_crop",
            "all_reasons": "invalid_crop",
            "blur_score": 0.0,
            "mean_intensity": 0.0,
            "contrast_std": 0.0,
            "dark_ratio": 0.0,
            "bright_ratio": 0.0,
            "edge_density": 0.0,
            "visibility_score": 0.0,
            "yaw_score": None,
            "pitch_score": None,
            "roll_degrees": None,
            "sp_score": None,
            "width": 0,
            "height": 0,
        }

    height, width = gray_image.shape[:2]
    blur_score = float(cv2.Laplacian(gray_image, cv2.CV_64F).var())
    mean_intensity = float(gray_image.mean())
    contrast_std = float(gray_image.std())
    dark_ratio = float((gray_image < 35).mean())
    bright_ratio = float((gray_image > 220).mean())
    edge_density = _compute_edge_density(gray_image)
    visibility_score = contrast_std * (1.0 - dark_ratio)
    yaw_score = _metric_value(pose_metrics, "yaw_score")
    pitch_score = _metric_value(pose_metrics, "pitch_score")
    roll_degrees = _metric_value(pose_metrics, "roll_degrees")
    sp_score = _metric_value(pose_metrics, "sp_score")

    reasons = _collect_quality_reasons(
        mean_intensity,
        contrast_std,
        dark_ratio,
        blur_score,
        edge_density,
        pose_metrics,
        yaw_score,
        pitch_score,
        roll_degrees,
        sp_score,
    )
    reason = reasons[0] if reasons else "valid"
    decision = _quality_decision_from_reasons(reasons)

    return {
        "decision": decision,
        "reason": reason,
        "all_reasons": ";".join(reasons) if reasons else "valid",
        "blur_score": blur_score,
        "mean_intensity": mean_intensity,
        "contrast_std": contrast_std,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
        "edge_density": edge_density,
        "visibility_score": visibility_score,
        "yaw_score": yaw_score,
        "pitch_score": pitch_score,
        "roll_degrees": roll_degrees,
        "sp_score": sp_score,
        "width": width,
        "height": height,
    }


def build_quality_record(
    dataset_name,
    split_name,
    identity_name,
    sample_index,
    image_path,
    quality_result,
):
    """Build one CSV-friendly quality row."""

    record = {
        "dataset": dataset_name,
        "split": split_name,
        "identity": identity_name,
        "sample_index": sample_index,
        "relative_path": _relative_project_path(image_path),
    }
    record.update(quality_result)
    return record


def normalize_quality_filter_mode(mode=None):
    """Return the effective quality-filter mode."""

    if mode is None:
        return "report_only"

    if mode not in QUALITY_FILTER_MODES:
        raise ValueError(
            f"Unsupported recognition quality filter mode: {mode}. "
            f"Expected one of: {', '.join(QUALITY_FILTER_MODES)}"
        )

    return mode


def should_reject_quality_record(record, mode=None, split_name=None):
    """Return True when a quality record should be skipped by the loader."""

    normalized_mode = normalize_quality_filter_mode(mode)

    if normalized_mode in {"none", "report_only"}:
        return False
    if normalized_mode == "probe_only" and split_name == "gallery":
        return False

    return record["decision"] == "reject"


def save_quality_debug_image(record, image, debug_counts):
    """
    Save a small number of suspicious sample previews per category.
    """

    global _QUALITY_DEBUG_COUNTS

    if record["decision"] == "valid":
        return

    reason = record["reason"]
    current_count = _QUALITY_DEBUG_COUNTS.get(reason, 0)
    if current_count >= RECOGNITION_QUALITY_DEBUG_LIMIT:
        return None

    gray_image = _as_grayscale(image)
    if gray_image is None:
        return None

    debug_dir = RECOGNITION_QUALITY_DEBUG_ROOT / _sanitize_filename_part(reason)
    debug_dir.mkdir(parents=True, exist_ok=True)

    filename = (
        f"{_sanitize_filename_part(record['split'])}_"
        f"{_sanitize_filename_part(record['identity'])}_"
        f"{int(record['sample_index']):04d}_"
        f"{_sanitize_filename_part(record['dataset'])}_"
        f"{_sanitize_filename_part(Path(record['relative_path']).stem)}.png"
    )
    debug_image_path = debug_dir / filename
    cv2.imwrite(str(debug_image_path), gray_image)
    _QUALITY_DEBUG_COUNTS[reason] = current_count + 1
    return debug_image_path


def export_quality_debug_record(record, debug_image_path, output_stem=None):
    """Append one saved debug-image row with its quality values."""

    if debug_image_path is None:
        return None

    output_path = _resolve_quality_output_path(
        RECOGNITION_QUALITY_DEBUG_ROOT,
        output_stem,
        "quality_debug_values.csv",
        "quality_debug_values.csv",
    )
    output_key = str(output_path.resolve())
    write_header = output_key not in _QUALITY_DEBUG_VALUES_INITIALIZED_PATHS

    RECOGNITION_QUALITY_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if not write_header else "w"
    with open(output_path, write_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=DEBUG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        row = {key: record.get(key) for key in DEBUG_FIELDNAMES}
        row["debug_image_path"] = _relative_project_path(debug_image_path)
        writer.writerow(row)

    _QUALITY_DEBUG_VALUES_INITIALIZED_PATHS.add(output_key)
    return output_path


def export_quality_threshold_summary(output_stem=None):
    """Write the active quality thresholds for debugging."""

    output_path = _resolve_quality_output_path(
        RECOGNITION_QUALITY_DEBUG_ROOT,
        output_stem,
        "quality_threshold_summary.txt",
        "quality_threshold_summary.txt",
    )
    RECOGNITION_QUALITY_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "Recognition quality thresholds",
        f"too_dark: mean_intensity < {RECOGNITION_MIN_MEAN_INTENSITY}",
        f"low_contrast: contrast_std < {RECOGNITION_MIN_CONTRAST_STD}",
        f"poor_visibility: dark_ratio > {RECOGNITION_MAX_DARK_RATIO}",
        f"too_blurry: laplacian_var < {RECOGNITION_MIN_BLUR_SCORE}",
        f"low_detail: edge_density < {RECOGNITION_MIN_EDGE_DENSITY}",
        "landmarks_missing: pose_metrics is None",
        f"too_much_yaw: yaw_score < {RECOGNITION_MIN_YAW_SCORE}",
        f"too_much_pitch warning-only: pitch_score < {RECOGNITION_MIN_PITCH_SCORE}",
        f"too_much_roll warning-only: abs(roll_degrees) > {RECOGNITION_MAX_ABS_ROLL_DEGREES}",
        f"poor_spatial_pose warning-only: sp_score < {RECOGNITION_MIN_SP_SCORE}",
    ]
    output_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return output_path


def export_quality_records(records, output_stem=None):
    """Append quality records to one recognition quality CSV."""

    if not records:
        return None

    output_path = _resolve_quality_output_path(
        RECOGNITION_QUALITY_DEBUG_METRICS_DIR,
        output_stem,
        "recognition_quality_report.csv",
        "quality_report.csv",
    )
    output_key = str(output_path.resolve())
    write_header = output_key not in _QUALITY_REPORT_INITIALIZED_PATHS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if not write_header else "w"
    with open(output_path, write_mode, newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key) for key in QUALITY_FIELDNAMES})

    _QUALITY_REPORT_INITIALIZED_PATHS.add(output_key)
    return output_path


def _build_quality_split_summary_rows(records, quality_filter_mode):
    """Build per-split summary rows from quality records."""

    rows = []
    normalized_mode = normalize_quality_filter_mode(quality_filter_mode)
    grouped_records = {}
    for record in records:
        key = (record["dataset"], record["split"])
        grouped_records.setdefault(key, []).append(record)

    for (dataset_name, split_name), split_records in sorted(grouped_records.items()):
        reason_counts = Counter(record["reason"] for record in split_records)
        effective_reject_records = [
            record
            for record in split_records
            if should_reject_quality_record(
                record,
                mode=normalized_mode,
                split_name=split_name,
            )
        ]
        reject_reason_counts = Counter(
            record["reason"] for record in effective_reject_records
        )
        total = len(split_records)
        valid = sum(1 for record in split_records if record["decision"] == "valid")
        warning = sum(1 for record in split_records if record["decision"] == "warning")
        rejected = len(effective_reject_records)
        kept = total - rejected
        identities_before = {record["identity"] for record in split_records}
        if normalized_mode in {"none", "report_only"} or (
            normalized_mode == "probe_only" and split_name == "gallery"
        ):
            identities_after = identities_before
        else:
            identities_after = {
                record["identity"]
                for record in split_records
                if not should_reject_quality_record(
                    record,
                    mode=normalized_mode,
                    split_name=split_name,
                )
            }
        top_reject_reason = "n/a"
        if reject_reason_counts:
            top_reject_reason = reject_reason_counts.most_common(1)[0][0]

        row = {
            "dataset": dataset_name,
            "split": split_name,
            "quality_filter_mode": normalized_mode,
            "total": total,
            "kept_samples": kept,
            "valid_samples": valid,
            "warning_samples": warning,
            "rejected_samples": rejected,
            "rejected_percent": round((rejected / total) * 100.0, 2) if total else 0.0,
            "top_reject_reason": top_reject_reason,
            "identities_before": len(identities_before),
            "identities_after": len(identities_after),
        }
        for reason in QUALITY_REASONS:
            row[f"reason_{reason}"] = reason_counts.get(reason, 0)
        rows.append(row)

    return rows


def export_quality_split_summary(
    records,
    quality_filter_mode="report_only",
    output_stem=None,
):
    """Append per-split quality summary rows to one summary CSV."""

    if not records:
        return None

    rows = _build_quality_split_summary_rows(records, quality_filter_mode)
    if not rows:
        return None

    output_path = _resolve_quality_output_path(
        RECOGNITION_QUALITY_OFFICIAL_DIR,
        output_stem,
        "recognition_quality_split_summary.csv",
        "quality_split_summary.csv",
    )
    output_key = str(output_path.resolve())
    write_header = output_key not in _QUALITY_SPLIT_SUMMARY_INITIALIZED_PATHS

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_mode = "a" if not write_header else "w"
    with open(
        output_path,
        write_mode,
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=QUALITY_SPLIT_SUMMARY_FIELDNAMES)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: row.get(key) for key in QUALITY_SPLIT_SUMMARY_FIELDNAMES}
            )

    _QUALITY_SPLIT_SUMMARY_INITIALIZED_PATHS.add(output_key)
    return output_path


def format_quality_summary(records, title=None):
    """Return a compact split-level summary for quality diagnostics."""

    if not records:
        return "Recognition quality summary: no records"

    reason_counts = Counter(record["reason"] for record in records)
    decision_counts = Counter(record["decision"] for record in records)
    total = len(records)
    parts = [
        f"total={total}",
        f"valid={decision_counts.get('valid', 0)}",
        f"warning={decision_counts.get('warning', 0)}",
        f"reject={decision_counts.get('reject', 0)}",
    ]
    for reason in QUALITY_REASONS:
        if reason == "valid":
            continue
        parts.append(f"{reason}={reason_counts.get(reason, 0)}")

    prefix = title or "Recognition quality summary:"
    return f"{prefix} | " + " | ".join(parts)
