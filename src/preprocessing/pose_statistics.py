"""
Module: pose_statistics

Description:
This module aggregates pose-metric observations for offline training
and evaluation reports.

It keeps the project-level logic lightweight: callers can feed
per-sample pose dictionaries and later print compact summaries without
changing the model pipeline.
"""


POSE_METRIC_KEYS = (
    "roll_degrees",
    "roll_score",
    "yaw_score",
    "pitch_score",
    "sp_score",
)


def create_pose_summary():
    """
    Create an empty pose summary accumulator.
    """

    summary = {
        "samples_with_pose": 0,
        "samples_without_pose": 0,
    }

    for metric_key in POSE_METRIC_KEYS:
        summary[metric_key] = {
            "count": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
        }

    return summary


def update_pose_summary(summary, pose_metrics):
    """
    Update an accumulator with one pose-metrics observation.
    """

    if summary is None:
        return None

    if pose_metrics is None:
        summary["samples_without_pose"] += 1
        return summary

    summary["samples_with_pose"] += 1

    for metric_key in POSE_METRIC_KEYS:
        value = pose_metrics.get(metric_key)
        if value is None:
            continue

        metric_summary = summary[metric_key]
        metric_summary["count"] += 1
        metric_summary["sum"] += float(value)
        metric_summary["min"] = (
            float(value)
            if metric_summary["min"] is None
            else min(metric_summary["min"], float(value))
        )
        metric_summary["max"] = (
            float(value)
            if metric_summary["max"] is None
            else max(metric_summary["max"], float(value))
        )

    return summary


def merge_pose_summaries(base_summary, other_summary):
    """
    Merge two pose summaries into one accumulator.
    """

    if base_summary is None:
        return other_summary

    if other_summary is None:
        return base_summary

    base_summary["samples_with_pose"] += other_summary["samples_with_pose"]
    base_summary["samples_without_pose"] += other_summary["samples_without_pose"]

    for metric_key in POSE_METRIC_KEYS:
        base_metric = base_summary[metric_key]
        other_metric = other_summary[metric_key]
        base_metric["count"] += other_metric["count"]
        base_metric["sum"] += other_metric["sum"]
        if other_metric["min"] is not None:
            base_metric["min"] = (
                other_metric["min"]
                if base_metric["min"] is None
                else min(base_metric["min"], other_metric["min"])
            )
        if other_metric["max"] is not None:
            base_metric["max"] = (
                other_metric["max"]
                if base_metric["max"] is None
                else max(base_metric["max"], other_metric["max"])
            )

    return base_summary


def _format_optional_float(value, digits=2):
    """
    Format a possibly-missing floating-point value.
    """

    if value is None:
        return "n/a"

    return f"{value:.{digits}f}"


def _format_metric_line(metric_name, metric_summary, digits=2):
    """
    Format one metric row for a compact report line.
    """

    if metric_summary["count"] == 0:
        return f"{metric_name}=n/a"

    mean_value = metric_summary["sum"] / metric_summary["count"]
    return (
        f"{metric_name}=mean:{_format_optional_float(mean_value, digits)} | "
        f"min:{_format_optional_float(metric_summary['min'], digits)} | "
        f"max:{_format_optional_float(metric_summary['max'], digits)}"
    )


def format_pose_summary(summary, title=None):
    """
    Convert a pose summary accumulator into a readable multi-part string.
    """

    if summary is None:
        return "pose_summary=n/a"

    total_samples = summary["samples_with_pose"] + summary["samples_without_pose"]
    coverage = (
        (summary["samples_with_pose"] / total_samples) * 100.0
        if total_samples > 0
        else 0.0
    )

    lines = []
    if title is not None:
        lines.append(title)

    lines.append(
        "pose_samples="
        f"{summary['samples_with_pose']}/{total_samples} "
        f"({coverage:.1f}%)"
    )
    lines.append(f"pose_missing={summary['samples_without_pose']}")

    for metric_key in POSE_METRIC_KEYS:
        lines.append(_format_metric_line(metric_key, summary[metric_key]))

    return " | ".join(lines)
