import csv

import pytest

from src.recognition import recognition_quality_filter as quality_filter


def _quality_record(decision="reject", reason="too_dark"):
    return {
        "dataset": "custom",
        "split": "gallery",
        "identity": "alice",
        "sample_index": 1,
        "relative_path": "data/recognition/custom_raw/gallery/alice/001.png",
        "decision": decision,
        "reason": reason,
        "all_reasons": reason if decision != "valid" else "valid",
        "blur_score": 10.0,
        "mean_intensity": 20.0,
        "contrast_std": 5.0,
        "dark_ratio": 0.9,
        "bright_ratio": 0.0,
        "edge_density": 0.0,
        "visibility_score": 1.0,
        "yaw_score": 1.0,
        "pitch_score": 1.0,
        "roll_degrees": 0.0,
        "sp_score": 1.0,
        "width": 128,
        "height": 128,
    }


@pytest.fixture
def isolated_quality_exports(tmp_path, monkeypatch):
    monkeypatch.setattr(
        quality_filter,
        "RECOGNITION_QUALITY_OFFICIAL_DIR",
        tmp_path / "official",
    )
    monkeypatch.setattr(
        quality_filter,
        "RECOGNITION_QUALITY_DEBUG_METRICS_DIR",
        tmp_path / "debug_metrics",
    )
    monkeypatch.setattr(
        quality_filter,
        "RECOGNITION_QUALITY_DEBUG_ROOT",
        tmp_path / "debug_root",
    )
    quality_filter._QUALITY_REPORT_INITIALIZED_PATHS.clear()
    quality_filter._QUALITY_SPLIT_SUMMARY_INITIALIZED_PATHS.clear()
    quality_filter._QUALITY_DEBUG_VALUES_INITIALIZED_PATHS.clear()
    quality_filter._QUALITY_DEBUG_COUNTS.clear()
    return tmp_path


def test_quality_report_export_uses_stem_aware_filename(isolated_quality_exports):
    output_path = quality_filter.export_quality_records(
        [_quality_record()],
        output_stem="recognition_custom_gallery",
    )

    assert output_path.name == "recognition_custom_gallery_quality_report.csv"
    with open(output_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["split"] == "gallery"


def test_quality_split_summary_export_uses_stem_aware_filename(
    isolated_quality_exports,
):
    output_path = quality_filter.export_quality_split_summary(
        [_quality_record()],
        quality_filter_mode="report_only",
        output_stem="recognition_custom_probe",
    )

    assert output_path.name == "recognition_custom_probe_quality_split_summary.csv"
    with open(output_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["quality_filter_mode"] == "report_only"
    assert rows[0]["top_reject_reason"] == "n/a"


def test_quality_threshold_summary_export_uses_stem_aware_filename(
    isolated_quality_exports,
):
    output_path = quality_filter.export_quality_threshold_summary(
        output_stem="recognition_custom_probe",
    )

    assert output_path.name == "recognition_custom_probe_quality_threshold_summary.txt"
    assert output_path.read_text(encoding="utf-8").startswith(
        "Recognition quality thresholds"
    )


def test_quality_debug_values_export_uses_stem_aware_filename(
    isolated_quality_exports,
):
    debug_image_path = isolated_quality_exports / "debug.png"
    debug_image_path.write_bytes(b"not-an-image-but-a-path-placeholder")

    output_path = quality_filter.export_quality_debug_record(
        _quality_record(),
        debug_image_path,
        output_stem="recognition_custom_probe",
    )

    assert output_path.name == "recognition_custom_probe_quality_debug_values.csv"
    with open(output_path, newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["debug_image_path"].endswith("debug.png")
