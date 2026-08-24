import json
import pytest
from pathlib import Path

from src.evaluation import system_evaluator
from src.evaluation import recognition_model_evaluator
from src.evaluation import pad_model_evaluator
from src.evaluation.pad_metrics import export_pad_metrics_report
from src.evaluation.recognition_metrics import export_recognition_metrics_report


def _pad_threshold_row(threshold):
    return {
        "threshold": threshold,
        "far": 0.0,
        "frr": 0.0,
        "gar": 1.0,
        "apcer": 0.0,
        "bpcer": 0.0,
        "acer": 0.0,
    }


def _recognition_threshold_row(threshold):
    return {
        "threshold": threshold,
        "DIR": 1.0,
        "FAR": 0.0,
        "FRR": 0.0,
        "GAR": 1.0,
        "misidentification_rate": 0.0,
    }


def test_pad_metrics_json_contains_all_export_paths(tmp_path):
    report = {
        "threshold_summary": {
            "selected_thresholds": {"low": False, "mid": True, "high": False},
            "low": _pad_threshold_row(0.25),
            "mid": _pad_threshold_row(0.50),
            "high": _pad_threshold_row(0.75),
        },
        "sweep_results": {
            "roc_points": [{"threshold": 0.50, "far": 0.0, "gar": 1.0}],
            "det_points": [{"threshold": 0.50, "far": 0.0, "frr": 0.0}],
        },
    }

    export_paths = export_pad_metrics_report(
        report,
        "unit_pad",
        output_root=tmp_path,
        export_curves=True,
    )

    exported_json = json.loads(open(export_paths["json"], encoding="utf-8").read())

    assert exported_json["export_paths"] == {
        "json": "official/unit_pad_metrics_report.json",
        "summary_csv": "official/unit_pad_threshold_summary.csv",
        "roc_csv": "curves/unit_pad_roc_points.csv",
        "det_csv": "curves/unit_pad_det_points.csv",
    }
    assert set(export_paths) == {"json", "summary_csv", "roc_csv", "det_csv"}


def test_recognition_metrics_json_contains_all_export_paths(tmp_path):
    report = {
        "threshold_summary": {
            "selected_thresholds": {"low": False, "mid": True, "high": False},
            "low": _recognition_threshold_row(20.0),
            "mid": _recognition_threshold_row(30.0),
            "high": _recognition_threshold_row(40.0),
        },
        "sweep_results": {
            "roc_points": [{"threshold": 30.0, "far": 0.0, "gar": 1.0}],
            "det_points": [{"threshold": 30.0, "far": 0.0, "frr": 0.0}],
        },
        "pairwise_analysis": {
            "available": True,
            "rows": [{"sample_index": 0, "rank": 1}],
            "cms_curve": {
                "available": True,
                "points": [{"rank": 1, "cms": 1.0, "hit_count": 1, "total": 1}],
            },
        },
    }

    export_paths = export_recognition_metrics_report(
        report,
        "unit_recognition",
        output_root=tmp_path,
        export_curves=True,
    )

    exported_json = json.loads(open(export_paths["json"], encoding="utf-8").read())

    stored_export_paths = exported_json["export_paths"]

    assert set(stored_export_paths) == set(export_paths)

    for key, stored_path in stored_export_paths.items():
        assert not Path(stored_path).is_absolute()
        assert (tmp_path / stored_path).resolve() == Path(export_paths[key]).resolve()

    assert {
        "json",
        "summary_csv",
        "roc_csv",
        "det_csv",
        "all_gallery_vs_all_probe_cms_csv",
    }.issubset(export_paths)


def test_system_metrics_json_contains_all_export_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(system_evaluator, "SYSTEM_METRICS_ROOT", tmp_path)
    report = {"scenario": "unit"}
    rows = [
        {
            "sample_type": "known_live",
            "path": "sample.png",
            "true_id": "alice",
            "pad_label": "live",
            "pad_live_score": 1.0,
            "predicted_id": "alice",
            "recognition_distance": 10.0,
            "recognition_only_decision": "accept",
            "final_decision": "accept",
        }
    ]

    export_paths = system_evaluator.export_system_evaluation(
        rows,
        report,
        "unit_system",
    )

    exported_json = json.loads(open(export_paths["json"], encoding="utf-8").read())

    assert exported_json["export_paths"] == export_paths
    assert set(export_paths) == {"json", "samples_csv"}


def test_system_evaluator_uses_saved_pad_threshold_when_available(monkeypatch):
    monkeypatch.setattr(
        system_evaluator,
        "load_system_pad_threshold_config",
        lambda artifact_tag=None: {
            "threshold": 0.73,
            "threshold_source": "custom_internal_validation_target_apcer",
        },
    )

    threshold, source, config = system_evaluator.resolve_pad_threshold(None)

    assert threshold == 0.73
    assert source == "custom_internal_validation_target_apcer"
    assert config["threshold"] == 0.73


def test_system_evaluator_pad_threshold_raises_when_no_saved_config(monkeypatch):
    monkeypatch.setattr(
        system_evaluator,
        "load_system_pad_threshold_config",
        lambda artifact_tag=None: None,
    )

    with pytest.raises(ValueError, match="No system PAD threshold config found"):
        system_evaluator.resolve_pad_threshold(None)


def test_pad_evaluator_saved_threshold_requires_matching_artifact_config(monkeypatch):
    monkeypatch.setattr(
        pad_model_evaluator,
        "load_system_pad_threshold_config",
        lambda artifact_tag=None: {
            "threshold": 0.61,
            "threshold_source": "custom_internal_validation_target_apcer",
            "pad_config": {"feature_dim": 126, "pad_debug_feature_mode": "tgsd"},
        },
    )

    with pytest.raises(ValueError, match="does not match the active PAD artifacts"):
        pad_model_evaluator.resolve_saved_pad_evaluation_threshold(
            {"feature_dim": 124, "pad_debug_feature_mode": "baseline"}
        )


def test_recognition_evaluator_requires_saved_threshold_config(monkeypatch):
    monkeypatch.setattr(
        recognition_model_evaluator,
        "load_system_recognition_threshold_config",
        lambda: None,
    )

    with pytest.raises(ValueError, match="No system recognition threshold config found"):
        recognition_model_evaluator.resolve_saved_recognition_evaluation_threshold(
            {"recognizer_type": "lbph_single"}
        )
