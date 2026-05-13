import json

from src.evaluation import system_evaluator
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

    assert exported_json["export_paths"] == export_paths
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

    assert exported_json["export_paths"] == export_paths
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
