import numpy as np

import src.evaluation.pad_model_evaluator as pad_model_evaluator


class _DummyScaler:
    def transform(self, x):
        return x


class _DummyClassifier:
    def predict(self, x):
        return np.ones(len(x), dtype=np.int32)


def test_evaluate_pad_validation_protocol_reuses_validation_threshold_for_final_test(monkeypatch):
    compute_calls = []
    diagnostics_calls = []

    def fake_load_configured_test_dataset(mode, **kwargs):
        if mode == "casia_validation":
            return "CASIA validation", (
                np.array([[0.1], [0.2]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            )
        assert mode == "casia"
        return "CASIA test", (
            np.array([[0.3], [0.4]], dtype=np.float32),
            np.array([1, 0], dtype=np.int32),
            None,
        )

    def fake_get_live_score_details(classifier, x_scaled):
        return np.array([0.9, 0.1], dtype=np.float64), "live_prob"

    def fake_sweep_pad_thresholds(y_true, y_score):
        return {"per_threshold": [{"threshold": 0.77, "apcer": 0.0, "bpcer": 0.0, "acer": 0.0}]}

    def fake_select_pad_threshold_by_target_apcer(sweep_results, target_apcer):
        return {
            "threshold": 0.77,
            "threshold_rule": "target_apcer",
            "target_apcer": target_apcer,
            "constraint_satisfied": True,
            "selected_metrics": {"threshold": 0.77, "apcer": 0.0, "bpcer": 0.0, "acer": 0.0},
        }

    def fake_compute_pad_evaluation_metrics(*args, **kwargs):
        compute_calls.append(kwargs.copy())
        return {
            "selected_metrics": {
                "threshold": kwargs["selected_threshold"],
                "far": 0.0,
                "frr": 0.0,
                "gar": 1.0,
                "apcer": 0.0,
                "bpcer": 0.0,
                "acer": 0.0,
            },
            "threshold_summary": {},
            "eer_results": {},
            "export_paths": {},
        }

    monkeypatch.setattr(pad_model_evaluator, "load_configured_test_dataset", fake_load_configured_test_dataset)
    monkeypatch.setattr(pad_model_evaluator, "get_live_score_details", fake_get_live_score_details)
    monkeypatch.setattr(pad_model_evaluator, "sweep_pad_thresholds", fake_sweep_pad_thresholds)
    monkeypatch.setattr(
        pad_model_evaluator,
        "select_pad_threshold_by_target_apcer",
        fake_select_pad_threshold_by_target_apcer,
    )
    monkeypatch.setattr(pad_model_evaluator, "compute_pad_evaluation_metrics", fake_compute_pad_evaluation_metrics)
    monkeypatch.setattr(
        pad_model_evaluator,
        "log_pad_evaluation_diagnostics",
        lambda dataset_name, *args, **kwargs: diagnostics_calls.append(dataset_name),
    )

    pad_model_evaluator.evaluate_pad_validation_protocol(
        "casia_validation",
        "casia",
        "CASIA",
        _DummyScaler(),
        _DummyClassifier(),
        output_stem="unit_pad",
        target_apcer=0.02,
    )

    assert len(compute_calls) == 2
    assert compute_calls[0]["selected_threshold"] == 0.77
    assert compute_calls[0]["threshold_source"] == "casia_validation_target_apcer_selection"
    assert compute_calls[0]["output_stem"] == "unit_pad_selection"
    assert compute_calls[1]["selected_threshold"] == 0.77
    assert compute_calls[1]["threshold_source"] == "casia_validation_target_apcer"
    assert compute_calls[1]["output_stem"] == "unit_pad"
    assert diagnostics_calls == ["CASIA validation", "CASIA test (threshold from CASIA validation)"]


def test_evaluate_custom_pad_protocol_uses_saved_validation_metadata(monkeypatch):
    compute_calls = []

    monkeypatch.setattr(
        pad_model_evaluator,
        "load_custom_pad_validation_dataset",
        lambda **kwargs: (
            np.array([[0.1], [0.2]], dtype=np.float32),
            np.array([1, 0], dtype=np.int32),
            None,
        ),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "load_custom_pad_test_dataset",
        lambda **kwargs: (
            np.array([[0.3], [0.4]], dtype=np.float32),
            np.array([1, 0], dtype=np.int32),
            None,
        ),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "get_live_score_details",
        lambda classifier, x_scaled: (np.array([0.8, 0.2], dtype=np.float64), "live_prob"),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "sweep_pad_thresholds",
        lambda y_true, y_score: {"per_threshold": [{"threshold": 0.66, "apcer": 0.0, "bpcer": 0.1, "acer": 0.05}]},
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "select_pad_threshold_by_target_apcer",
        lambda sweep_results, target_apcer: {
            "threshold": 0.66,
            "threshold_rule": "target_apcer",
            "target_apcer": target_apcer,
            "constraint_satisfied": True,
            "selected_metrics": {"threshold": 0.66, "apcer": 0.0, "bpcer": 0.1, "acer": 0.05},
        },
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "compute_pad_evaluation_metrics",
        lambda *args, **kwargs: compute_calls.append(kwargs.copy()) or {
            "selected_metrics": {
                "threshold": kwargs["selected_threshold"],
                "far": 0.0,
                "frr": 0.0,
                "gar": 1.0,
                "apcer": 0.0,
                "bpcer": 0.1,
                "acer": 0.05,
            },
            "threshold_summary": {},
            "eer_results": {},
            "export_paths": {},
        },
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "log_pad_evaluation_diagnostics",
        lambda *args, **kwargs: None,
    )

    pad_model_evaluator.evaluate_custom_pad_protocol(
        _DummyScaler(),
        _DummyClassifier(),
        {
            "custom_validation_ratio": 0.3,
            "custom_validation_seed": 7,
        },
        output_stem="unit_custom",
    )

    assert len(compute_calls) == 2
    assert compute_calls[0]["threshold_source"] == "custom_internal_validation_target_apcer_selection"
    assert compute_calls[0]["output_stem"] == "unit_custom_selection"
    assert compute_calls[1]["threshold_source"] == "custom_internal_validation_target_apcer"
    assert compute_calls[1]["selected_threshold"] == 0.66
