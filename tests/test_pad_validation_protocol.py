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

    def fake_load_configured_test_dataset(mode, split="test", **kwargs):
        if mode == "casia" and split == "validation":
            return "CASIA validation", (
                np.array([[0.1], [0.2]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            )
        assert mode == "casia"
        assert split == "test"
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


def test_evaluate_pad_validation_protocol_fails_when_no_threshold_is_selected(monkeypatch):
    monkeypatch.setattr(
        pad_model_evaluator,
        "load_configured_test_dataset",
        lambda mode, split="test", **kwargs: (
            "CASIA validation",
            (
                np.array([[0.1], [0.2]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            ),
        ) if mode == "casia" and split == "validation" else (
            "CASIA test",
            (
                np.array([[0.3], [0.4]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            ),
        ),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "get_live_score_details",
        lambda classifier, x_scaled: (np.array([0.9, 0.1], dtype=np.float64), "live_prob"),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "sweep_pad_thresholds",
        lambda y_true, y_score: {"per_threshold": []},
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "select_pad_threshold_by_target_apcer",
        lambda sweep_results, target_apcer: {
            "threshold": None,
            "threshold_rule": "unavailable",
            "target_apcer": target_apcer,
            "constraint_satisfied": False,
            "selected_metrics": None,
        },
    )

    try:
        pad_model_evaluator.evaluate_pad_validation_protocol(
            "casia",
            "CASIA",
            _DummyScaler(),
            _DummyClassifier(),
        )
    except ValueError as exc:
        assert "could not derive a PAD threshold" in str(exc)
    else:
        raise AssertionError("Expected ValueError when PAD validation selects no threshold.")


def test_evaluate_custom_pad_protocol_saves_system_threshold_config(monkeypatch):
    saved_configs = []

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
        lambda *args, **kwargs: {
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
    monkeypatch.setattr(
        pad_model_evaluator,
        "save_system_pad_threshold_config",
        lambda config_data, artifact_tag=None: saved_configs.append(config_data),
    )

    pad_model_evaluator.evaluate_custom_pad_protocol(
        _DummyScaler(),
        _DummyClassifier(),
        {
            "custom_validation_ratio": 0.3,
            "custom_validation_seed": 7,
            "pad_debug_feature_mode": "tgsd",
            "pad_lft_mode": "ggd_light",
            "feature_dim": 126,
        },
        allow_config_update=True,
    )

    assert len(saved_configs) == 1
    assert saved_configs[0]["threshold"] == 0.66
    assert saved_configs[0]["threshold_source"] == "custom_internal_validation_target_apcer"
    assert saved_configs[0]["pad_config"]["feature_dim"] == 126


def test_evaluate_all_pad_protocol_uses_one_combined_validation_threshold(monkeypatch):
    compute_calls = []
    test_calls = []
    saved_configs = []

    def fake_load_configured_test_dataset(mode, split="test", **kwargs):
        if split != "validation":
            raise AssertionError("Combined threshold selection should only load validation splits here.")
        if mode == "casia":
            return "CASIA validation", (
                np.array([[0.1], [0.2]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            )
        if mode == "replay":
            return "Replay-Attack validation", (
                np.array([[0.3], [0.4]], dtype=np.float32),
                np.array([1, 0], dtype=np.int32),
                None,
            )
        assert mode == "custom"
        return "Custom pseudo-validation", (
            np.array([[0.5], [0.6]], dtype=np.float32),
            np.array([1, 0], dtype=np.int32),
            None,
        )

    monkeypatch.setattr(pad_model_evaluator, "load_configured_test_dataset", fake_load_configured_test_dataset)
    monkeypatch.setattr(
        pad_model_evaluator,
        "get_live_score_details",
        lambda classifier, x_scaled: (np.linspace(0.9, 0.1, len(x_scaled), dtype=np.float64), "live_prob"),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "sweep_pad_thresholds",
        lambda y_true, y_score: {"per_threshold": [{"threshold": 0.55, "apcer": 0.0, "bpcer": 0.1, "acer": 0.05}]},
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "select_pad_threshold_by_target_apcer",
        lambda sweep_results, target_apcer: {
            "threshold": 0.55,
            "threshold_rule": "target_apcer",
            "target_apcer": target_apcer,
            "constraint_satisfied": True,
            "selected_metrics": {"threshold": 0.55, "apcer": 0.0, "bpcer": 0.1, "acer": 0.05},
        },
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "compute_pad_evaluation_metrics",
        lambda *args, **kwargs: compute_calls.append(kwargs.copy()) or {
            "selected_metrics": {
                "threshold": kwargs["selected_threshold"],
                "far": 0.0,
                "frr": 0.1,
                "gar": 0.9,
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
    monkeypatch.setattr(
        pad_model_evaluator,
        "evaluate_single_pad_dataset",
        lambda mode, scaler, classifier, selected_threshold, threshold_source, **kwargs: test_calls.append(
            (mode, selected_threshold, threshold_source, kwargs.get("output_stem"))
        ),
    )
    monkeypatch.setattr(
        pad_model_evaluator,
        "save_system_pad_threshold_config",
        lambda config_data, artifact_tag=None: saved_configs.append(config_data),
    )

    pad_model_evaluator.evaluate_all_pad_protocol(
        _DummyScaler(),
        _DummyClassifier(),
        {
            "custom_validation_ratio": 0.3,
            "custom_validation_seed": 7,
        },
        output_stem="unit_all",
        allow_config_update=True,
    )

    assert len(compute_calls) == 1
    assert compute_calls[0]["threshold_source"] == "all_combined_validation_target_apcer_selection"
    assert compute_calls[0]["selected_threshold"] == 0.55
    assert test_calls == [
        ("casia", 0.55, "all_combined_validation_target_apcer", "unit_all_casia"),
        ("replay", 0.55, "all_combined_validation_target_apcer", "unit_all_replay"),
        ("custom", 0.55, "all_combined_validation_target_apcer", "unit_all_custom"),
    ]
    assert saved_configs[0]["eval_mode"] == "all"
    assert saved_configs[0]["threshold_source"] == "all_combined_validation_target_apcer"
