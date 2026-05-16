import src.evaluation.recognition_model_evaluator as recognition_model_evaluator


def test_custom_recognition_dispatch_uses_internal_validation_and_probe_splits(monkeypatch):
    captured = {}

    def fake_evaluate_custom_open_set_dataset(*args, **kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        recognition_model_evaluator,
        "evaluate_custom_open_set_dataset",
        fake_evaluate_custom_open_set_dataset,
    )

    recognition_model_evaluator.evaluate_single_recognition_dataset(
        "custom",
        "custom",
        recognizer=object(),
        label_map={0: "alice"},
    )

    assert captured["validation_known_split"] == "validation_known_internal"
    assert captured["validation_unknown_split"] == "validation_unknown_internal"
    assert captured["final_known_split"] == "probe_known_internal"
    assert captured["final_unknown_split"] == "probe_unknown_internal"
    assert "dataset_name" not in captured
