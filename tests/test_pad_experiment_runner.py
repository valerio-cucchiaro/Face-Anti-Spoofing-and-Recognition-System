from src.antispoofing.training import pad_experiment_runner


def _patch_pad_runner_dependencies(monkeypatch):
    calls = {}

    def fake_train_pad_model(*args, **kwargs):
        calls["train_args"] = args
        calls["train_kwargs"] = kwargs

    def fake_evaluate_pad_model(*args, **kwargs):
        calls["eval_args"] = args
        calls["eval_kwargs"] = kwargs

    monkeypatch.setattr(pad_experiment_runner, "train_pad_model", fake_train_pad_model)
    monkeypatch.setattr(pad_experiment_runner, "evaluate_pad_model", fake_evaluate_pad_model)
    monkeypatch.setattr(
        pad_experiment_runner,
        "configure_pad_feature_modes",
        lambda **kwargs: None,
    )
    return calls


def test_pad_experiment_runner_default_full_promotes_runtime_artifacts(monkeypatch):
    calls = _patch_pad_runner_dependencies(monkeypatch)

    pad_experiment_runner.run(train_mode="custom", eval_mode="custom")

    assert calls["train_args"] == ("custom",)
    assert calls["train_kwargs"]["artifact_profile"] == "full"
    assert calls["train_kwargs"]["update_runtime_artifacts"] is True
    assert calls["eval_args"] == ("custom",)
    assert calls["eval_kwargs"]["artifact_profile"] == "full"


def test_pad_experiment_runner_fast_does_not_promote_runtime_artifacts(monkeypatch):
    calls = _patch_pad_runner_dependencies(monkeypatch)

    pad_experiment_runner.run(
        train_mode="custom",
        eval_mode="custom",
        experiment_speed="fast",
    )

    assert calls["train_args"] == ("custom",)
    assert calls["train_kwargs"]["artifact_profile"] == "fast"
    assert calls["train_kwargs"]["update_runtime_artifacts"] is False
    assert calls["eval_args"] == ("custom",)
    assert calls["eval_kwargs"]["artifact_profile"] == "fast"
