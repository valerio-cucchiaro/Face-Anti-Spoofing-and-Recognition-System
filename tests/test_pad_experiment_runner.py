from src.antispoofing.training import pad_experiment_runner
import pytest


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


def test_pad_experiment_runner_default_full_does_not_promote_runtime_artifacts_for_custom(
    monkeypatch,
):
    calls = _patch_pad_runner_dependencies(monkeypatch)

    pad_experiment_runner.run(train_mode="custom", eval_mode="custom")

    assert calls["train_args"] == ("custom",)
    assert calls["train_kwargs"]["artifact_profile"] == "full"
    assert calls["train_kwargs"]["update_runtime_artifacts"] is False
    assert calls["eval_args"] == ("custom",)
    assert calls["eval_kwargs"]["artifact_profile"] == "full"


def test_pad_experiment_runner_canonical_full_promotes_runtime_artifacts(monkeypatch):
    calls = _patch_pad_runner_dependencies(monkeypatch)

    pad_experiment_runner.run(
        train_mode="all",
        eval_mode="all",
        save_system_threshold=True,
    )

    assert calls["train_args"] == ("all",)
    assert calls["train_kwargs"]["artifact_profile"] == "full"
    assert calls["train_kwargs"]["update_runtime_artifacts"] is True
    assert calls["eval_args"] == ("all",)
    assert calls["eval_kwargs"]["artifact_profile"] == "full"
    assert calls["eval_kwargs"]["save_system_threshold"] is True


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


def test_pad_experiment_runner_forwards_custom_validation_parameters(monkeypatch):
    calls = _patch_pad_runner_dependencies(monkeypatch)

    pad_experiment_runner.run(
        train_mode="custom",
        eval_mode="custom",
        custom_pad_validation_ratio=0.3,
        custom_pad_validation_seed=7,
    )

    assert calls["train_kwargs"]["custom_validation_ratio"] == 0.3
    assert calls["train_kwargs"]["custom_validation_seed"] == 7
    assert calls["eval_kwargs"]["pad_validation_target_apcer"] is None


def test_pad_experiment_runner_rejects_save_system_threshold_for_non_canonical_run(
    monkeypatch,
):
    _patch_pad_runner_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="allowed only for the canonical"):
        pad_experiment_runner.run(
            train_mode="custom",
            eval_mode="custom",
            save_system_threshold=True,
        )
