import json

import pytest

from src.antispoofing.pad_model_config import (
    build_pad_model_config,
    load_pad_model_config,
)


def test_load_pad_model_config_rejects_feature_dim_mismatch(tmp_path):
    config_path = tmp_path / "pad_config.json"
    config_path.write_text(
        json.dumps(build_pad_model_config("baseline", "ggd_light", 124)),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="PAD configuration mismatch"):
        load_pad_model_config(expected_feature_dim=126, config_path=config_path)


def test_load_pad_model_config_rejects_missing_required_keys(tmp_path):
    config_path = tmp_path / "pad_config.json"
    config_path.write_text(
        json.dumps(
            {
                "pad_debug_feature_mode": "baseline",
                "feature_dim": 124,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required keys"):
        load_pad_model_config(expected_feature_dim=124, config_path=config_path)


def test_load_pad_model_config_infers_legacy_config_from_feature_dim(tmp_path):
    config = load_pad_model_config(
        expected_feature_dim=124,
        config_path=tmp_path / "missing_pad_config.json",
    )

    assert config["config_source"] == "inferred_from_feature_dim"
    assert config["pad_debug_feature_mode"] == "baseline"
    assert config["pad_lft_mode"] == "ggd_light"
    assert config["feature_dim"] == 124


def test_build_pad_model_config_persists_custom_validation_metadata():
    config = build_pad_model_config(
        "baseline",
        "ggd_light",
        124,
        custom_validation_enabled=True,
        custom_validation_ratio=0.2,
        custom_validation_seed=42,
    )

    assert config["custom_validation_enabled"] is True
    assert config["custom_validation_ratio"] == 0.2
    assert config["custom_validation_seed"] == 42
