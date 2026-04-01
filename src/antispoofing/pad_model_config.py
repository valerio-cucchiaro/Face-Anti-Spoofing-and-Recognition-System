"""
Module: pad_model_config

Description:
This module centralizes the configuration metadata associated with the
trained PAD model.

The PAD runtime, offline evaluators, and trainer must agree on the same
feature-extraction configuration. To avoid silent mismatches, the
selected PAD feature modes are stored next to the trained artifacts and
reapplied whenever the model is loaded.
"""

import json
from pathlib import Path

from src.antispoofing.pad_feature_extractor import configure_pad_feature_modes


# ----- CONSTANTS ----- #
MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"
PAD_CONFIG_PATH = MODELS_ROOT / "pad_config.json"


def get_pad_artifact_paths(artifact_profile="runtime"):
    """
    Resolve the PAD artifact paths associated with one profile.

    Profiles:
    - runtime: canonical artifacts used by webcam runtimes
    - full: snapshot produced by full PAD experiments
    - fast: snapshot produced by fast PAD experiments
    """

    if artifact_profile == "runtime":
        return {
            "model_path": MODELS_ROOT / "pad_svm.joblib",
            "scaler_path": MODELS_ROOT / "pad_scaler.joblib",
            "config_path": MODELS_ROOT / "pad_config.json",
        }

    if artifact_profile in {"full", "fast"}:
        suffix = artifact_profile
        return {
            "model_path": MODELS_ROOT / f"pad_svm_{suffix}.joblib",
            "scaler_path": MODELS_ROOT / f"pad_scaler_{suffix}.joblib",
            "config_path": MODELS_ROOT / f"pad_config_{suffix}.json",
        }

    raise ValueError(
        f"Unsupported PAD artifact profile: {artifact_profile}. "
        "Use 'runtime', 'full', or 'fast'."
    )


def infer_pad_feature_config_from_dim(feature_dim):
    """
    Infer a PAD feature configuration from the expected feature size.

    This fallback is useful for legacy PAD artifacts created before the
    configuration metadata file existed. The mapping covers the feature
    dimensions produced by the PAD variants documented in the project.
    """

    known_configs = {
        128: {
            "pad_debug_feature_mode": "baseline",
            "pad_lft_mode": "summary",
        },
        124: {
            "pad_debug_feature_mode": "baseline",
            "pad_lft_mode": "ggd_light",
        },
        130: {
            "pad_debug_feature_mode": "tgsd",
            "pad_lft_mode": "summary",
        },
        126: {
            "pad_debug_feature_mode": "tgsd",
            "pad_lft_mode": "ggd_light",
        },
        642: {
            "pad_debug_feature_mode": "full",
            "pad_lft_mode": "summary",
        },
        638: {
            "pad_debug_feature_mode": "full",
            "pad_lft_mode": "ggd_light",
        },
    }

    if feature_dim not in known_configs:
        raise ValueError(
            "Unable to infer PAD feature configuration from feature_dim="
            f"{feature_dim}. Save pad_config.json by retraining the PAD model."
        )

    inferred_config = known_configs[feature_dim].copy()
    inferred_config["feature_dim"] = int(feature_dim)
    inferred_config["config_source"] = "inferred_from_feature_dim"
    return inferred_config


def build_pad_model_config(debug_feature_mode, lft_mode, feature_dim):
    """
    Build the PAD configuration dictionary saved next to the model.
    """

    return {
        "pad_debug_feature_mode": debug_feature_mode,
        "pad_lft_mode": lft_mode,
        "feature_dim": int(feature_dim),
        "config_source": "saved_by_trainer",
    }


def save_pad_model_config(config, config_path=None):
    """
    Save the PAD model configuration to disk.
    """

    target_config_path = config_path or PAD_CONFIG_PATH
    target_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(target_config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)


def load_pad_model_config(expected_feature_dim=None, config_path=None):
    """
    Load the PAD model configuration from disk.

    If the metadata file does not exist yet, the configuration is
    inferred from the expected feature dimension of the loaded scaler or
    classifier. This keeps legacy PAD artifacts usable while making the
    mismatch explicit and recoverable.
    """

    target_config_path = config_path or PAD_CONFIG_PATH

    if target_config_path.exists():
        with open(target_config_path, "r", encoding="utf-8") as file:
            config = json.load(file)
    else:
        if expected_feature_dim is None:
            raise FileNotFoundError(
                "PAD configuration file not found and no expected feature "
                "dimension was provided to infer a legacy configuration."
            )
        config = infer_pad_feature_config_from_dim(expected_feature_dim)

    required_keys = {"pad_debug_feature_mode", "pad_lft_mode", "feature_dim"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(
            "PAD configuration is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    return config


def configure_pad_feature_extractor_from_config(config):
    """
    Apply the PAD feature modes described by a loaded model config.
    """

    configure_pad_feature_modes(
        debug_mode=config["pad_debug_feature_mode"],
        lft_mode=config["pad_lft_mode"],
    )


def get_expected_feature_dim_from_scaler(scaler):
    """
    Read the expected PAD feature dimension from a fitted scaler.
    """

    feature_dim = getattr(scaler, "n_features_in_", None)
    if feature_dim is None:
        raise ValueError(
            "Loaded PAD scaler does not expose n_features_in_. "
            "Unable to determine the expected feature dimension."
        )
    return int(feature_dim)
