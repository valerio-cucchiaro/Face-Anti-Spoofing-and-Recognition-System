"""
Module: pad_model_config

Description:
This module centralizes the configuration metadata associated with the
trained PAD model.

The PAD runtime, offline evaluators, and trainer must agree on the same
feature-extraction configuration. To avoid silent mismatches, the
selected PAD feature modes are stored next to the trained artifacts and
reapplied whenever the model is loaded. The canonical runtime artifact
set is distinct from the `full` and `fast` experiment snapshots.
"""

import json
import re
from pathlib import Path

from src.antispoofing.pad_feature_extractor import configure_pad_feature_modes


# ----- CONSTANTS ----- #
MODELS_ROOT = Path(__file__).resolve().parent.parent.parent / "models"
PAD_MODELS_ROOT = MODELS_ROOT / "pad"
PAD_CONFIG_PATH = PAD_MODELS_ROOT / "pad_config.json"
SYSTEM_PAD_THRESHOLD_CONFIG_PATH = PAD_MODELS_ROOT / "system_pad_threshold_config.json"


def normalize_pad_artifact_tag(artifact_tag):
    """
    Normalize and validate an optional PAD artifact tag.
    """

    if artifact_tag is None:
        return None

    normalized_tag = str(artifact_tag).strip()
    if not normalized_tag:
        raise ValueError("PAD artifact tag cannot be empty.")

    if not re.fullmatch(r"[A-Za-z0-9_-]+", normalized_tag):
        raise ValueError(
            "PAD artifact tag may contain only letters, numbers, '_' and '-'."
        )

    return normalized_tag


def _append_tag(stem, artifact_tag):
    """
    Append an optional artifact tag to a filename stem.
    """

    normalized_tag = normalize_pad_artifact_tag(artifact_tag)
    if normalized_tag is None:
        return stem
    return f"{stem}_{normalized_tag}"


def get_pad_artifact_paths(artifact_profile="runtime", artifact_tag=None):
    """
    Resolve the PAD artifact paths associated with one profile.

    Profiles:
    - runtime: canonical artifacts used by webcam runtimes
    - full: snapshot produced by full PAD experiments
    - fast: snapshot produced by fast PAD experiments

    The runtime profile is intentionally separate from the experiment
    profiles so only approved runs can promote artifacts to webcam
    inference.
    """

    if artifact_profile == "runtime":
        return {
            "model_path": PAD_MODELS_ROOT / f"{_append_tag('pad_svm', artifact_tag)}.joblib",
            "scaler_path": PAD_MODELS_ROOT / f"{_append_tag('pad_scaler', artifact_tag)}.joblib",
            "config_path": PAD_MODELS_ROOT / f"{_append_tag('pad_config', artifact_tag)}.json",
        }

    if artifact_profile in {"full", "fast"}:
        suffix = artifact_profile
        return {
            "model_path": PAD_MODELS_ROOT / f"{_append_tag(f'pad_svm_{suffix}', artifact_tag)}.joblib",
            "scaler_path": PAD_MODELS_ROOT / f"{_append_tag(f'pad_scaler_{suffix}', artifact_tag)}.joblib",
            "config_path": PAD_MODELS_ROOT / f"{_append_tag(f'pad_config_{suffix}', artifact_tag)}.json",
        }

    raise ValueError(
        f"Unsupported PAD artifact profile: {artifact_profile}. "
        "Use 'runtime', 'full', or 'fast'."
    )


def get_system_pad_threshold_config_path(artifact_tag=None):
    """
    Resolve the saved PAD system-threshold config path.
    """

    return PAD_MODELS_ROOT / f"{_append_tag('system_pad_threshold_config', artifact_tag)}.json"


def build_pad_model_config(
    debug_feature_mode,
    lft_mode,
    feature_dim,
    tgsd_tau=None,
    tgsd_top_percent=None,
    tgsd_gradient_operator=None,
    custom_validation_enabled=None,
    custom_validation_ratio=None,
    custom_validation_seed=None,
):
    """
    Build the PAD configuration dictionary saved next to the model.
    """

    config = {
        "pad_debug_feature_mode": debug_feature_mode,
        "pad_lft_mode": lft_mode,
        "feature_dim": int(feature_dim),
        "tgsd_tau": float(tgsd_tau) if tgsd_tau is not None else 170.0,
        "tgsd_top_percent": (
            float(tgsd_top_percent) if tgsd_top_percent is not None else 0.05
        ),
        "tgsd_gradient_operator": (
            tgsd_gradient_operator if tgsd_gradient_operator is not None else "sobel"
        ),
        "config_source": "saved_by_trainer",
    }

    if custom_validation_enabled is not None:
        config["custom_validation_enabled"] = bool(custom_validation_enabled)
    if custom_validation_ratio is not None:
        config["custom_validation_ratio"] = float(custom_validation_ratio)
    if custom_validation_seed is not None:
        config["custom_validation_seed"] = int(custom_validation_seed)

    return config


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
    """

    target_config_path = config_path or PAD_CONFIG_PATH

    if not target_config_path.exists():
        raise FileNotFoundError(
            "PAD configuration file not found. The PAD runtime and offline "
            "evaluators require an explicit saved config. Retrain or restore "
            "a coherent PAD artifact set under models/pad."
        )

    with open(target_config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    required_keys = {"pad_debug_feature_mode", "pad_lft_mode", "feature_dim"}
    missing_keys = required_keys - set(config.keys())
    if missing_keys:
        raise ValueError(
            "PAD configuration is missing required keys: "
            f"{sorted(missing_keys)}"
        )

    config.setdefault("tgsd_tau", 170.0)
    config.setdefault("tgsd_top_percent", 0.05)
    config.setdefault("tgsd_gradient_operator", "sobel")

    config_feature_dim = int(config["feature_dim"])
    if expected_feature_dim is not None and config_feature_dim != int(expected_feature_dim):
        raise ValueError(
            "PAD configuration mismatch: config declares "
            f"{config['pad_debug_feature_mode']} + {config['pad_lft_mode']} "
            f"with feature_dim={config_feature_dim}, but the loaded artifact "
            f"expects feature_dim={int(expected_feature_dim)}. Realign the "
            "saved config or promote a coherent artifact set."
        )

    return config


def configure_pad_feature_extractor_from_config(config):
    """
    Apply the PAD feature modes described by a loaded model config.
    """

    configure_pad_feature_modes(
        debug_mode=config["pad_debug_feature_mode"],
        lft_mode=config["pad_lft_mode"],
        tgsd_tau=config.get("tgsd_tau"),
        tgsd_top_percent=config.get("tgsd_top_percent"),
        tgsd_gradient_operator=config.get("tgsd_gradient_operator"),
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
