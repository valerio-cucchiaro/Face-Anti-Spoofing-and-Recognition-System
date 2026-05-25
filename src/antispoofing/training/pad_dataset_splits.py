"""
Module: pad_dataset_splits

Description:
Dataset paths, deterministic sampling helpers, and custom PAD internal
train/validation split resolution for the PAD training pipeline.
"""

from pathlib import Path

import numpy as np


# ----- CASIA DATASET ----- #
CASIA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad" / "casia_pad"

TRAIN_LIVE_PATH = CASIA_ROOT / "train" / "live"
TRAIN_SPOOF_PATH = CASIA_ROOT / "train" / "spoof"
VALIDATION_LIVE_PATH = CASIA_ROOT / "validation" / "live"
VALIDATION_SPOOF_PATH = CASIA_ROOT / "validation" / "spoof"
TEST_LIVE_PATH = CASIA_ROOT / "test" / "live"
TEST_SPOOF_PATH = CASIA_ROOT / "test" / "spoof"

# ----- REPLAY ATTACK DATASET ----- #
REPLAY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad" / "replay_attack_pad"

REPLAY_TRAIN_LIVE_PATH = REPLAY_ROOT / "train" / "live"
REPLAY_TRAIN_SPOOF_PATH = REPLAY_ROOT / "train" / "spoof"
REPLAY_VALIDATION_LIVE_PATH = REPLAY_ROOT / "validation" / "live"
REPLAY_VALIDATION_SPOOF_PATH = REPLAY_ROOT / "validation" / "spoof"
REPLAY_TEST_LIVE_PATH = REPLAY_ROOT / "test" / "live"
REPLAY_TEST_SPOOF_PATH = REPLAY_ROOT / "test" / "spoof"

# ----- CUSTOM PAD DATASET ----- #
PAD_CUSTOM_RAW_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad" / "pad_custom_raw"

CUSTOM_TRAIN_LIVE_PATH = PAD_CUSTOM_RAW_ROOT / "train" / "live"
CUSTOM_TRAIN_SPOOF_PATH = PAD_CUSTOM_RAW_ROOT / "train" / "spoof"
CUSTOM_TEST_LIVE_PATH = PAD_CUSTOM_RAW_ROOT / "test" / "live"
CUSTOM_TEST_SPOOF_PATH = PAD_CUSTOM_RAW_ROOT / "test" / "spoof"

# ----- SAMPLING CONSTANTS ----- #
TRAIN_SAMPLING_SEED = 42
CASIA_SAMPLE_RATIO = 1.0
REPLAY_SAMPLE_RATIO = 1.0
CUSTOM_PAD_VALIDATION_RATIO = 0.2
CUSTOM_PAD_VALIDATION_SEED = 42
_CUSTOM_PAD_VALID_SPLIT_CACHE = {}


def _assert_input_directory(path, description):
    """
    Validate that one PAD input directory exists and is a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _validate_custom_pad_split_paths(
    split_paths,
    live_paths,
    spoof_paths,
):
    """
    Verify that the custom PAD internal split is disjoint and complete.
    """
    train_live_set = set(split_paths["train_live_paths"])
    validation_live_set = set(split_paths["validation_live_paths"])
    train_spoof_set = set(split_paths["train_spoof_paths"])
    validation_spoof_set = set(split_paths["validation_spoof_paths"])

    live_pool = set(live_paths)
    spoof_pool = set(spoof_paths)

    if not live_pool or not spoof_pool:
        raise RuntimeError(
            "Custom PAD filtering produced an empty class before internal validation split."
        )

    if train_live_set & validation_live_set:
        raise RuntimeError("Custom PAD live train/validation subsets overlap.")

    if train_spoof_set & validation_spoof_set:
        raise RuntimeError("Custom PAD spoof train/validation subsets overlap.")

    if train_live_set | validation_live_set != live_pool:
        raise RuntimeError(
            "Custom PAD live train/validation subsets do not reconstruct the filtered live pool."
        )

    if train_spoof_set | validation_spoof_set != spoof_pool:
        raise RuntimeError(
            "Custom PAD spoof train/validation subsets do not reconstruct the filtered spoof pool."
        )


def sample_image_paths(paths, ratio=1.0, seed=None):
    """
    Deterministically downsample a list of image paths.
    """
    if ratio >= 1.0 or len(paths) <= 1:
        return list(paths)

    if ratio <= 0.0:
        return []

    n_samples = max(1, int(len(paths) * ratio))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(paths), size=n_samples, replace=False)
    indices = np.sort(indices)

    return [paths[i] for i in indices]


def collect_image_paths(path_or_paths):
    """
    Collect supported image paths from one directory or from a path list.
    """
    valid_extensions = {".png", ".jpg", ".jpeg"}

    if isinstance(path_or_paths, (str, Path)):
        _assert_input_directory(
            Path(path_or_paths),
            "PAD dataset split directory",
        )
        return [
            img_path for img_path in sorted(Path(path_or_paths).iterdir())
            if img_path.suffix.lower() in valid_extensions
        ]

    return [
        Path(img_path) for img_path in path_or_paths
        if Path(img_path).suffix.lower() in valid_extensions
    ]


def split_image_paths_for_validation(image_paths, validation_ratio, seed):
    """
    Deterministically split one class into train and validation subsets.
    """
    if not 0.0 < float(validation_ratio) < 1.0:
        raise ValueError(
            "custom PAD validation_ratio must be between 0 and 1. "
            f"Received: {validation_ratio}"
        )

    paths = list(image_paths)
    if len(paths) <= 1:
        return paths, []

    validation_count = int(round(len(paths) * float(validation_ratio)))
    validation_count = max(1, min(len(paths) - 1, validation_count))

    rng = np.random.default_rng(seed)
    validation_indices = np.sort(
        rng.choice(len(paths), size=validation_count, replace=False)
    )
    validation_index_set = set(int(index) for index in validation_indices)

    train_paths = [
        path for index, path in enumerate(paths)
        if index not in validation_index_set
    ]
    validation_paths = [
        path for index, path in enumerate(paths)
        if index in validation_index_set
    ]
    return train_paths, validation_paths


def resolve_custom_pad_train_validation_paths(
    validation_ratio=CUSTOM_PAD_VALIDATION_RATIO,
    validation_seed=CUSTOM_PAD_VALIDATION_SEED,
):
    """
    Split the custom PAD train data into deterministic train and
    validation subsets after keeping only valid samples.

    This keeps the custom PAD protocol aligned with the custom
    recognition protocol: the internal holdout is computed on the
    samples that survive PAD loading and quality filtering, not on the
    raw file lists alone.
    """
    cache_key = (
        str(CUSTOM_TRAIN_LIVE_PATH),
        str(CUSTOM_TRAIN_SPOOF_PATH),
        float(validation_ratio),
        int(validation_seed),
    )
    cached_split = _CUSTOM_PAD_VALID_SPLIT_CACHE.get(cache_key)
    if cached_split is not None:
        return {
            split_name: list(split_paths)
            for split_name, split_paths in cached_split.items()
        }

    from src.antispoofing.training.pad_dataset_builders import load_pad_split_dataset

    _, labels, sample_paths = load_pad_split_dataset(
        CUSTOM_TRAIN_LIVE_PATH,
        CUSTOM_TRAIN_SPOOF_PATH,
        balance=False,
        dataset_type="final_pad_crop",
        return_image_paths=True,
        collect_pose_metrics=False,
        return_pose_summary=False,
        augment_live_photometric=False,
    )

    live_paths = [
        Path(sample_path)
        for sample_path, label in zip(sample_paths, labels)
        if int(label) == 1
    ]
    spoof_paths = [
        Path(sample_path)
        for sample_path, label in zip(sample_paths, labels)
        if int(label) == 0
    ]

    train_live_paths, validation_live_paths = split_image_paths_for_validation(
        live_paths,
        validation_ratio,
        validation_seed,
    )
    train_spoof_paths, validation_spoof_paths = split_image_paths_for_validation(
        spoof_paths,
        validation_ratio,
        validation_seed + 1,
    )

    split_paths = {
        "train_live_paths": train_live_paths,
        "train_spoof_paths": train_spoof_paths,
        "validation_live_paths": validation_live_paths,
        "validation_spoof_paths": validation_spoof_paths,
    }
    _validate_custom_pad_split_paths(split_paths, live_paths, spoof_paths)
    _CUSTOM_PAD_VALID_SPLIT_CACHE[cache_key] = {
        split_name: list(paths)
        for split_name, paths in split_paths.items()
    }
    return {
        split_name: list(paths)
        for split_name, paths in split_paths.items()
    }
