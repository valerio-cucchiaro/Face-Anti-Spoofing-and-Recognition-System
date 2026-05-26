"""
Module: recognition_dataset_splits

Description:
Dataset paths, experiment-mode resolution, logical split resolution,
and deterministic image sampling helpers for the recognition dataset
loading stack.
"""

from pathlib import Path

import numpy as np

from src.preprocessing.frame_preprocessing import DEFAULT_MIN_FACE_AREA


RECOGNITION_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "recognition"
CUSTOM_RAW_ROOT = RECOGNITION_ROOT / "custom_raw"
DATASET_SPLIT_DIRS = {
    "custom": {
        "gallery": CUSTOM_RAW_ROOT / "gallery",
        "probe_known": CUSTOM_RAW_ROOT / "probe_known",
        "probe_unknown": CUSTOM_RAW_ROOT / "probe_unknown",
    },
    "orl": {
        "gallery": RECOGNITION_ROOT / "orl" / "gallery",
        "validation_known": RECOGNITION_ROOT / "orl" / "validation_known",
        "probe_known": RECOGNITION_ROOT / "orl" / "probe_known",
        "validation_unknown": RECOGNITION_ROOT / "orl" / "validation_unknown",
        "probe_unknown": RECOGNITION_ROOT / "orl" / "probe_unknown",
    },
    "lfw": {
        "gallery": RECOGNITION_ROOT / "lfw" / "gallery",
        "validation_known": RECOGNITION_ROOT / "lfw" / "validation_known",
        "probe_known": RECOGNITION_ROOT / "lfw" / "probe_known",
        "validation_unknown": RECOGNITION_ROOT / "lfw" / "validation_unknown",
        "probe_unknown": RECOGNITION_ROOT / "lfw" / "probe_unknown",
    },
    "multipie": {
        "gallery": RECOGNITION_ROOT / "multipie" / "gallery",
        "validation_known_hr": RECOGNITION_ROOT / "multipie" / "validation_known_hr",
        "probe_known_hr": RECOGNITION_ROOT / "multipie" / "probe_known_hr",
        "validation_unknown_hr": RECOGNITION_ROOT / "multipie" / "validation_unknown_hr",
        "probe_unknown_hr": RECOGNITION_ROOT / "multipie" / "probe_unknown_hr",
        "validation_known_lr": RECOGNITION_ROOT / "multipie" / "validation_known_lr",
        "probe_known_lr": RECOGNITION_ROOT / "multipie" / "probe_known_lr",
        "validation_unknown_lr": RECOGNITION_ROOT / "multipie" / "validation_unknown_lr",
        "probe_unknown_lr": RECOGNITION_ROOT / "multipie" / "probe_unknown_lr",
    },
}
CUSTOM_PROBE_SPLITS = ("probe_known", "probe_unknown")
VALIDATED_OPEN_SET_SPLITS = {
    "validation": ("validation_known", "validation_unknown"),
    "validation_known": ("validation_known",),
    "validation_unknown": ("validation_unknown",),
    "probe": ("probe_known", "probe_unknown"),
    "probe_known": ("probe_known",),
    "probe_unknown": ("probe_unknown",),
}
MULTIPIE_PROBE_SPLITS = {
    "validation": (
        "validation_known_hr",
        "validation_unknown_hr",
        "validation_known_lr",
        "validation_unknown_lr",
    ),
    "validation_hr": ("validation_known_hr", "validation_unknown_hr"),
    "validation_lr": ("validation_known_lr", "validation_unknown_lr"),
    "probe": (
        "probe_known_hr",
        "probe_unknown_hr",
        "probe_known_lr",
        "probe_unknown_lr",
    ),
    "probe_hr": ("probe_known_hr", "probe_unknown_hr"),
    "probe_lr": ("probe_known_lr", "probe_unknown_lr"),
    "validation_known_hr": ("validation_known_hr",),
    "validation_unknown_hr": ("validation_unknown_hr",),
    "validation_known_lr": ("validation_known_lr",),
    "validation_unknown_lr": ("validation_unknown_lr",),
    "probe_known_hr": ("probe_known_hr",),
    "probe_unknown_hr": ("probe_unknown_hr",),
    "probe_known_lr": ("probe_known_lr",),
    "probe_unknown_lr": ("probe_unknown_lr",),
}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm"}
CUSTOM_RAW_MIN_FACE_AREA = DEFAULT_MIN_FACE_AREA
TRAIN_SAMPLING_SEED = 42
LFW_SAMPLE_RATIO = 1.0
DATASET_MODE_MAP = {
    "custom": ["custom"],
    "orl": ["orl"],
    "lfw": ["lfw"],
    "multipie": ["multipie"],
    "all": ["custom", "orl", "lfw", "multipie"],
}


def _assert_directory_exists(path, description):
    """
    Validate that one dataset directory exists and is a directory.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")

    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def resolve_dataset_names(mode):
    """
    Resolve a configured recognition mode into one or more dataset names.
    """
    if mode in DATASET_MODE_MAP:
        return DATASET_MODE_MAP[mode]

    raise ValueError(
        f"Unsupported recognition mode: {mode}. "
        "Use 'custom', 'orl', 'lfw', 'multipie', or 'all'."
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


def resolve_split_names(dataset_name, split):
    """
    Resolve one logical split into one or more concrete folder names.

    For the custom dataset, the ``*_internal`` split names are virtual
    subsets computed after loading the on-disk ``probe_known`` /
    ``probe_unknown`` folders. They are not separate directories.
    """
    if dataset_name == "custom":
        if split == "gallery":
            return ["gallery"]

        if split in {"probe", None}:
            return list(CUSTOM_PROBE_SPLITS)

        if split in CUSTOM_PROBE_SPLITS:
            return [split]

        if split in {
            "validation_known_internal",
            "probe_known_internal",
            "validation_unknown_internal",
            "probe_unknown_internal",
        }:
            return [split]

        raise ValueError(
            f"Unsupported split for {dataset_name} dataset: {split}. "
            "Use 'gallery', 'probe_known', 'probe_unknown', 'probe', "
            "'validation_known_internal', 'probe_known_internal', "
            "'validation_unknown_internal', or 'probe_unknown_internal'."
        )

    if dataset_name in {"orl", "lfw"}:
        if split == "gallery":
            return ["gallery"]

        if split in {None, "probe"}:
            return list(VALIDATED_OPEN_SET_SPLITS["probe"])

        if split in VALIDATED_OPEN_SET_SPLITS:
            return list(VALIDATED_OPEN_SET_SPLITS[split])

        raise ValueError(
            f"Unsupported split for {dataset_name} dataset: {split}. "
            "Use 'gallery', 'validation', 'validation_known', "
            "'validation_unknown', 'probe', 'probe_known', or 'probe_unknown'."
        )

    if dataset_name == "multipie":
        if split == "gallery":
            return ["gallery"]

        if split in MULTIPIE_PROBE_SPLITS:
            return list(MULTIPIE_PROBE_SPLITS[split])

        raise ValueError(
            f"Unsupported split for Multi-PIE dataset: {split}. "
            "Use 'gallery', 'validation', 'validation_hr', 'validation_lr', "
            "'probe', 'probe_hr', 'probe_lr', or the concrete known/unknown HR/LR splits."
        )

    if split in {"gallery", "probe"}:
        return [split]

    raise ValueError(
        f"Unsupported split for dataset '{dataset_name}': {split}. "
        "Use 'gallery' or 'probe'."
    )


def get_split_directories(mode, split):
    """
    Resolve the selected datasets to their gallery or probe directories.

    The custom ``*_internal`` splits intentionally reuse the same
    on-disk probe directories and are separated later by the protocol
    helper after loading and quality filtering.
    """
    dataset_dirs = []

    for dataset_name in resolve_dataset_names(mode):
        for concrete_split in resolve_split_names(dataset_name, split):
            directory_key = concrete_split
            if dataset_name == "custom":
                if concrete_split in {"validation_known_internal", "probe_known_internal"}:
                    directory_key = "probe_known"
                elif concrete_split in {"validation_unknown_internal", "probe_unknown_internal"}:
                    directory_key = "probe_unknown"

            split_dir = DATASET_SPLIT_DIRS[dataset_name][directory_key]
            _assert_directory_exists(
                split_dir,
                f"recognition {dataset_name} {directory_key} directory",
            )
            dataset_dirs.append((dataset_name, concrete_split, split_dir))

    return dataset_dirs
