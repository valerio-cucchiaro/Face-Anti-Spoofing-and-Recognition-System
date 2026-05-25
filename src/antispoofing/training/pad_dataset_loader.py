"""
Module: pad_dataset_loader

Description:
Public entrypoint for the PAD dataset loading stack.

The implementation is split across:
- ``pad_dataset_splits`` for dataset paths and deterministic split helpers
- ``pad_dataset_preprocessing`` for image-to-feature dispatch helpers
- ``pad_dataset_builders`` for train/test/validation dataset builders

External callers should import the PAD dataset loading API from this
module. The other ``pad_dataset_*`` modules remain internal
organization helpers for the loader implementation.
"""

from src.antispoofing.training.pad_dataset_splits import (
    CASIA_ROOT,
    CASIA_SAMPLE_RATIO,
    CUSTOM_PAD_VALIDATION_RATIO,
    CUSTOM_PAD_VALIDATION_SEED,
    CUSTOM_TEST_LIVE_PATH,
    CUSTOM_TEST_SPOOF_PATH,
    CUSTOM_TRAIN_LIVE_PATH,
    CUSTOM_TRAIN_SPOOF_PATH,
    PAD_CUSTOM_RAW_ROOT,
    REPLAY_ROOT,
    REPLAY_SAMPLE_RATIO,
    REPLAY_TEST_LIVE_PATH,
    REPLAY_TEST_SPOOF_PATH,
    REPLAY_TRAIN_LIVE_PATH,
    REPLAY_TRAIN_SPOOF_PATH,
    REPLAY_VALIDATION_LIVE_PATH,
    REPLAY_VALIDATION_SPOOF_PATH,
    TEST_LIVE_PATH,
    TEST_SPOOF_PATH,
    TRAIN_LIVE_PATH,
    TRAIN_SAMPLING_SEED,
    TRAIN_SPOOF_PATH,
    VALIDATION_LIVE_PATH,
    VALIDATION_SPOOF_PATH,
    _CUSTOM_PAD_VALID_SPLIT_CACHE,
    collect_image_paths,
    resolve_custom_pad_train_validation_paths,
    sample_image_paths,
    split_image_paths_for_validation,
)
from src.antispoofing.training.pad_dataset_preprocessing import (
    LIVE_PHOTOMETRIC_AUGMENTATION_MODE,
    LIVE_PHOTOMETRIC_AUGMENTATION_SEED,
    PAD_IMAGE_PROGRESS_STEP,
    THRESHOLD_AREA,
    apply_live_photometric_augmentation,
    extract_pad_features_from_dataset_image,
    extract_pad_features_from_final_crop,
    extract_pad_features_from_raw_frame,
    get_valid_face,
)
from src.antispoofing.training.pad_dataset_builders import (
    load_casia_pad_training_dataset,
    load_combined_pad_training_dataset,
    load_combined_with_custom_pad_training_dataset,
    load_custom_pad_test_dataset,
    load_custom_pad_training_dataset,
    load_custom_pad_training_subset,
    load_custom_pad_validation_dataset,
    load_pad_split_dataset,
    load_replay_pad_training_dataset,
    load_pad_test_dataset_casia,
    load_pad_test_dataset_replay,
    load_pad_validation_dataset_casia,
    load_pad_validation_dataset_replay,
)


__all__ = [
    "CASIA_ROOT",
    "CASIA_SAMPLE_RATIO",
    "CUSTOM_PAD_VALIDATION_RATIO",
    "CUSTOM_PAD_VALIDATION_SEED",
    "CUSTOM_TEST_LIVE_PATH",
    "CUSTOM_TEST_SPOOF_PATH",
    "CUSTOM_TRAIN_LIVE_PATH",
    "CUSTOM_TRAIN_SPOOF_PATH",
    "LIVE_PHOTOMETRIC_AUGMENTATION_MODE",
    "LIVE_PHOTOMETRIC_AUGMENTATION_SEED",
    "PAD_CUSTOM_RAW_ROOT",
    "PAD_IMAGE_PROGRESS_STEP",
    "REPLAY_ROOT",
    "REPLAY_SAMPLE_RATIO",
    "REPLAY_TEST_LIVE_PATH",
    "REPLAY_TEST_SPOOF_PATH",
    "REPLAY_TRAIN_LIVE_PATH",
    "REPLAY_TRAIN_SPOOF_PATH",
    "REPLAY_VALIDATION_LIVE_PATH",
    "REPLAY_VALIDATION_SPOOF_PATH",
    "TEST_LIVE_PATH",
    "TEST_SPOOF_PATH",
    "THRESHOLD_AREA",
    "TRAIN_LIVE_PATH",
    "TRAIN_SAMPLING_SEED",
    "TRAIN_SPOOF_PATH",
    "VALIDATION_LIVE_PATH",
    "VALIDATION_SPOOF_PATH",
    "_CUSTOM_PAD_VALID_SPLIT_CACHE",
    "apply_live_photometric_augmentation",
    "collect_image_paths",
    "extract_pad_features_from_dataset_image",
    "extract_pad_features_from_final_crop",
    "extract_pad_features_from_raw_frame",
    "get_valid_face",
    "load_casia_pad_training_dataset",
    "load_combined_pad_training_dataset",
    "load_combined_with_custom_pad_training_dataset",
    "load_custom_pad_test_dataset",
    "load_custom_pad_training_dataset",
    "load_custom_pad_training_subset",
    "load_custom_pad_validation_dataset",
    "load_pad_split_dataset",
    "load_replay_pad_training_dataset",
    "load_pad_test_dataset_casia",
    "load_pad_test_dataset_replay",
    "load_pad_validation_dataset_casia",
    "load_pad_validation_dataset_replay",
    "resolve_custom_pad_train_validation_paths",
    "sample_image_paths",
    "split_image_paths_for_validation",
]
