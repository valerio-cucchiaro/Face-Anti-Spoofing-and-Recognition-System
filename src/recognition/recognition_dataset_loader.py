"""
Module: recognition_dataset_loader

Description:
Public entrypoint for the recognition dataset loading stack.

The implementation is split across:
- ``recognition_dataset_splits`` for dataset paths, split resolution,
  and deterministic sampling helpers
- ``recognition_dataset_protocol`` for identity naming and custom
  internal validation/probe split semantics
- ``recognition_dataset_preprocessing`` for sample-level loading,
  preprocessing, pose collection, and quality filtering
- ``recognition_dataset_builders`` for gallery/probe dataset builders

External callers should import the recognition dataset loading API from
this module. The other ``recognition_dataset_*`` modules remain
internal organization helpers for the loader implementation.
"""

from src.detection.face_detector import load_mediapipe_detector
from src.detection.face_landmarks import load_mediapipe_face_mesh
from src.recognition.recognition_dataset_builders import (
    load_gallery_dataset_impl,
    load_probe_dataset_impl,
)
from src.recognition.recognition_dataset_preprocessing import (
    load_images_from_identity_dir,
)
from src.recognition.recognition_dataset_protocol import (
    CUSTOM_OPEN_SET_VALIDATION_RATIO,
    CUSTOM_OPEN_SET_VALIDATION_SEED,
    _assert_identity_subset_consistency,
    build_identity_name,
    normalize_identity_name,
    should_namespace_identities,
    split_custom_open_set_identity_images,
)
from src.recognition.recognition_dataset_splits import (
    CUSTOM_PROBE_SPLITS,
    CUSTOM_RAW_MIN_FACE_AREA,
    CUSTOM_RAW_ROOT,
    DATASET_MODE_MAP,
    DATASET_SPLIT_DIRS,
    LFW_SAMPLE_RATIO,
    MULTIPIE_PROBE_SPLITS,
    RECOGNITION_ROOT,
    TRAIN_SAMPLING_SEED,
    VALIDATED_OPEN_SET_SPLITS,
    VALID_IMAGE_EXTENSIONS,
    get_split_directories,
    resolve_dataset_names,
    resolve_split_names,
    sample_image_paths,
)


def load_gallery_dataset(
    mode,
    collect_pose_metrics=False,
    lfw_sample_ratio=LFW_SAMPLE_RATIO,
    lfw_sample_seed=TRAIN_SAMPLING_SEED,
    quality_filter_mode="report_only",
    debug_quality=False,
    quality_output_stem=None,
):
    """
    Load one or more gallery datasets for recognition training.
    """
    namespace_identities = should_namespace_identities(mode)
    return load_gallery_dataset_impl(
        mode,
        namespace_identities,
        collect_pose_metrics,
        lfw_sample_ratio,
        lfw_sample_seed,
        quality_filter_mode,
        debug_quality,
        quality_output_stem,
        get_split_directories_fn=get_split_directories,
        resolve_dataset_names_fn=resolve_dataset_names,
        build_identity_name_fn=build_identity_name,
        normalize_identity_name_fn=normalize_identity_name,
        load_images_from_identity_dir_fn=load_images_from_identity_dir,
        sample_image_paths_fn=sample_image_paths,
        detector_loader=load_mediapipe_detector,
        face_mesh_loader=load_mediapipe_face_mesh,
    )


def load_probe_dataset(
    mode,
    split="probe",
    namespace_identities=False,
    collect_pose_metrics=False,
    lfw_sample_ratio=LFW_SAMPLE_RATIO,
    lfw_sample_seed=TRAIN_SAMPLING_SEED,
    quality_filter_mode="report_only",
    debug_quality=False,
    quality_output_stem=None,
):
    """
    Load one or more probe datasets for recognition evaluation.

    For public open-set datasets, ``validation`` is used for threshold
    selection and ``probe`` is used for final metrics. Custom open-set
    runs can also request internal holdout splits derived from the
    original custom probe folders after quality filtering.
    """
    return load_probe_dataset_impl(
        mode,
        split,
        namespace_identities,
        collect_pose_metrics,
        lfw_sample_ratio,
        lfw_sample_seed,
        quality_filter_mode,
        debug_quality,
        quality_output_stem,
        get_split_directories_fn=get_split_directories,
        resolve_dataset_names_fn=resolve_dataset_names,
        build_identity_name_fn=build_identity_name,
        split_custom_open_set_identity_images_fn=split_custom_open_set_identity_images,
        assert_identity_subset_consistency_fn=_assert_identity_subset_consistency,
        load_images_from_identity_dir_fn=load_images_from_identity_dir,
        sample_image_paths_fn=sample_image_paths,
        detector_loader=load_mediapipe_detector,
        face_mesh_loader=load_mediapipe_face_mesh,
    )


__all__ = [
    "CUSTOM_OPEN_SET_VALIDATION_RATIO",
    "CUSTOM_OPEN_SET_VALIDATION_SEED",
    "CUSTOM_PROBE_SPLITS",
    "CUSTOM_RAW_MIN_FACE_AREA",
    "CUSTOM_RAW_ROOT",
    "DATASET_MODE_MAP",
    "DATASET_SPLIT_DIRS",
    "LFW_SAMPLE_RATIO",
    "MULTIPIE_PROBE_SPLITS",
    "RECOGNITION_ROOT",
    "TRAIN_SAMPLING_SEED",
    "VALIDATED_OPEN_SET_SPLITS",
    "VALID_IMAGE_EXTENSIONS",
    "_assert_identity_subset_consistency",
    "build_identity_name",
    "get_split_directories",
    "load_gallery_dataset",
    "load_images_from_identity_dir",
    "load_probe_dataset",
    "normalize_identity_name",
    "resolve_dataset_names",
    "resolve_split_names",
    "sample_image_paths",
    "should_namespace_identities",
    "split_custom_open_set_identity_images",
]
