"""
Module: recognition_dataset_loader

Description:
This module provides the dataset loading utilities used by the
recognition training and evaluation pipeline.

It supports the custom, ORL, LFW, and Multi-PIE recognition datasets and can
load them individually or combine them when required by an experiment.
Gallery images are used for training, while open-set evaluation data is
separated into validation and test splits for threshold selection and
final metrics.

The supported experiment modes are intentionally explicit:
- `custom`, `orl`, `lfw`, `multipie`: single-dataset runs
- `custom_orl`: realistic mixed training setup without LFW
- `all`: full mixed experiment including LFW and Multi-PIE, mainly useful for broad
  comparisons and stress-style analysis
"""

import cv2
from pathlib import Path
import numpy as np
import time

from src.detection.face_detector import load_mediapipe_detector
from src.detection.face_landmarks import load_mediapipe_face_mesh
from src.preprocessing.face_pose import compute_pose_metrics_from_image
from src.preprocessing.pose_statistics import (
    create_pose_summary,
    format_pose_summary,
    update_pose_summary,
)
from src.preprocessing.frame_preprocessing import get_valid_face
from src.preprocessing.face_preprocessing import (
    preprocess_face_for_recognition,
    preprocess_loaded_face_for_recognition,
)


# ----- CONSTANTS ----- #
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
        "test_known": RECOGNITION_ROOT / "orl" / "test_known",
        "validation_unknown": RECOGNITION_ROOT / "orl" / "validation_unknown",
        "test_unknown": RECOGNITION_ROOT / "orl" / "test_unknown",
        "probe_known": RECOGNITION_ROOT / "orl" / "probe_known",
        "probe_unknown": RECOGNITION_ROOT / "orl" / "probe_unknown",
    },
    "lfw": {
        "gallery": RECOGNITION_ROOT / "lfw" / "gallery",
        "validation_known": RECOGNITION_ROOT / "lfw" / "validation_known",
        "test_known": RECOGNITION_ROOT / "lfw" / "test_known",
        "validation_unknown": RECOGNITION_ROOT / "lfw" / "validation_unknown",
        "test_unknown": RECOGNITION_ROOT / "lfw" / "test_unknown",
        "probe_known": RECOGNITION_ROOT / "lfw" / "probe_known",
        "probe_unknown": RECOGNITION_ROOT / "lfw" / "probe_unknown",
    },
    "multipie": {
        "gallery": RECOGNITION_ROOT / "multipie" / "gallery",
        "validation_known_hr": RECOGNITION_ROOT / "multipie" / "validation_known_hr",
        "test_known_hr": RECOGNITION_ROOT / "multipie" / "test_known_hr",
        "validation_unknown_hr": RECOGNITION_ROOT / "multipie" / "validation_unknown_hr",
        "test_unknown_hr": RECOGNITION_ROOT / "multipie" / "test_unknown_hr",
        "validation_known_lr": RECOGNITION_ROOT / "multipie" / "validation_known_lr",
        "test_known_lr": RECOGNITION_ROOT / "multipie" / "test_known_lr",
        "validation_unknown_lr": RECOGNITION_ROOT / "multipie" / "validation_unknown_lr",
        "test_unknown_lr": RECOGNITION_ROOT / "multipie" / "test_unknown_lr",
        "probe_known_hr": RECOGNITION_ROOT / "multipie" / "probe_known_hr",
        "probe_known_lr": RECOGNITION_ROOT / "multipie" / "probe_known_lr",
        "probe_unknown_hr": RECOGNITION_ROOT / "multipie" / "probe_unknown_hr",
        "probe_unknown_lr": RECOGNITION_ROOT / "multipie" / "probe_unknown_lr",
    },
}
CUSTOM_PROBE_SPLITS = ("probe_known", "probe_unknown")
VALIDATED_OPEN_SET_SPLITS = {
    "validation": ("validation_known", "validation_unknown"),
    "test": ("test_known", "test_unknown"),
    "validation_known": ("validation_known",),
    "test_known": ("test_known",),
    "validation_unknown": ("validation_unknown",),
    "test_unknown": ("test_unknown",),
    "probe": ("test_known", "test_unknown"),
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
    "test": (
        "test_known_hr",
        "test_unknown_hr",
        "test_known_lr",
        "test_unknown_lr",
    ),
    "validation_hr": ("validation_known_hr", "validation_unknown_hr"),
    "test_hr": ("test_known_hr", "test_unknown_hr"),
    "validation_lr": ("validation_known_lr", "validation_unknown_lr"),
    "test_lr": ("test_known_lr", "test_unknown_lr"),
    "probe": (
        "test_known_hr",
        "test_unknown_hr",
        "test_known_lr",
        "test_unknown_lr",
    ),
    "probe_hr": ("test_known_hr", "test_unknown_hr"),
    "probe_lr": ("test_known_lr", "test_unknown_lr"),
    "validation_known_hr": ("validation_known_hr",),
    "test_known_hr": ("test_known_hr",),
    "validation_unknown_hr": ("validation_unknown_hr",),
    "test_unknown_hr": ("test_unknown_hr",),
    "validation_known_lr": ("validation_known_lr",),
    "test_known_lr": ("test_known_lr",),
    "validation_unknown_lr": ("validation_unknown_lr",),
    "test_unknown_lr": ("test_unknown_lr",),
    "probe_known_hr": ("probe_known_hr",),
    "probe_known_lr": ("probe_known_lr",),
    "probe_unknown_hr": ("probe_unknown_hr",),
    "probe_unknown_lr": ("probe_unknown_lr",),
}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm"}
IDENTITY_PROGRESS_STEP = 50
CUSTOM_RAW_MIN_FACE_AREA = 8000
TRAIN_SAMPLING_SEED = 42
LFW_SAMPLE_RATIO = 1.0
DATASET_MODE_MAP = {
    "custom": ["custom"],
    "orl": ["orl"],
    "lfw": ["lfw"],
    "multipie": ["multipie"],
    "custom_orl": ["custom", "orl"],
    "all": ["custom", "orl", "lfw", "multipie"],
}


def _compute_pose_metrics_for_loaded_image(
    dataset_name,
    image,
    face_mesh,
    face_rect=None,
):
    """
    Compute pose metrics for one loaded recognition sample.
    """

    if image is None or face_mesh is None:
        return None

    if dataset_name == "custom" and face_rect is not None:
        x, y, w, h = face_rect
        if image.ndim == 3:
            sample_image = image[y:y + h, x:x + w]
        else:
            sample_image = image[y:y + h, x:x + w]
    else:
        sample_image = image

    return compute_pose_metrics_from_image(sample_image, face_mesh)


def resolve_dataset_names(mode):
    """
    Resolve a configured recognition mode into one or more dataset names.

    The same mode vocabulary is used for both training and evaluation so
    experiment definitions stay simple. Combined modes are expanded into
    the ordered list of datasets they include.
    """
    if mode in DATASET_MODE_MAP:
        return DATASET_MODE_MAP[mode]

    raise ValueError(
        f"Unsupported recognition mode: {mode}. "
        "Use 'custom', 'orl', 'lfw', 'multipie', 'custom_orl', or 'all'."
    )


def should_namespace_identities(mode):
    """
    Decide whether identity names should be prefixed with the dataset name.

    Namespacing is enabled for combined modes so identities coming from
    different datasets cannot collide in the shared label space.
    """
    return mode in {"custom_orl", "all"}


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


def build_identity_name(dataset_name, person_name, namespace_identities=False):
    """
    Build the identity label used in the current experiment.
    """
    if namespace_identities:
        return f"{dataset_name}:{person_name}"

    return person_name


def normalize_identity_name(identity_name):
    """
    Normalize an identity label to its canonical comparison form.

    Dataset prefixes such as ``custom:`` are removed so identities can
    be compared consistently across training and evaluation.
    """
    if identity_name is None:
        return None

    return identity_name.split(":", 1)[-1].strip().lower()


def resolve_split_names(dataset_name, split):
    """
    Resolve one logical split into one or more concrete folder names.

    The custom dataset distinguishes known and unknown evaluation images
    explicitly. A generic ``probe`` request for custom data expands to
    both probe folders so older call sites keep working while the caller
    can migrate to the explicit split names.
    """
    if dataset_name == "custom":
        if split == "gallery":
            return ["gallery"]

        if split in {"probe", None}:
            return list(CUSTOM_PROBE_SPLITS)

        if split in CUSTOM_PROBE_SPLITS:
            return [split]

        raise ValueError(
            f"Unsupported split for {dataset_name} dataset: {split}. "
            "Use 'gallery', 'probe_known', 'probe_unknown', or 'probe'."
        )

    if dataset_name in {"orl", "lfw"}:
        if split == "gallery":
            return ["gallery"]

        if split in {None, "probe"}:
            return list(VALIDATED_OPEN_SET_SPLITS["probe"])

        if split in VALIDATED_OPEN_SET_SPLITS:
            concrete_splits = list(VALIDATED_OPEN_SET_SPLITS[split])
            missing_new_splits = [
                concrete_split
                for concrete_split in concrete_splits
                if not DATASET_SPLIT_DIRS[dataset_name][concrete_split].exists()
            ]
            if missing_new_splits and split in {"probe_known", "probe_unknown"}:
                return [split]
            return concrete_splits

        raise ValueError(
            f"Unsupported split for {dataset_name} dataset: {split}. "
            "Use 'gallery', 'validation', 'test', 'validation_known', "
            "'test_known', 'validation_unknown', 'test_unknown', or 'probe'."
        )

    if dataset_name == "multipie":
        if split == "gallery":
            return ["gallery"]

        if split in MULTIPIE_PROBE_SPLITS:
            return list(MULTIPIE_PROBE_SPLITS[split])

        raise ValueError(
            f"Unsupported split for Multi-PIE dataset: {split}. "
            "Use 'gallery', 'validation', 'test', 'validation_hr', 'test_hr', "
            "'validation_lr', 'test_lr', or the concrete known/unknown HR/LR splits."
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

    Open-set datasets support explicit validation/test splits in addition
    to the generic ``probe`` compatibility alias.
    """
    dataset_dirs = []

    for dataset_name in resolve_dataset_names(mode):
        for concrete_split in resolve_split_names(dataset_name, split):
            split_dir = DATASET_SPLIT_DIRS[dataset_name][concrete_split]

            if not split_dir.exists():
                raise FileNotFoundError(
                    f"Recognition {concrete_split} directory not found: {split_dir}"
                )

            dataset_dirs.append((dataset_name, concrete_split, split_dir))

    return dataset_dirs


def load_images_from_identity_dir(
    identity_dir,
    dataset_name,
    detector=None,
    collect_pose_metrics=False,
    pose_summary=None,
    face_mesh=None,
    sample_ratio=1.0,
    sample_seed=None,
):
    """
    Load all valid images from one identity directory.
    """
    images = []
    image_paths = [
        image_path
        for image_path in sorted(identity_dir.iterdir(), key=lambda path: path.name)
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if dataset_name == "lfw":
        image_paths = sample_image_paths(
            image_paths,
            ratio=sample_ratio,
            seed=sample_seed,
        )

    for image_path in image_paths:

        if dataset_name == "custom":
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        if dataset_name == "custom":
            if detector is None:
                raise ValueError("A face detector is required to load custom raw images.")

            face_rect, gray_frame, message = get_valid_face(
                image,
                detector,
                CUSTOM_RAW_MIN_FACE_AREA,
            )

            if message != "Face detected" or face_rect is None or gray_frame is None:
                continue

            x, y, w, h = face_rect
            preprocessed_image = preprocess_face_for_recognition(
                gray_frame,
                x,
                y,
                w,
                h,
            )
            if collect_pose_metrics:
                pose_metrics = _compute_pose_metrics_for_loaded_image(
                    dataset_name,
                    image,
                    face_mesh,
                    face_rect=face_rect,
                )
                update_pose_summary(pose_summary, pose_metrics)
        else:
            preprocessed_image = preprocess_loaded_face_for_recognition(image)
            if collect_pose_metrics:
                pose_metrics = _compute_pose_metrics_for_loaded_image(
                    dataset_name,
                    image,
                    face_mesh,
                )
                update_pose_summary(pose_summary, pose_metrics)
        if preprocessed_image is None:
            continue

        images.append((image_path, preprocessed_image))

    return images


def load_gallery_dataset(
    mode,
    collect_pose_metrics=False,
    lfw_sample_ratio=LFW_SAMPLE_RATIO,
    lfw_sample_seed=TRAIN_SAMPLING_SEED,
):
    """
    Load one or more gallery datasets for recognition training.
    """
    namespace_identities = should_namespace_identities(mode)
    faces = []
    labels = []
    label_map = {}
    dataset_counts = {}
    pose_summary = create_pose_summary() if collect_pose_metrics else None
    custom_detector = (
        load_mediapipe_detector() if "custom" in resolve_dataset_names(mode) else None
    )
    face_mesh = load_mediapipe_face_mesh() if collect_pose_metrics else None

    try:
        for dataset_name, _split_name, gallery_dir in get_split_directories(mode, "gallery"):
            start_time = time.perf_counter()
            dataset_counts[dataset_name] = 0
            identity_count = 0
            person_dirs = sorted(
                [path for path in gallery_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
            )

            print(f"Loading gallery dataset: {dataset_name}")
            print(f"Found {len(person_dirs)} gallery identities")

            for person_dir in person_dirs:
                identity_name = build_identity_name(
                    dataset_name,
                    person_dir.name,
                    namespace_identities=namespace_identities,
                )
                normalized_identity_name = normalize_identity_name(identity_name)

                identity_images = load_images_from_identity_dir(
                    person_dir,
                    dataset_name=dataset_name,
                    detector=custom_detector,
                    collect_pose_metrics=collect_pose_metrics,
                    pose_summary=pose_summary,
                    face_mesh=face_mesh,
                    sample_ratio=lfw_sample_ratio,
                    sample_seed=lfw_sample_seed,
                )

                if not identity_images:
                    continue

                current_label = len(label_map)
                label_map[current_label] = normalized_identity_name
                identity_count += 1

                if identity_count % IDENTITY_PROGRESS_STEP == 0:
                    print(f"Processed {identity_count}/{len(person_dirs)} gallery identities")

                for _, image in identity_images:
                    faces.append(image)
                    labels.append(current_label)
                    dataset_counts[dataset_name] += 1

            elapsed = time.perf_counter() - start_time
            print(
                f"Loaded {dataset_counts[dataset_name]} gallery images "
                f"from {identity_count} identities in {elapsed:.2f}s"
            )
            if dataset_name == "lfw" and lfw_sample_ratio < 1.0:
                print(
                    f"LFW gallery sampling: ratio={lfw_sample_ratio:.3f} "
                    f"seed={lfw_sample_seed}"
                )
    finally:
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()

    return faces, np.array(labels), label_map, dataset_counts, pose_summary


def load_probe_dataset(
    mode,
    split="probe",
    namespace_identities=False,
    collect_pose_metrics=False,
    lfw_sample_ratio=LFW_SAMPLE_RATIO,
    lfw_sample_seed=TRAIN_SAMPLING_SEED,
):
    """
    Load one or more probe datasets for recognition evaluation.

    For public open-set datasets, ``validation`` is used for threshold
    selection and ``test`` is used for final metrics. Passing one of the
    explicit split names loads only that branch.
    """
    probe_records = []
    dataset_counts = {}
    pose_summary = create_pose_summary() if collect_pose_metrics else None
    custom_detector = (
        load_mediapipe_detector() if "custom" in resolve_dataset_names(mode) else None
    )
    face_mesh = load_mediapipe_face_mesh() if collect_pose_metrics else None

    try:
        for dataset_name, split_name, probe_dir in get_split_directories(mode, split):
            start_time = time.perf_counter()
            split_key = f"{dataset_name}/{split_name}"
            dataset_counts[split_key] = 0
            identity_count = 0
            person_dirs = sorted(
                [path for path in probe_dir.iterdir() if path.is_dir()],
                key=lambda path: path.name,
            )

            print(f"Loading recognition evaluation split: {dataset_name} ({split_name})")
            print(f"Found {len(person_dirs)} identities in split")

            for person_dir in person_dirs:
                identity_name = build_identity_name(
                    dataset_name,
                    person_dir.name,
                    namespace_identities=namespace_identities,
                )

                identity_images = load_images_from_identity_dir(
                    person_dir,
                    dataset_name=dataset_name,
                    detector=custom_detector,
                    collect_pose_metrics=collect_pose_metrics,
                    pose_summary=pose_summary,
                    face_mesh=face_mesh,
                    sample_ratio=lfw_sample_ratio,
                    sample_seed=lfw_sample_seed,
                )
                identity_count += 1

                if identity_count % IDENTITY_PROGRESS_STEP == 0:
                    print(f"Processed {identity_count}/{len(person_dirs)} probe identities")

                for image_path, image in identity_images:
                    probe_records.append(
                        {
                            "dataset": dataset_name,
                            "split": split_name,
                            "identity": identity_name,
                            "path": image_path,
                            "image": image,
                        }
                    )
                    dataset_counts[split_key] += 1

            elapsed = time.perf_counter() - start_time
            print(
                f"Loaded {dataset_counts[split_key]} recognition images "
                f"from {identity_count} identities in {elapsed:.2f}s"
            )
            if dataset_name == "lfw" and lfw_sample_ratio < 1.0:
                print(
                    f"LFW probe sampling: ratio={lfw_sample_ratio:.3f} "
                    f"seed={lfw_sample_seed}"
                )
    finally:
        if face_mesh is not None and hasattr(face_mesh, "close"):
            face_mesh.close()

    return probe_records, dataset_counts, pose_summary
