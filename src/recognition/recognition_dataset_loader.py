"""
Module: recognition_dataset_loader

Description:
This module provides the dataset loading utilities used by the
recognition training and evaluation pipeline.

It supports the custom, ORL, and LFW recognition datasets and can
load them individually or combine them when required by an experiment.
Gallery images are used for training, while probe images are used for
evaluation.
"""

import cv2
from pathlib import Path
import numpy as np
import time


# ----- CONSTANTS ----- #
RECOGNITION_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "recognition"
DATASET_SPLIT_DIRS = {
    "custom": {
        "gallery": RECOGNITION_ROOT / "custom" / "gallery",
        "probe": RECOGNITION_ROOT / "custom" / "probe",
    },
    "orl": {
        "gallery": RECOGNITION_ROOT / "orl" / "gallery",
        "probe": RECOGNITION_ROOT / "orl" / "probe",
    },
    "lfw": {
        "gallery": RECOGNITION_ROOT / "lfw" / "gallery",
        "probe": RECOGNITION_ROOT / "lfw" / "probe",
    },
}
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".pgm"}
IDENTITY_PROGRESS_STEP = 50


def resolve_dataset_names(mode):
    """
    Resolve a configured recognition mode into one or more dataset names.
    """
    if mode == "all":
        return ["custom", "orl", "lfw"]

    if mode in DATASET_SPLIT_DIRS:
        return [mode]

    raise ValueError(
        f"Unsupported recognition mode: {mode}. Use 'custom', 'orl', 'lfw', or 'all'."
    )


def should_namespace_identities(mode):
    """
    Decide whether identity names should be prefixed with the dataset name.
    """
    return mode == "all"


def build_identity_name(dataset_name, person_name, namespace_identities=False):
    """
    Build the identity label used in the current experiment.
    """
    if namespace_identities:
        return f"{dataset_name}:{person_name}"

    return person_name


def get_split_directories(mode, split):
    """
    Resolve the selected datasets to their gallery or probe directories.
    """
    dataset_dirs = []

    for dataset_name in resolve_dataset_names(mode):
        split_dir = DATASET_SPLIT_DIRS[dataset_name][split]

        if not split_dir.exists():
            raise FileNotFoundError(f"Recognition {split} directory not found: {split_dir}")

        dataset_dirs.append((dataset_name, split_dir))

    return dataset_dirs


def load_images_from_identity_dir(identity_dir):
    """
    Load all valid grayscale images from one identity directory.
    """
    images = []

    for image_path in sorted(identity_dir.iterdir(), key=lambda path: path.name):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Warning: Could not read image {image_path}")
            continue

        images.append((image_path, image))

    return images


def load_gallery_dataset(mode):
    """
    Load one or more gallery datasets for recognition training.
    """
    namespace_identities = should_namespace_identities(mode)
    faces = []
    labels = []
    label_map = {}
    dataset_counts = {}

    for dataset_name, gallery_dir in get_split_directories(mode, "gallery"):
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

            identity_images = load_images_from_identity_dir(person_dir)

            if not identity_images:
                continue

            current_label = len(label_map)
            label_map[current_label] = identity_name
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

    return faces, np.array(labels), label_map, dataset_counts


def load_probe_dataset(mode, namespace_identities=False):
    """
    Load one or more probe datasets for recognition evaluation.
    """
    probe_records = []
    dataset_counts = {}

    for dataset_name, probe_dir in get_split_directories(mode, "probe"):
        start_time = time.perf_counter()
        dataset_counts[dataset_name] = 0
        identity_count = 0
        person_dirs = sorted(
            [path for path in probe_dir.iterdir() if path.is_dir()],
            key=lambda path: path.name,
        )

        print(f"Loading probe dataset: {dataset_name}")
        print(f"Found {len(person_dirs)} probe identities")

        for person_dir in person_dirs:
            identity_name = build_identity_name(
                dataset_name,
                person_dir.name,
                namespace_identities=namespace_identities,
            )

            identity_images = load_images_from_identity_dir(person_dir)
            identity_count += 1

            if identity_count % IDENTITY_PROGRESS_STEP == 0:
                print(f"Processed {identity_count}/{len(person_dirs)} probe identities")

            for image_path, image in identity_images:
                probe_records.append(
                    {
                        "dataset": dataset_name,
                        "identity": identity_name,
                        "path": image_path,
                        "image": image,
                    }
                )
                dataset_counts[dataset_name] += 1

        elapsed = time.perf_counter() - start_time
        print(
            f"Loaded {dataset_counts[dataset_name]} probe images "
            f"from {identity_count} identities in {elapsed:.2f}s"
        )

    return probe_records, dataset_counts
