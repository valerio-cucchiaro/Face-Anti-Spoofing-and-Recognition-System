"""
Module: lfw_dataset_splitter

Description:
This module prepares the LFW face dataset for recognition experiments.
It reads the original identity folders from the raw dataset directory,
filters out identities with too few images, and copies the remaining
images into separate gallery, validation, and final probe partitions.

Known subjects are enrolled in the gallery. Unknown subjects never
appear in the gallery and are used only as biometric impostors.
"""

from pathlib import Path
import random
import shutil


# ----- CONSTANTS ----- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LFW_ROOT = PROJECT_ROOT / "data" / "recognition" / "lfw"
LFW_RAW_DIR = LFW_ROOT / "raw"
LFW_GALLERY_DIR = LFW_ROOT / "gallery"
LFW_VALIDATION_KNOWN_DIR = LFW_ROOT / "validation_known"
LFW_PROBE_KNOWN_DIR = LFW_ROOT / "probe_known"
LFW_VALIDATION_UNKNOWN_DIR = LFW_ROOT / "validation_unknown"
LFW_PROBE_UNKNOWN_DIR = LFW_ROOT / "probe_unknown"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES_PER_IDENTITY = 3
KNOWN_SUBJECT_RATIO = 0.8
RANDOM_SEED = 42

OUTPUT_DIRS = (
    LFW_GALLERY_DIR,
    LFW_VALIDATION_KNOWN_DIR,
    LFW_PROBE_KNOWN_DIR,
    LFW_VALIDATION_UNKNOWN_DIR,
    LFW_PROBE_UNKNOWN_DIR,
)


def is_valid_image(image_path: Path):
    """
    Check whether a file has a supported LFW image extension.
    """
    return image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def get_identity_image_paths(identity_dir: Path):
    """
    Load and sort all valid image paths for one LFW identity.
    """
    return sorted(
        [image_path for image_path in identity_dir.iterdir() if is_valid_image(image_path)],
        key=lambda image_path: image_path.name,
    )


def split_identity_images(image_paths):
    """
    Split one identity's images into gallery and known evaluation subsets.
    """
    gallery_count = max(1, len(image_paths) // 2)
    remaining_count = len(image_paths) - gallery_count

    if remaining_count < 1:
        return [], [], []

    gallery_paths = image_paths[:gallery_count]
    remaining_paths = image_paths[gallery_count:]
    validation_count = max(1, len(remaining_paths) // 2)
    validation_paths = remaining_paths[:validation_count]
    probe_paths = remaining_paths[validation_count:]

    if not probe_paths:
        probe_paths = validation_paths[-1:]
        validation_paths = validation_paths[:-1]

    return gallery_paths, validation_paths, probe_paths


def copy_known_identity_split(identity_name, gallery_paths, validation_paths, probe_paths):
    """
    Copy one known LFW identity into gallery, validation, and final probe splits.
    """
    gallery_dir = LFW_GALLERY_DIR / identity_name
    validation_dir = LFW_VALIDATION_KNOWN_DIR / identity_name
    probe_dir = LFW_PROBE_KNOWN_DIR / identity_name

    gallery_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    for image_path in gallery_paths:
        shutil.copy2(image_path, gallery_dir / image_path.name)

    for image_path in validation_paths:
        shutil.copy2(image_path, validation_dir / image_path.name)

    for image_path in probe_paths:
        shutil.copy2(image_path, probe_dir / image_path.name)


def copy_unknown_identity_split(identity_name, image_paths, output_root):
    """
    Copy one unknown LFW identity into one unknown evaluation split.
    """
    output_dir = output_root / identity_name
    output_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        shutil.copy2(image_path, output_dir / image_path.name)


def select_known_unknown_identities(identity_records):
    """
    Select known and unknown LFW identities with a reproducible shuffle.
    """
    if len(identity_records) < 2:
        raise ValueError(
            "Not enough LFW identities for an open-set split: "
            f"found {len(identity_records)}, need at least 2."
        )

    shuffled_records = sorted(identity_records, key=lambda record: record[0])
    random.Random(RANDOM_SEED).shuffle(shuffled_records)

    known_count = int(len(shuffled_records) * KNOWN_SUBJECT_RATIO)
    known_count = max(1, min(known_count, len(shuffled_records) - 1))

    known_records = shuffled_records[:known_count]
    unknown_records = shuffled_records[known_count:]
    return known_records, unknown_records


def reset_output_directories():
    """
    Recreate generated LFW split directories.
    """
    for output_dir in OUTPUT_DIRS:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

def prepare_lfw_gallery_probe():
    """
    Create gallery and known/unknown probe splits from the raw LFW dataset.
    """
    print("====== START LFW DATASET SPLIT ======")
    if not LFW_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing LFW raw dataset directory: {LFW_RAW_DIR}")

    identity_records = []
    skipped_identities = 0

    for identity_dir in sorted(LFW_RAW_DIR.iterdir(), key=lambda path: path.name):
        if not identity_dir.is_dir():
            continue

        image_paths = get_identity_image_paths(identity_dir)

        if not image_paths:
            skipped_identities += 1
            continue

        if len(image_paths) < MIN_IMAGES_PER_IDENTITY:
            print(f"[WARNING] Skipping {identity_dir.name}: only {len(image_paths)} valid images")
            skipped_identities += 1
            continue

        identity_records.append((identity_dir.name, image_paths))

    known_records, unknown_records = select_known_unknown_identities(identity_records)
    reset_output_directories()

    known_gallery_images = 0
    validation_known_images = 0
    probe_known_images = 0
    validation_unknown_images = 0
    probe_unknown_images = 0

    for identity_name, image_paths in known_records:
        gallery_paths, validation_paths, probe_paths = split_identity_images(image_paths)

        if not gallery_paths or not validation_paths or not probe_paths:
            print(f"[WARNING] Skipping {identity_name}: split did not produce gallery, validation, and final probe")
            skipped_identities += 1
            continue

        copy_known_identity_split(identity_name, gallery_paths, validation_paths, probe_paths)
        known_gallery_images += len(gallery_paths)
        validation_known_images += len(validation_paths)
        probe_known_images += len(probe_paths)

    unknown_midpoint = len(unknown_records) // 2
    validation_unknown_records = unknown_records[:unknown_midpoint]
    probe_unknown_records = unknown_records[unknown_midpoint:]

    for identity_name, image_paths in validation_unknown_records:
        copy_unknown_identity_split(identity_name, image_paths, LFW_VALIDATION_UNKNOWN_DIR)
        validation_unknown_images += len(image_paths)

    for identity_name, image_paths in probe_unknown_records:
        copy_unknown_identity_split(identity_name, image_paths, LFW_PROBE_UNKNOWN_DIR)
        probe_unknown_images += len(image_paths)

    print(f"Known identities: {len(known_records)}")
    print(f"Unknown identities: {len(unknown_records)}")
    print(f"Gallery images: {known_gallery_images}")
    print(f"Validation known images: {validation_known_images}")
    print(f"Final probe known images: {probe_known_images}")
    print(f"Validation unknown images: {validation_unknown_images}")
    print(f"Final probe unknown images: {probe_unknown_images}")
    print(f"Skipped identities: {skipped_identities}")
    print("====== END LFW DATASET SPLIT ======")


def main():
    """
    Prepare the LFW recognition dataset using a fixed open-set split.
    """
    prepare_lfw_gallery_probe()


if __name__ == "__main__":
    main()
