"""
Module: multipie_dataset_splitter

Description:
This module prepares the Multi-PIE HR/LR dataset for recognition
experiments. It reads the original flat HR and LR folders from the raw
dataset directory and copies a limited, reproducible subset into
gallery, validation, and test partitions.

The split keeps gallery images high-resolution only and separates
known probes from unknown probes at subject level:
- known subjects are enrolled in the gallery
- unknown subjects never appear in the gallery
"""

from pathlib import Path
import random
import shutil


# ----- CONSTANTS ----- #
MULTIPIE_ROOT = Path("data/recognition/multipie")
RAW_HR_DIR = MULTIPIE_ROOT / "raw" / "HR_128"
RAW_LR_DIR = MULTIPIE_ROOT / "raw" / "LR_4x_Nearest"

GALLERY_DIR = MULTIPIE_ROOT / "gallery"
VALIDATION_KNOWN_HR_DIR = MULTIPIE_ROOT / "validation_known_hr"
TEST_KNOWN_HR_DIR = MULTIPIE_ROOT / "test_known_hr"
VALIDATION_KNOWN_LR_DIR = MULTIPIE_ROOT / "validation_known_lr"
TEST_KNOWN_LR_DIR = MULTIPIE_ROOT / "test_known_lr"
VALIDATION_UNKNOWN_HR_DIR = MULTIPIE_ROOT / "validation_unknown_hr"
TEST_UNKNOWN_HR_DIR = MULTIPIE_ROOT / "test_unknown_hr"
VALIDATION_UNKNOWN_LR_DIR = MULTIPIE_ROOT / "validation_unknown_lr"
TEST_UNKNOWN_LR_DIR = MULTIPIE_ROOT / "test_unknown_lr"
PROBE_KNOWN_HR_DIR = MULTIPIE_ROOT / "probe_known_hr"
PROBE_KNOWN_LR_DIR = MULTIPIE_ROOT / "probe_known_lr"
PROBE_UNKNOWN_HR_DIR = MULTIPIE_ROOT / "probe_unknown_hr"
PROBE_UNKNOWN_LR_DIR = MULTIPIE_ROOT / "probe_unknown_lr"

OUTPUT_DIRS = (
    GALLERY_DIR,
    VALIDATION_KNOWN_HR_DIR,
    TEST_KNOWN_HR_DIR,
    VALIDATION_KNOWN_LR_DIR,
    TEST_KNOWN_LR_DIR,
    VALIDATION_UNKNOWN_HR_DIR,
    TEST_UNKNOWN_HR_DIR,
    VALIDATION_UNKNOWN_LR_DIR,
    TEST_UNKNOWN_LR_DIR,
)

VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

RANDOM_SEED = 42
KNOWN_SUBJECTS = 30
UNKNOWN_SUBJECTS = 10
VALIDATION_UNKNOWN_SUBJECTS = 5
GALLERY_PER_SUBJECT = 5
PROBE_PER_SUBJECT = 30


def parse_multipie_filename(image_path):
    """
    Parse the minimal Multi-PIE filename fields used by the splitter.

    Expected example:
    001_01_01_010_00_crop_128.png
    """

    parts = image_path.stem.split("_")
    if len(parts) < 5:
        raise ValueError(f"Invalid Multi-PIE filename: {image_path.name}")

    return {
        "subject_id": parts[0],
        "session": parts[1],
        "expression": parts[2],
        "pose": parts[3],
        "illumination": parts[4],
    }


def get_subject_id(image_path):
    """
    Extract the subject id from one Multi-PIE image path.
    """

    return parse_multipie_filename(image_path)["subject_id"]


def is_valid_image(image_path):
    """
    Check whether a file has a supported image extension.
    """

    return image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS


def group_hr_images_by_subject():
    """
    Load and group all HR image paths by Multi-PIE subject id.
    """

    if not RAW_HR_DIR.exists():
        raise FileNotFoundError(f"Missing Multi-PIE HR directory: {RAW_HR_DIR}")

    if not RAW_LR_DIR.exists():
        raise FileNotFoundError(f"Missing Multi-PIE LR directory: {RAW_LR_DIR}")

    grouped_paths = {}

    for image_path in sorted(RAW_HR_DIR.iterdir(), key=lambda path: path.name):
        if not is_valid_image(image_path):
            continue

        try:
            subject_id = get_subject_id(image_path)
        except ValueError as exc:
            print(f"[WARNING] {exc}")
            continue

        grouped_paths.setdefault(subject_id, []).append(image_path)

    return grouped_paths


def select_known_unknown_subjects(subject_ids):
    """
    Select known and unknown subject ids with a reproducible shuffle.
    """

    shuffled_subjects = sorted(subject_ids)
    random.Random(RANDOM_SEED).shuffle(shuffled_subjects)

    required_subjects = KNOWN_SUBJECTS + UNKNOWN_SUBJECTS
    if len(shuffled_subjects) < required_subjects:
        raise ValueError(
            "Not enough Multi-PIE subjects: "
            f"found {len(shuffled_subjects)}, need {required_subjects}."
        )

    known_subjects = shuffled_subjects[:KNOWN_SUBJECTS]
    unknown_subjects = shuffled_subjects[KNOWN_SUBJECTS:required_subjects]
    return known_subjects, unknown_subjects


def reset_output_directories():
    """
    Recreate the generated Multi-PIE split directories.
    """

    for output_dir in OUTPUT_DIRS:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for legacy_dir in (
        PROBE_KNOWN_HR_DIR,
        PROBE_KNOWN_LR_DIR,
        PROBE_UNKNOWN_HR_DIR,
        PROBE_UNKNOWN_LR_DIR,
    ):
        if legacy_dir.exists():
            shutil.rmtree(legacy_dir)


def copy_image(image_path, destination_dir):
    """
    Copy one image into an identity split directory.
    """

    destination_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, destination_dir / image_path.name)


def copy_lr_match(hr_path, destination_dir):
    """
    Copy the LR image with the same filename as the selected HR image.
    """

    lr_path = RAW_LR_DIR / hr_path.name
    if not lr_path.exists():
        return False

    copy_image(lr_path, destination_dir)
    return True


def copy_known_subject_split(subject_id, image_paths):
    """
    Copy gallery, known HR/LR validation, and known HR/LR test images for one subject.
    """

    identity_dir_name = f"s{subject_id}"
    gallery_paths = image_paths[:GALLERY_PER_SUBJECT]
    evaluation_paths = image_paths[
        GALLERY_PER_SUBJECT:GALLERY_PER_SUBJECT + PROBE_PER_SUBJECT
    ]
    validation_count = len(evaluation_paths) // 2
    validation_paths = evaluation_paths[:validation_count]
    test_paths = evaluation_paths[validation_count:]

    copied = {
        "gallery": 0,
        "validation_known_hr": 0,
        "test_known_hr": 0,
        "validation_known_lr": 0,
        "test_known_lr": 0,
        "missing_lr": 0,
    }

    for image_path in gallery_paths:
        copy_image(image_path, GALLERY_DIR / identity_dir_name)
        copied["gallery"] += 1

    for image_path in validation_paths:
        copy_image(image_path, VALIDATION_KNOWN_HR_DIR / identity_dir_name)
        copied["validation_known_hr"] += 1

        if copy_lr_match(image_path, VALIDATION_KNOWN_LR_DIR / identity_dir_name):
            copied["validation_known_lr"] += 1
        else:
            copied["missing_lr"] += 1

    for image_path in test_paths:
        copy_image(image_path, TEST_KNOWN_HR_DIR / identity_dir_name)
        copied["test_known_hr"] += 1

        if copy_lr_match(image_path, TEST_KNOWN_LR_DIR / identity_dir_name):
            copied["test_known_lr"] += 1
        else:
            copied["missing_lr"] += 1

    return copied


def copy_unknown_subject_split(subject_id, image_paths, hr_output_root, lr_output_root):
    """
    Copy unknown HR and LR evaluation images for one non-enrolled subject.
    """

    identity_dir_name = f"s{subject_id}"
    probe_paths = image_paths[:PROBE_PER_SUBJECT]

    copied = {
        "unknown_hr": 0,
        "unknown_lr": 0,
        "missing_lr": 0,
    }

    for image_path in probe_paths:
        copy_image(image_path, hr_output_root / identity_dir_name)
        copied["unknown_hr"] += 1

        if copy_lr_match(image_path, lr_output_root / identity_dir_name):
            copied["unknown_lr"] += 1
        else:
            copied["missing_lr"] += 1

    return copied


def prepare_multipie_gallery_probe():
    """
    Create Multi-PIE gallery and known/unknown HR/LR probe splits.
    """

    grouped_paths = group_hr_images_by_subject()
    known_subjects, unknown_subjects = select_known_unknown_subjects(
        grouped_paths.keys()
    )
    reset_output_directories()

    totals = {
        "gallery": 0,
        "validation_known_hr": 0,
        "test_known_hr": 0,
        "validation_known_lr": 0,
        "test_known_lr": 0,
        "validation_unknown_hr": 0,
        "test_unknown_hr": 0,
        "validation_unknown_lr": 0,
        "test_unknown_lr": 0,
        "missing_lr": 0,
    }

    for subject_id in known_subjects:
        copied = copy_known_subject_split(subject_id, grouped_paths[subject_id])
        for key, value in copied.items():
            totals[key] += value

    validation_unknown_subjects = unknown_subjects[:VALIDATION_UNKNOWN_SUBJECTS]
    test_unknown_subjects = unknown_subjects[VALIDATION_UNKNOWN_SUBJECTS:]

    for subject_id in validation_unknown_subjects:
        copied = copy_unknown_subject_split(
            subject_id,
            grouped_paths[subject_id],
            VALIDATION_UNKNOWN_HR_DIR,
            VALIDATION_UNKNOWN_LR_DIR,
        )
        totals["validation_unknown_hr"] += copied["unknown_hr"]
        totals["validation_unknown_lr"] += copied["unknown_lr"]
        totals["missing_lr"] += copied["missing_lr"]

    for subject_id in test_unknown_subjects:
        copied = copy_unknown_subject_split(
            subject_id,
            grouped_paths[subject_id],
            TEST_UNKNOWN_HR_DIR,
            TEST_UNKNOWN_LR_DIR,
        )
        totals["test_unknown_hr"] += copied["unknown_hr"]
        totals["test_unknown_lr"] += copied["unknown_lr"]
        totals["missing_lr"] += copied["missing_lr"]

    print("Multi-PIE split completed")
    print(f"Known subjects: {len(known_subjects)}")
    print(f"Unknown subjects: {len(unknown_subjects)}")
    print(f"Gallery images: {totals['gallery']}")
    print(f"Validation known HR images: {totals['validation_known_hr']}")
    print(f"Test known HR images: {totals['test_known_hr']}")
    print(f"Validation known LR images: {totals['validation_known_lr']}")
    print(f"Test known LR images: {totals['test_known_lr']}")
    print(f"Validation unknown HR images: {totals['validation_unknown_hr']}")
    print(f"Test unknown HR images: {totals['test_unknown_hr']}")
    print(f"Validation unknown LR images: {totals['validation_unknown_lr']}")
    print(f"Test unknown LR images: {totals['test_unknown_lr']}")
    print(f"Missing LR matches: {totals['missing_lr']}")


def main():
    """
    Prepare the Multi-PIE recognition dataset using a fixed reproducible split.
    """

    prepare_multipie_gallery_probe()


if __name__ == "__main__":
    main()
