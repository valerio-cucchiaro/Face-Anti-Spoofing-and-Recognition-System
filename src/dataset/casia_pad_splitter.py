"""
Module: casia_pad_splitter

Description:
Create a derived CASIA PAD working split from the immutable raw dataset.

CASIA frame names encode both subject and source video, for example:
- bs10v1f42.png -> subject s10, video group bs10v1
- s10v3f42.png  -> subject s10, video group s10v3

The splitter copies complete subjects instead of individual videos,
avoiding subject leakage between train and validation while preserving
the original raw dataset unchanged.
"""

import argparse
import shutil
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASIA_RAW_ROOT = PROJECT_ROOT / "data" / "pad" / "casia_pad_raw"
CASIA_OUTPUT_ROOT = PROJECT_ROOT / "data" / "pad" / "casia_pad"
VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
DEFAULT_VALIDATION_RATIO = 0.20
DEFAULT_SPLIT_SEED = 42


def get_subject_group_name(image_path):
    """
    Return the CASIA subject identifier for one extracted frame.
    """

    stem = image_path.stem
    marker_index = stem.find("v")
    if marker_index <= 0:
        return stem

    prefix = stem[:marker_index]
    digits = ""
    for character in reversed(prefix):
        if not character.isdigit():
            break
        digits = character + digits

    if not digits:
        return stem

    return f"s{digits}"


def collect_subject_groups(split_label_path):
    """
    Group frame paths by CASIA subject.
    """

    groups = {}
    for image_path in sorted(Path(split_label_path).iterdir()):
        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue

        group_name = get_subject_group_name(image_path)
        groups.setdefault(group_name, []).append(image_path)

    return groups


def count_image_files(split_label_path):
    """
    Count supported image files in one split directory.
    """

    return sum(
        1
        for image_path in Path(split_label_path).iterdir()
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS
    )


def select_validation_groups(groups, validation_ratio, seed):
    """
    Select deterministic validation groups.
    """

    group_names = sorted(groups)
    if not group_names:
        return []

    validation_count = max(1, int(round(len(group_names) * validation_ratio)))
    validation_count = min(validation_count, len(group_names))
    rng = np.random.default_rng(seed)
    selected_indices = rng.choice(
        len(group_names),
        size=validation_count,
        replace=False,
    )

    return [group_names[index] for index in sorted(selected_indices)]


def ensure_raw_dataset_ready():
    """
    Ensure the immutable CASIA raw dataset exposes the expected structure.
    """

    required_directories = (
        CASIA_RAW_ROOT / "train" / "live",
        CASIA_RAW_ROOT / "train" / "spoof",
        CASIA_RAW_ROOT / "test" / "live",
        CASIA_RAW_ROOT / "test" / "spoof",
    )
    missing_directories = [
        directory for directory in required_directories
        if not directory.exists()
    ]
    if missing_directories:
        formatted_missing = ", ".join(str(directory) for directory in missing_directories)
        raise FileNotFoundError(
            "CASIA raw dataset is incomplete. Missing directories: "
            f"{formatted_missing}"
        )


def reset_output_directories():
    """
    Recreate the generated CASIA PAD output directories from scratch.
    """

    if CASIA_OUTPUT_ROOT.exists():
        shutil.rmtree(CASIA_OUTPUT_ROOT)

    for split_name in ("train", "validation", "test"):
        for label in ("live", "spoof"):
            (CASIA_OUTPUT_ROOT / split_name / label).mkdir(parents=True, exist_ok=True)


def collect_all_train_groups():
    """
    Collect the union of available train subjects across live and spoof.
    """

    all_groups = {}
    for label in ("live", "spoof"):
        train_dir = CASIA_RAW_ROOT / "train" / label

        label_groups = collect_subject_groups(train_dir)
        for group_name, paths in label_groups.items():
            all_groups.setdefault(group_name, []).extend(paths)

    return all_groups


def validate_selected_groups(selected_groups, train_live_groups, train_spoof_groups):
    """
    Ensure each selected validation subject exists in both live and spoof.
    """

    missing_live = sorted(
        group_name for group_name in selected_groups
        if group_name not in train_live_groups
    )
    missing_spoof = sorted(
        group_name for group_name in selected_groups
        if group_name not in train_spoof_groups
    )
    if missing_live or missing_spoof:
        raise ValueError(
            "Selected validation subjects are not consistent across CASIA labels: "
            f"missing_live={missing_live or '(none)'} | "
            f"missing_spoof={missing_spoof or '(none)'}"
        )


def copy_group_files(group_names, groups, target_dir):
    """
    Copy all frame files for the selected subject groups into one target split.
    """

    copied_files = 0
    for group_name in group_names:
        for source_path in groups[group_name]:
            shutil.copy2(source_path, target_dir / source_path.name)
            copied_files += 1

    return copied_files


def build_label_split(label, selected_groups):
    """
    Copy one CASIA raw train label into generated train + validation splits.
    """

    raw_train_dir = CASIA_RAW_ROOT / "train" / label
    generated_train_dir = CASIA_OUTPUT_ROOT / "train" / label
    generated_validation_dir = CASIA_OUTPUT_ROOT / "validation" / label

    groups = collect_subject_groups(raw_train_dir)
    available_selected_groups = [
        group_name for group_name in selected_groups
        if group_name in groups
    ]
    remaining_groups = [
        group_name for group_name in sorted(groups)
        if group_name not in available_selected_groups
    ]
    selected_files = sum(
        len(groups[group_name]) for group_name in available_selected_groups
    )
    remaining_files = sum(
        len(groups[group_name]) for group_name in remaining_groups
    )
    total_files = sum(len(paths) for paths in groups.values())

    print(f"CASIA {label}: raw_train_dir={raw_train_dir}")
    print(f"CASIA {label}: subject_groups={len(groups)} | files={total_files}")
    print(
        f"CASIA {label}: validation_groups={len(available_selected_groups)} | "
        f"validation_files={selected_files}"
    )
    print(
        f"CASIA {label}: remaining_train_groups={len(remaining_groups)} | "
        f"remaining_train_files={remaining_files}"
    )

    copied_validation = copy_group_files(
        available_selected_groups,
        groups,
        generated_validation_dir,
    )
    copied_train = copy_group_files(
        remaining_groups,
        groups,
        generated_train_dir,
    )

    print(
        f"CASIA {label}: copied_to_validation={copied_validation} | "
        f"copied_to_train={copied_train}"
    )


def copy_test_split(label):
    """
    Copy one CASIA raw test label unchanged into the generated output.
    """

    raw_test_dir = CASIA_RAW_ROOT / "test" / label
    generated_test_dir = CASIA_OUTPUT_ROOT / "test" / label
    copied_files = 0

    for image_path in sorted(raw_test_dir.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in VALID_IMAGE_EXTENSIONS:
            shutil.copy2(image_path, generated_test_dir / image_path.name)
            copied_files += 1

    print(f"CASIA {label}: copied_test_files={copied_files}")


def validate_generated_counts(selected_groups):
    """
    Verify that the generated CASIA PAD split matches the intended counts.
    """

    train_live_groups = collect_subject_groups(CASIA_RAW_ROOT / "train" / "live")
    train_spoof_groups = collect_subject_groups(CASIA_RAW_ROOT / "train" / "spoof")

    expected_counts = {}
    for label, raw_groups in (("live", train_live_groups), ("spoof", train_spoof_groups)):
        expected_validation = sum(
            len(raw_groups[group_name])
            for group_name in selected_groups
            if group_name in raw_groups
        )
        expected_train = sum(len(paths) for paths in raw_groups.values()) - expected_validation
        expected_test = count_image_files(CASIA_RAW_ROOT / "test" / label)

        actual_train = count_image_files(CASIA_OUTPUT_ROOT / "train" / label)
        actual_validation = count_image_files(CASIA_OUTPUT_ROOT / "validation" / label)
        actual_test = count_image_files(CASIA_OUTPUT_ROOT / "test" / label)

        expected_counts[f"train/{label}"] = (expected_train, actual_train)
        expected_counts[f"validation/{label}"] = (expected_validation, actual_validation)
        expected_counts[f"test/{label}"] = (expected_test, actual_test)

    mismatches = [
        f"{split_name}: expected={expected}, actual={actual}"
        for split_name, (expected, actual) in expected_counts.items()
        if expected != actual
    ]
    if mismatches:
        raise ValueError(
            "Generated CASIA PAD counts do not match the expected split: "
            + "; ".join(mismatches)
        )


def build_arg_parser():
    """
    Build the CLI for CASIA PAD validation split creation.
    """

    parser = argparse.ArgumentParser(
        description="Create CASIA PAD validation split by copying whole groups from casia_pad_raw."
    )
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VALIDATION_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_SEED)
    return parser


def main(
    validation_ratio=DEFAULT_VALIDATION_RATIO,
    seed=DEFAULT_SPLIT_SEED,
):
    """
    Create the CASIA validation split.
    """

    if validation_ratio <= 0.0 or validation_ratio >= 1.0:
        raise ValueError("validation_ratio must be between 0 and 1.")

    print("====== START CASIA PAD VALIDATION SPLIT ======")
    print("Validation ratio:", validation_ratio)
    print("Split seed:", seed)
    print("Split unit: subject")
    print("Input dataset:", CASIA_RAW_ROOT)
    print("Output dataset:", CASIA_OUTPUT_ROOT)

    ensure_raw_dataset_ready()

    all_groups = collect_all_train_groups()
    selected_groups = select_validation_groups(all_groups, validation_ratio, seed)
    train_live_groups = collect_subject_groups(CASIA_RAW_ROOT / "train" / "live")
    train_spoof_groups = collect_subject_groups(CASIA_RAW_ROOT / "train" / "spoof")
    validate_selected_groups(selected_groups, train_live_groups, train_spoof_groups)
    print(
        "Selected validation subjects:",
        ", ".join(selected_groups) if selected_groups else "(none)",
    )

    reset_output_directories()

    build_label_split("live", selected_groups)
    build_label_split("spoof", selected_groups)
    copy_test_split("live")
    copy_test_split("spoof")
    validate_generated_counts(selected_groups)

    print("====== END CASIA PAD VALIDATION SPLIT ======")


if __name__ == "__main__":
    cli_args = build_arg_parser().parse_args()
    main(
        validation_ratio=cli_args.validation_ratio,
        seed=cli_args.seed,
    )
