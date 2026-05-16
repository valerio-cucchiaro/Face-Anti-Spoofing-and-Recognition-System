"""
Module: orl_dataset_splitter

Description:
This module prepares the ORL face dataset for recognition experiments.
It reads the original identity folders from the raw dataset directory
and copies a reproducible subject-level open-set split into gallery,
validation, and final probe partitions.

Known subjects are enrolled in the gallery. Unknown subjects never
appear in the gallery and are used only as biometric impostors.
"""

from pathlib import Path
import random
import shutil


# ----- CONSTANTS ----- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORL_ROOT = PROJECT_ROOT / "data" / "recognition" / "orl"
ORL_RAW_DIR = ORL_ROOT / "raw"
ORL_GALLERY_DIR = ORL_ROOT / "gallery"
ORL_VALIDATION_KNOWN_DIR = ORL_ROOT / "validation_known"
ORL_PROBE_KNOWN_DIR = ORL_ROOT / "probe_known"
ORL_VALIDATION_UNKNOWN_DIR = ORL_ROOT / "validation_unknown"
ORL_PROBE_UNKNOWN_DIR = ORL_ROOT / "probe_unknown"
IMAGE_EXTENSION = ".pgm"
GALLERY_IMAGE_IDS = {1, 2, 3, 4, 5}
VALIDATION_KNOWN_IMAGE_IDS = {6, 7}
PROBE_KNOWN_IMAGE_IDS = {8, 9, 10}
UNKNOWN_PROBE_IMAGE_IDS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
RANDOM_SEED = 42
KNOWN_SUBJECTS = 30
UNKNOWN_SUBJECTS = 10
VALIDATION_UNKNOWN_SUBJECTS = 5

OUTPUT_DIRS = (
    ORL_GALLERY_DIR,
    ORL_VALIDATION_KNOWN_DIR,
    ORL_PROBE_KNOWN_DIR,
    ORL_VALIDATION_UNKNOWN_DIR,
    ORL_PROBE_UNKNOWN_DIR,
)


def count_split_files(root_dir: Path):
    """
    Count identity directories and copied images under one ORL split root.
    """
    if not root_dir.exists():
        return 0, 0

    non_empty_identity_count = 0
    image_count = 0
    for identity_dir in [path for path in root_dir.iterdir() if path.is_dir()]:
        identity_image_count = len(
            [
                path for path in identity_dir.iterdir()
                if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION
            ]
        )
        image_count += identity_image_count
        if identity_image_count > 0:
            non_empty_identity_count += 1

    return non_empty_identity_count, image_count


def verify_orl_output_counts(expected_counts):
    """
    Verify that the generated ORL split matches the reported counts.
    """
    actual_counts = {
        "gallery": count_split_files(ORL_GALLERY_DIR),
        "validation_known": count_split_files(ORL_VALIDATION_KNOWN_DIR),
        "probe_known": count_split_files(ORL_PROBE_KNOWN_DIR),
        "validation_unknown": count_split_files(ORL_VALIDATION_UNKNOWN_DIR),
        "probe_unknown": count_split_files(ORL_PROBE_UNKNOWN_DIR),
    }

    expected_pairs = {
        "gallery": (expected_counts["known_identities_copied"], expected_counts["gallery_images"]),
        "validation_known": (
            expected_counts["known_identities_copied"],
            expected_counts["validation_known_images"],
        ),
        "probe_known": (
            expected_counts["known_identities_copied"],
            expected_counts["probe_known_images"],
        ),
        "validation_unknown": (
            expected_counts["validation_unknown_identities_copied"],
            expected_counts["validation_unknown_images"],
        ),
        "probe_unknown": (
            expected_counts["probe_unknown_identities_copied"],
            expected_counts["probe_unknown_images"],
        ),
    }

    mismatches = []
    for split_name, expected_pair in expected_pairs.items():
        actual_pair = actual_counts[split_name]
        if actual_pair != expected_pair:
            mismatches.append(
                f"{split_name}: expected identities/images={expected_pair}, actual={actual_pair}"
            )

    if mismatches:
        raise RuntimeError(
            "ORL split verification failed after copy: " + "; ".join(mismatches)
        )


def get_image_id(image_path: Path) -> int:
    """
    Extract the numeric image id from an ORL filename such as `1.pgm`.
    """
    try:
        return int(image_path.stem)
    except ValueError as exc:
        raise ValueError(f"Invalid ORL image name: {image_path.name}") from exc


def get_identity_id(identity_dir: Path) -> int:
    """
    Extract the numeric subject id from an ORL identity folder such as `s1`.
    """
    name = identity_dir.name

    if not name.startswith("s"):
        raise ValueError(f"Invalid ORL identity name: {name}")

    try:
        return int(name[1:])
    except ValueError as exc:
        raise ValueError(f"Invalid ORL identity name: {name}") from exc


def copy_known_identity_split(identity_dir: Path):
    """
    Copy one known ORL identity into gallery, validation, and final probe splits.
    """
    gallery_dir = ORL_GALLERY_DIR / identity_dir.name
    validation_dir = ORL_VALIDATION_KNOWN_DIR / identity_dir.name
    probe_dir = ORL_PROBE_KNOWN_DIR / identity_dir.name

    gallery_dir.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [path for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION],
        key=get_image_id,
    )

    if len(image_paths) != 10:
        print(f"[WARNING] Skipping {identity_dir.name}: contains {len(image_paths)} images instead of 10")
        return {
            "copied": False,
            "gallery_images": 0,
            "validation_known_images": 0,
            "probe_known_images": 0,
            "unexpected_image_ids": 0,
        }

    gallery_images = 0
    validation_known_images = 0
    probe_known_images = 0
    unexpected_image_ids = 0

    for image_path in image_paths:
        image_id = get_image_id(image_path)

        if image_id in GALLERY_IMAGE_IDS:
            destination = gallery_dir / image_path.name
            gallery_images += 1
        elif image_id in VALIDATION_KNOWN_IMAGE_IDS:
            destination = validation_dir / image_path.name
            validation_known_images += 1
        elif image_id in PROBE_KNOWN_IMAGE_IDS:
            destination = probe_dir / image_path.name
            probe_known_images += 1
        else:
            print(f"[WARNING] Skipping unexpected ORL image id: {image_path}")
            unexpected_image_ids += 1
            continue

        shutil.copy2(image_path, destination)

    return {
        "copied": True,
        "gallery_images": gallery_images,
        "validation_known_images": validation_known_images,
        "probe_known_images": probe_known_images,
        "unexpected_image_ids": unexpected_image_ids,
    }


def copy_unknown_identity_split(identity_dir: Path, output_root: Path):
    """
    Copy one unknown ORL identity into one unknown evaluation split.
    """
    output_dir = output_root / identity_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [path for path in identity_dir.iterdir() if path.is_file() and path.suffix.lower() == IMAGE_EXTENSION],
        key=get_image_id,
    )

    if len(image_paths) != 10:
        print(f"[WARNING] Skipping {identity_dir.name}: contains {len(image_paths)} images instead of 10")
        return {"copied": False, "unknown_images": 0, "unexpected_image_ids": 0}

    unknown_images = 0
    unexpected_image_ids = 0

    for image_path in image_paths:
        image_id = get_image_id(image_path)

        if image_id not in UNKNOWN_PROBE_IMAGE_IDS:
            print(f"[WARNING] Skipping unexpected ORL image id: {image_path}")
            unexpected_image_ids += 1
            continue

        shutil.copy2(image_path, output_dir / image_path.name)
        unknown_images += 1

    return {
        "copied": True,
        "unknown_images": unknown_images,
        "unexpected_image_ids": unexpected_image_ids,
    }


def select_known_unknown_identities(identity_dirs):
    """
    Select known and unknown ORL identities with a reproducible shuffle.
    """
    shuffled_dirs = sorted(identity_dirs, key=get_identity_id)
    random.Random(RANDOM_SEED).shuffle(shuffled_dirs)

    required_subjects = KNOWN_SUBJECTS + UNKNOWN_SUBJECTS
    if len(shuffled_dirs) < required_subjects:
        raise ValueError(
            "Not enough ORL subjects: "
            f"found {len(shuffled_dirs)}, need {required_subjects}."
        )

    known_dirs = shuffled_dirs[:KNOWN_SUBJECTS]
    unknown_dirs = shuffled_dirs[KNOWN_SUBJECTS:required_subjects]
    return known_dirs, unknown_dirs


def reset_output_directories():
    """
    Recreate generated ORL split directories.
    """
    for output_dir in OUTPUT_DIRS:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)


def prepare_orl_gallery_probe():
    """
    Create gallery and known/unknown probe splits from the raw ORL dataset.
    """
    print("====== START ORL DATASET SPLIT ======")
    if not ORL_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing ORL raw dataset directory: {ORL_RAW_DIR}")

    if not ORL_RAW_DIR.is_dir():
        raise NotADirectoryError(f"ORL raw path is not a directory: {ORL_RAW_DIR}")

    valid_identity_dirs = []
    skipped_identities = 0
    skip_reason_counts = {
        "invalid_identity_dir": 0,
        "unexpected_image_count": 0,
    }
    unexpected_image_ids = 0

    for identity_dir in ORL_RAW_DIR.iterdir():
        if not identity_dir.is_dir():
            continue

        try:
            get_identity_id(identity_dir)
        except ValueError:
            print(f"[WARNING] Skipping unexpected ORL identity directory: {identity_dir.name}")
            skipped_identities += 1
            skip_reason_counts["invalid_identity_dir"] += 1
            continue

        valid_identity_dirs.append(identity_dir)

    if not valid_identity_dirs:
        raise RuntimeError(
            f"No valid ORL identity directories were found under {ORL_RAW_DIR}."
        )

    known_dirs, unknown_dirs = select_known_unknown_identities(valid_identity_dirs)
    reset_output_directories()

    known_identities_copied = 0
    gallery_images = 0
    validation_known_images = 0
    probe_known_images = 0
    for identity_dir in known_dirs:
        result = copy_known_identity_split(identity_dir)
        if not result["copied"]:
            skipped_identities += 1
            skip_reason_counts["unexpected_image_count"] += 1
            continue
        known_identities_copied += 1
        gallery_images += result["gallery_images"]
        validation_known_images += result["validation_known_images"]
        probe_known_images += result["probe_known_images"]
        unexpected_image_ids += result["unexpected_image_ids"]

    validation_unknown_dirs = unknown_dirs[:VALIDATION_UNKNOWN_SUBJECTS]
    probe_unknown_dirs = unknown_dirs[VALIDATION_UNKNOWN_SUBJECTS:]

    validation_unknown_identities_copied = 0
    validation_unknown_images = 0
    for identity_dir in validation_unknown_dirs:
        result = copy_unknown_identity_split(identity_dir, ORL_VALIDATION_UNKNOWN_DIR)
        if not result["copied"]:
            skipped_identities += 1
            skip_reason_counts["unexpected_image_count"] += 1
            continue
        validation_unknown_identities_copied += 1
        validation_unknown_images += result["unknown_images"]
        unexpected_image_ids += result["unexpected_image_ids"]

    probe_unknown_identities_copied = 0
    probe_unknown_images = 0
    for identity_dir in probe_unknown_dirs:
        result = copy_unknown_identity_split(identity_dir, ORL_PROBE_UNKNOWN_DIR)
        if not result["copied"]:
            skipped_identities += 1
            skip_reason_counts["unexpected_image_count"] += 1
            continue
        probe_unknown_identities_copied += 1
        probe_unknown_images += result["unknown_images"]
        unexpected_image_ids += result["unexpected_image_ids"]

    verify_orl_output_counts(
        {
            "known_identities_copied": known_identities_copied,
            "gallery_images": gallery_images,
            "validation_known_images": validation_known_images,
            "probe_known_images": probe_known_images,
            "validation_unknown_identities_copied": validation_unknown_identities_copied,
            "validation_unknown_images": validation_unknown_images,
            "probe_unknown_identities_copied": probe_unknown_identities_copied,
            "probe_unknown_images": probe_unknown_images,
        }
    )

    print(f"Known identities selected: {len(known_dirs)}")
    print(f"Known identities copied: {known_identities_copied}")
    print(f"Known identities skipped: {len(known_dirs) - known_identities_copied}")
    print(f"Unknown identities selected: {len(unknown_dirs)}")
    print(f"Validation unknown identities copied: {validation_unknown_identities_copied}")
    print(f"Final probe unknown identities copied: {probe_unknown_identities_copied}")
    print(f"Unknown identities skipped: {len(unknown_dirs) - validation_unknown_identities_copied - probe_unknown_identities_copied}")
    print(f"Gallery images: {gallery_images}")
    print(f"Validation known images: {validation_known_images}")
    print(f"Final probe known images: {probe_known_images}")
    print(f"Validation unknown images: {validation_unknown_images}")
    print(f"Final probe unknown images: {probe_unknown_images}")
    print(f"Skipped identities: {skipped_identities}")
    print(
        "Skip reasons: "
        f"invalid_identity_dir={skip_reason_counts['invalid_identity_dir']} | "
        f"unexpected_image_count={skip_reason_counts['unexpected_image_count']}"
    )
    if unexpected_image_ids:
        print(f"Unexpected image ids skipped: {unexpected_image_ids}")
    print("====== END ORL DATASET SPLIT ======")


def main():
    """
    Prepare the ORL recognition dataset using a fixed open-set split.
    """
    prepare_orl_gallery_probe()


if __name__ == "__main__":
    main()
