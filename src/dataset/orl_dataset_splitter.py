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
        print(f"[WARNING] {identity_dir.name} contains {len(image_paths)} images instead of 10")
        return

    for image_path in image_paths:
        image_id = get_image_id(image_path)

        if image_id in GALLERY_IMAGE_IDS:
            destination = gallery_dir / image_path.name
        elif image_id in VALIDATION_KNOWN_IMAGE_IDS:
            destination = validation_dir / image_path.name
        elif image_id in PROBE_KNOWN_IMAGE_IDS:
            destination = probe_dir / image_path.name
        else:
            print(f"[WARNING] Skipping unexpected ORL image id: {image_path}")
            continue

        shutil.copy2(image_path, destination)


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
        print(f"[WARNING] {identity_dir.name} contains {len(image_paths)} images instead of 10")
        return

    for image_path in image_paths:
        image_id = get_image_id(image_path)

        if image_id not in UNKNOWN_PROBE_IMAGE_IDS:
            print(f"[WARNING] Skipping unexpected ORL image id: {image_path}")
            continue

        shutil.copy2(image_path, output_dir / image_path.name)


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

    valid_identity_dirs = []

    for identity_dir in ORL_RAW_DIR.iterdir():
        if not identity_dir.is_dir():
            continue

        try:
            get_identity_id(identity_dir)
        except ValueError:
            print(f"[WARNING] Skipping unexpected ORL identity directory: {identity_dir.name}")
            continue

        valid_identity_dirs.append(identity_dir)

    known_dirs, unknown_dirs = select_known_unknown_identities(valid_identity_dirs)
    reset_output_directories()

    for identity_dir in known_dirs:
        copy_known_identity_split(identity_dir)

    validation_unknown_dirs = unknown_dirs[:VALIDATION_UNKNOWN_SUBJECTS]
    probe_unknown_dirs = unknown_dirs[VALIDATION_UNKNOWN_SUBJECTS:]

    for identity_dir in validation_unknown_dirs:
        copy_unknown_identity_split(identity_dir, ORL_VALIDATION_UNKNOWN_DIR)

    for identity_dir in probe_unknown_dirs:
        copy_unknown_identity_split(identity_dir, ORL_PROBE_UNKNOWN_DIR)

    print(f"Known identities: {len(known_dirs)}")
    print(f"Unknown identities: {len(unknown_dirs)}")
    print(f"Gallery images: {len(known_dirs) * len(GALLERY_IMAGE_IDS)}")
    print(f"Validation known images: {len(known_dirs) * len(VALIDATION_KNOWN_IMAGE_IDS)}")
    print(f"Final probe known images: {len(known_dirs) * len(PROBE_KNOWN_IMAGE_IDS)}")
    print(f"Validation unknown images: {len(validation_unknown_dirs) * len(UNKNOWN_PROBE_IMAGE_IDS)}")
    print(f"Final probe unknown images: {len(probe_unknown_dirs) * len(UNKNOWN_PROBE_IMAGE_IDS)}")
    print("====== END ORL DATASET SPLIT ======")


def main():
    """
    Prepare the ORL recognition dataset using a fixed open-set split.
    """
    prepare_orl_gallery_probe()


if __name__ == "__main__":
    main()
