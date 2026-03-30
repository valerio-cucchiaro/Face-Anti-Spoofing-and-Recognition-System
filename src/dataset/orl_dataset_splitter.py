"""
Module: orl_dataset_splitter

Description:
This module prepares the ORL face dataset for recognition experiments.
It reads the original identity folders from the raw dataset directory
and copies each subject's images into separate gallery and probe
partitions.

The default split follows the standard fixed rule used in this project:
images 1-5 go to the gallery and images 6-10 go to the probe set.
"""

from pathlib import Path
import shutil


# ----- CONSTANTS ----- #
ORL_RAW_DIR = Path("data/recognition/orl/raw")
ORL_GALLERY_DIR = Path("data/recognition/orl/gallery")
ORL_PROBE_DIR = Path("data/recognition/orl/probe")
IMAGE_EXTENSION = ".pgm"
GALLERY_IMAGE_IDS = {1, 2, 3, 4, 5}
PROBE_IMAGE_IDS = {6, 7, 8, 9, 10}


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


def copy_identity_split(identity_dir: Path):
    """
    Copy one ORL identity into the gallery/probe directory structure.
    """
    gallery_dir = ORL_GALLERY_DIR / identity_dir.name
    probe_dir = ORL_PROBE_DIR / identity_dir.name

    gallery_dir.mkdir(parents=True, exist_ok=True)
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
        elif image_id in PROBE_IMAGE_IDS:
            destination = probe_dir / image_path.name
        else:
            print(f"[WARNING] Skipping unexpected ORL image id: {image_path}")
            continue

        shutil.copy2(image_path, destination)


def prepare_orl_gallery_probe():
    """
    Create gallery and probe splits from the raw ORL dataset.
    """
    if not ORL_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing ORL raw dataset directory: {ORL_RAW_DIR}")

    if ORL_GALLERY_DIR.exists():
        shutil.rmtree(ORL_GALLERY_DIR)

    if ORL_PROBE_DIR.exists():
        shutil.rmtree(ORL_PROBE_DIR)

    ORL_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    ORL_PROBE_DIR.mkdir(parents=True, exist_ok=True)

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

    for identity_dir in sorted(valid_identity_dirs, key=get_identity_id):
        copy_identity_split(identity_dir)


def main():
    """
    Prepare the ORL recognition dataset using the fixed gallery/probe split.
    """
    prepare_orl_gallery_probe()


if __name__ == "__main__":
    main()
