"""
Module: lfw_dataset_splitter

Description:
This module prepares the LFW face dataset for recognition experiments.
It reads the original identity folders from the raw dataset directory,
filters out identities with too few images, and copies the remaining
images into separate gallery and probe partitions.

The split is computed independently for each identity so that both
partitions contain at least one image whenever the identity is kept.
"""

from pathlib import Path
import shutil


# ----- CONSTANTS ----- #
LFW_RAW_DIR = Path("data/recognition/lfw/raw")
LFW_GALLERY_DIR = Path("data/recognition/lfw/gallery")
LFW_PROBE_DIR = Path("data/recognition/lfw/probe")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MIN_IMAGES_PER_IDENTITY = 3


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
    Split one identity's images into gallery and probe subsets.
    """
    gallery_count = max(1, len(image_paths) // 2)
    probe_count = len(image_paths) - gallery_count

    if probe_count < 1:
        return [], []

    gallery_paths = image_paths[:gallery_count]
    probe_paths = image_paths[gallery_count:]
    return gallery_paths, probe_paths


def copy_identity_split(identity_name, gallery_paths, probe_paths):
    """
    Copy the gallery and probe images for one LFW identity.
    """
    gallery_dir = LFW_GALLERY_DIR / identity_name
    probe_dir = LFW_PROBE_DIR / identity_name

    gallery_dir.mkdir(parents=True, exist_ok=True)
    probe_dir.mkdir(parents=True, exist_ok=True)

    for image_path in gallery_paths:
        shutil.copy2(image_path, gallery_dir / image_path.name)

    for image_path in probe_paths:
        shutil.copy2(image_path, probe_dir / image_path.name)


def prepare_lfw_gallery_probe():
    """
    Create gallery and probe splits from the raw LFW dataset.
    """
    if not LFW_RAW_DIR.exists():
        raise FileNotFoundError(f"Missing LFW raw dataset directory: {LFW_RAW_DIR}")

    if LFW_GALLERY_DIR.exists():
        shutil.rmtree(LFW_GALLERY_DIR)

    if LFW_PROBE_DIR.exists():
        shutil.rmtree(LFW_PROBE_DIR)

    LFW_GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    LFW_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    processed_identities = 0
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

        gallery_paths, probe_paths = split_identity_images(image_paths)

        if not gallery_paths or not probe_paths:
            print(f"[WARNING] Skipping {identity_dir.name}: split did not produce both gallery and probe")
            skipped_identities += 1
            continue

        copy_identity_split(identity_dir.name, gallery_paths, probe_paths)
        processed_identities += 1

    print(f"Processed identities: {processed_identities}")
    print(f"Skipped identities: {skipped_identities}")


def main():
    """
    Prepare the LFW recognition dataset using the configured gallery/probe split.
    """
    prepare_lfw_gallery_probe()


if __name__ == "__main__":
    main()
