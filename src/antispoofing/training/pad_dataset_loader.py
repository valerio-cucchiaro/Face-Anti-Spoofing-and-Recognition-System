"""
Module: pad_dataset_loader

Description:
This module provides dataset loading utilities for the presentation
attack detection (PAD) training pipeline.

The loader supports both training and evaluation dataset loading,
including single-dataset and combined-dataset variants.

The active dataset contract is ``raw_frame``: images are treated as
raw frames that still need detection, crop construction, and PAD
feature extraction. ``final_pad_crop`` is retained only as a legacy
compatibility path.

The resulting dataset is returned as a pair of NumPy arrays:
one for the extracted feature vectors and one for the class labels.
"""

import cv2
import numpy as np
from pathlib import Path
import time

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_frame_quality import PAD_MIN_VARIANCE_LAPLACIAN
from src.antispoofing.pad_inference_preprocessing import extract_pad_features_from_crop
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.frame_preprocessing import get_valid_face


# ----- CASIA DATASET ----- #
CASIA_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "casia_pad"

TRAIN_LIVE_PATH = CASIA_ROOT / "train" / "live"
TRAIN_SPOOF_PATH = CASIA_ROOT / "train" / "spoof"
TEST_LIVE_PATH = CASIA_ROOT / "test" / "live"
TEST_SPOOF_PATH = CASIA_ROOT / "test" / "spoof"

# ----- REPLAY ATTACK DATASET ----- #
REPLAY_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "replay_attack_pad"

REPLAY_TRAIN_LIVE_PATH = REPLAY_ROOT / "train" / "live"
REPLAY_TRAIN_SPOOF_PATH = REPLAY_ROOT / "train" / "spoof"
REPLAY_TEST_LIVE_PATH = REPLAY_ROOT / "test" / "live"
REPLAY_TEST_SPOOF_PATH = REPLAY_ROOT / "test" / "spoof"

# ----- CUSTOM PAD DATASET ----- #
PAD_CUSTOM_RAW_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad_custom_raw"

CUSTOM_TRAIN_LIVE_PATH = PAD_CUSTOM_RAW_ROOT / "train" / "live"
CUSTOM_TRAIN_SPOOF_PATH = PAD_CUSTOM_RAW_ROOT / "train" / "spoof"
CUSTOM_TEST_LIVE_PATH = PAD_CUSTOM_RAW_ROOT / "test" / "live"
CUSTOM_TEST_SPOOF_PATH = PAD_CUSTOM_RAW_ROOT / "test" / "spoof"

# Legacy aliases kept for compatibility while the rest of the project is
# being migrated away from the old runtime naming.
RUNTIME_ROOT = PAD_CUSTOM_RAW_ROOT
RUNTIME_TRAIN_LIVE_PATH = CUSTOM_TRAIN_LIVE_PATH
RUNTIME_TRAIN_SPOOF_PATH = CUSTOM_TRAIN_SPOOF_PATH
RUNTIME_TEST_LIVE_PATH = CUSTOM_TEST_LIVE_PATH
RUNTIME_TEST_SPOOF_PATH = CUSTOM_TEST_SPOOF_PATH

# ----- CONSTANTS ----- #
THRESHOLD_AREA = 8000
TRAIN_SAMPLING_SEED = 42
CASIA_SAMPLE_RATIO = 1.0
REPLAY_SAMPLE_RATIO = 1.0
PAD_IMAGE_PROGRESS_STEP = 500
LIVE_PHOTOMETRIC_AUGMENTATION_SEED = 42
LIVE_PHOTOMETRIC_AUGMENTATION_MODE = "mix"


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


def apply_gamma_adjustment(gray_image, rng):
    """
    Internal helper used by the promoted ``mix`` live augmentation.
    """

    gamma = float(rng.uniform(0.8, 1.25))
    normalized = gray_image.astype(np.float32) / 255.0
    adjusted = np.power(normalized, gamma)
    return np.clip(adjusted * 255.0, 0, 255).astype(np.uint8), "gamma"


def apply_brightness_contrast_adjustment(gray_image, rng):
    """
    Internal helper used by the promoted ``mix`` live augmentation.
    """

    alpha = float(rng.uniform(0.9, 1.1))
    beta = float(rng.uniform(-15.0, 15.0))
    adjusted = gray_image.astype(np.float32) * alpha + beta
    return np.clip(adjusted, 0, 255).astype(np.uint8), "brightness_contrast"


def apply_side_shading_adjustment(gray_image, rng):
    """
    Internal helper used by the promoted ``mix`` live augmentation.
    """

    height, width = gray_image.shape[:2]
    if width <= 1:
        return gray_image.copy(), "side_shading(flat)"

    start = float(rng.uniform(0.88, 1.0))
    end = float(rng.uniform(1.0, 1.12))
    if rng.random() < 0.5:
        start, end = end, start

    gradient = np.linspace(start, end, width, dtype=np.float32)
    shading_map = np.tile(gradient, (height, 1))
    shaded = gray_image.astype(np.float32) * shading_map
    return np.clip(shaded, 0, 255).astype(np.uint8), "side_shading"


def apply_combined_light_adjustment(gray_image, rng):
    """
    Internal helper used by the promoted ``mix`` live augmentation.
    """

    adjusted_image, _ = apply_gamma_adjustment(gray_image, rng)
    adjusted_image, _ = apply_brightness_contrast_adjustment(adjusted_image, rng)
    return adjusted_image, "combined"


def apply_gamma_side_shading_adjustment(gray_image, rng):
    """
    Internal helper used by the promoted ``mix`` live augmentation.
    """

    adjusted_image, _ = apply_gamma_adjustment(gray_image, rng)
    adjusted_image, _ = apply_side_shading_adjustment(adjusted_image, rng)
    return adjusted_image, "gamma_side_shading"


def apply_live_photometric_augmentation(gray_image, rng, mode=None):
    """
    Create one mild photometric live augmentation.

    The public PAD training setup now supports only the promoted
    ``mix`` mode. The helper augmentation functions remain separate
    only so the mix can randomly sample one family per live image.
    """

    selected_mode = mode or LIVE_PHOTOMETRIC_AUGMENTATION_MODE
    if selected_mode != "mix":
        raise ValueError(
            "Unsupported live photometric augmentation mode: "
            f"{selected_mode}. Use 'mix'."
        )

    augmentation_functions = (
        apply_gamma_adjustment,
        apply_brightness_contrast_adjustment,
        apply_side_shading_adjustment,
        apply_combined_light_adjustment,
        apply_gamma_side_shading_adjustment,
    )
    augmentation_function = augmentation_functions[int(rng.integers(len(augmentation_functions)))]

    return augmentation_function(gray_image, rng)


def extract_pad_features_from_raw_frame(
    img,
    detector,
    apply_quality_gate=True,
):
    """
    Extract PAD features from a raw frame by running face detection and
    shared PAD crop construction, then delegating the entire post-crop
    path to the canonical PAD crop helper.
    """

    # Detect the face inside the image and validate it.
    face_box, gray_frame, _ = get_valid_face(
        img,
        detector,
        THRESHOLD_AREA
    )

    if face_box is None:
        return None

    x, y, w, h = face_box

    # Reuse the same padded crop used by the runtime pipeline before
    # applying the standard PAD normalization.
    pad_face_crop = extract_pad_face_crop(gray_frame, x, y, w, h)
    if pad_face_crop is None or pad_face_crop.size == 0:
        return None

    return extract_pad_features_from_final_crop(
        pad_face_crop,
        apply_quality_gate=apply_quality_gate,
    )


def extract_pad_features_from_final_crop(
    img,
    apply_quality_gate=True,
):
    """
    Legacy helper for PAD crops that were already preprocessed before
    being stored on disk.

    New dataset flows should use ``raw_frame`` and let the loader run
    face detection and crop construction itself.
    """

    return extract_pad_features_from_crop(
        img,
        apply_quality_gate=apply_quality_gate,
        min_variance_laplacian=PAD_MIN_VARIANCE_LAPLACIAN,
    )


def extract_pad_features_from_dataset_image(
    img,
    detector=None,
    dataset_type="raw_frame",
    apply_quality_gate=True,
):
    """
    Dispatch PAD feature extraction according to the declared dataset
    image type.

    Supported dataset types:
    - ``raw_frame``: original image that still needs detection/cropping
    - ``final_pad_crop``: legacy compatibility crop already produced on disk
    """

    if dataset_type == "raw_frame":
        if detector is None:
            raise ValueError("A detector is required for dataset_type='raw_frame'.")
        return extract_pad_features_from_raw_frame(
            img,
            detector,
            apply_quality_gate=apply_quality_gate,
        )

    if dataset_type == "final_pad_crop":
        return extract_pad_features_from_final_crop(
            img,
            apply_quality_gate=apply_quality_gate,
        )

    raise ValueError(
        "Unsupported dataset_type: "
        f"{dataset_type}. Use 'raw_frame' or 'final_pad_crop'."
    )


def load_combined_pad_training_dataset(
    casia_sample_ratio=CASIA_SAMPLE_RATIO,
    replay_sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    casia_balance=False,
    replay_balance=True,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Load the combined PAD training dataset (CASIA + Replay-Attack).
    """
    # CASIA dataset
    print("Loading CASIA PAD dataset...")
    X1, y1 = load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=casia_balance,
        live_ratio=casia_sample_ratio,
        spoof_ratio=casia_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
    )

    # ReplayAttack dataset
    print("Loading ReplayAttack PAD dataset...")
    X2, y2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=replay_balance,
        live_ratio=replay_sample_ratio,
        spoof_ratio=replay_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
    )

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return X, y


def load_casia_pad_training_dataset(
    sample_ratio=CASIA_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    balance=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the CASIA PAD training dataset with deterministic sampling.
    """
    return load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=balance,
        live_ratio=sample_ratio,
        spoof_ratio=sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
    )


def load_replay_pad_training_dataset(
    sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    balance=True,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the Replay-Attack PAD training dataset only.
    """
    return load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=balance,
        live_ratio=sample_ratio,
        spoof_ratio=sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
    )


def load_custom_pad_training_dataset(
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load the custom PAD training dataset from pad_custom_raw.
    """
    return load_pad_split_dataset(
        CUSTOM_TRAIN_LIVE_PATH,
        CUSTOM_TRAIN_SPOOF_PATH,
        balance=False,
        dataset_type="raw_frame",
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
    )


def load_runtime_pad_training_dataset(
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Legacy alias for the custom PAD training dataset.
    """
    return load_custom_pad_training_dataset(
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
    )


def load_custom_pad_test_dataset(return_image_paths=False):
    """
    Load the custom PAD test dataset from pad_custom_raw.
    """
    return load_pad_split_dataset(
        CUSTOM_TEST_LIVE_PATH,
        CUSTOM_TEST_SPOOF_PATH,
        dataset_type="raw_frame",
        return_image_paths=return_image_paths,
    )


def load_pad_test_dataset():
    """
    Load the default PAD test dataset (CASIA) and extract feature vectors.
    """
    return load_pad_split_dataset(
        TEST_LIVE_PATH,
        TEST_SPOOF_PATH,
        dataset_type="raw_frame",
    )


def load_pad_test_dataset_casia():
    """
    Load the CASIA PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(
        TEST_LIVE_PATH,
        TEST_SPOOF_PATH,
        dataset_type="raw_frame",
    )


def load_pad_test_dataset_replay():
    """
    Load the Replay-Attack PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(
        REPLAY_TEST_LIVE_PATH,
        REPLAY_TEST_SPOOF_PATH,
        balance=False,
        dataset_type="raw_frame",
    )


def load_pad_test_dataset_combined():
    """
    Load the combined PAD test dataset (CASIA + Replay-Attack).
    """

    print("Loading CASIA PAD test dataset...")
    X1, y1 = load_pad_test_dataset_casia()

    print("Loading ReplayAttack PAD test dataset...")
    X2, y2 = load_pad_test_dataset_replay()

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return X, y


def load_combined_with_custom_pad_training_dataset(
    runtime_repeat=5,
    casia_sample_ratio=CASIA_SAMPLE_RATIO,
    replay_sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    casia_balance=False,
    replay_balance=True,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Legacy combined training helper kept for compatibility while the
    project moves to the custom PAD naming.
    """
    return load_combined_with_runtime_pad_training_dataset(
        runtime_repeat=runtime_repeat,
        casia_sample_ratio=casia_sample_ratio,
        replay_sample_ratio=replay_sample_ratio,
        sample_seed=sample_seed,
        casia_balance=casia_balance,
        replay_balance=replay_balance,
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
        augment_live_photometric_scope=augment_live_photometric_scope,
    )


def load_combined_with_runtime_pad_training_dataset(
    runtime_repeat=5,
    casia_sample_ratio=CASIA_SAMPLE_RATIO,
    replay_sample_ratio=REPLAY_SAMPLE_RATIO,
    sample_seed=TRAIN_SAMPLING_SEED,
    casia_balance=False,
    replay_balance=True,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
    augment_live_photometric_scope="all",
):
    """
    Load the combined PAD training dataset and oversample the runtime
    training split so it has a stronger influence during training.
    """

    print("Loading CASIA PAD dataset...")
    X1, y1 = load_pad_split_dataset(
        TRAIN_LIVE_PATH,
        TRAIN_SPOOF_PATH,
        balance=casia_balance,
        live_ratio=casia_sample_ratio,
        spoof_ratio=casia_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
    )

    print("Loading ReplayAttack PAD dataset...")
    X2, y2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=replay_balance,
        live_ratio=replay_sample_ratio,
        spoof_ratio=replay_sample_ratio,
        sample_seed=sample_seed,
        dataset_type="raw_frame",
        augment_live_photometric=(
            augment_live_photometric and augment_live_photometric_scope == "all"
        ),
        augment_live_photometric_mode=augment_live_photometric_mode,
    )

    print("Loading runtime PAD dataset...")
    X3, y3 = load_runtime_pad_training_dataset(
        augment_live_photometric=augment_live_photometric,
        augment_live_photometric_mode=augment_live_photometric_mode,
    )
    runtime_base_count = len(y3)

    if len(y3) > 0 and runtime_repeat > 1:
        X3 = np.concatenate([X3] * runtime_repeat, axis=0)
        y3 = np.concatenate([y3] * runtime_repeat, axis=0)

    X = np.concatenate([X1, X2, X3], axis=0)
    y = np.concatenate([y1, y2, y3], axis=0)

    casia_count = len(y1)
    replay_count = len(y2)
    runtime_effective_count = len(y3)
    total_count = len(y)

    casia_weight = (casia_count / total_count * 100) if total_count else 0.0
    replay_weight = (replay_count / total_count * 100) if total_count else 0.0
    runtime_weight = (runtime_effective_count / total_count * 100) if total_count else 0.0

    print("Number of samples effectively used for training")
    print("Training sampling seed:", sample_seed)
    print(f"CASIA sample ratio: {casia_sample_ratio:.2f}")
    print(f"Replay sample ratio: {replay_sample_ratio:.2f}")
    print("CASIA pre-extraction balance:", casia_balance)
    print("Replay pre-extraction balance:", replay_balance)
    print("CASIA train:", casia_count)
    print("Replay-Attack train:", replay_count)
    print("Runtime train (base):", runtime_base_count)
    print("Runtime repeat:", runtime_repeat)
    print("Runtime train (effective):", runtime_effective_count)
    print("Final training set:", total_count)
    print(f"CASIA weight: {casia_weight:.2f}%")
    print(f"Replay-Attack weight: {replay_weight:.2f}%")
    print(f"Runtime weight: {runtime_weight:.2f}%")

    return X, y


def load_pad_split_dataset(
    live_path,
    spoof_path,
    balance=False,
    live_ratio=1.0,
    spoof_ratio=1.0,
    sample_seed=None,
    dataset_type="raw_frame",
    apply_quality_gate=True,
    return_image_paths=False,
    augment_live_photometric=False,
    augment_live_photometric_mode=None,
):
    """
    Load a custom PAD dataset split from specified directories.

    This function reads live and spoof images from the provided
    directories, extracts PAD feature vectors, and assigns binary labels.

    If balance=True, the larger class is downsampled before feature
    extraction to match the smaller one. After detection, quality
    filtering, and feature extraction the final valid sample counts
    can still become unbalanced.

    The shared quality gate applies only to the crop loading path
    selected by ``apply_quality_gate``; it does not duplicate the
    runtime pipeline history gate.
    """

    features = []
    labels = []
    sample_paths = []
    valid_extensions = {".png", ".jpg", ".jpeg"}
    start_time = time.perf_counter()
    augmentation_seed = (
        sample_seed if sample_seed is not None else LIVE_PHOTOMETRIC_AUGMENTATION_SEED
    )
    augmentation_rng = np.random.default_rng(augmentation_seed)
    augmented_live = 0
    augmentation_mode_counts = {}

    # Load the detector only when the dataset images are still raw
    # frames that require face detection and crop construction.
    detector = None
    if dataset_type == "raw_frame":
        detector = load_mediapipe_detector()

    # Build image path lists
    live_paths = [
        img_path for img_path in sorted(Path(live_path).iterdir())
        if img_path.suffix.lower() in valid_extensions
    ]

    spoof_paths = [
        img_path for img_path in sorted(Path(spoof_path).iterdir())
        if img_path.suffix.lower() in valid_extensions
    ]

    live_paths = sample_image_paths(
        live_paths,
        ratio=live_ratio,
        seed=sample_seed,
    )
    spoof_paths = sample_image_paths(
        spoof_paths,
        ratio=spoof_ratio,
        seed=None if sample_seed is None else sample_seed + 1,
    )

    # Optionally downsample the larger class before feature extraction
    # so the split starts from a balanced set of image paths.
    if balance:
        n_live = len(live_paths)
        n_spoof = len(spoof_paths)
        n_target = min(n_live, n_spoof)
        rng = np.random.default_rng(sample_seed)

        if n_live > n_target:
            indices = rng.choice(len(live_paths), size=n_target, replace=False)
            indices = np.sort(indices)
            live_paths = [live_paths[i] for i in indices]

        if n_spoof > n_target:
            indices = rng.choice(len(spoof_paths), size=n_target, replace=False)
            indices = np.sort(indices)
            spoof_paths = [spoof_paths[i] for i in indices]

    print(f"Loading PAD split: live={live_path} spoof={spoof_path}")
    print(f"Dataset type: {dataset_type}")
    print(f"Pre-extraction balance enabled: {balance}")
    print(f"Live photometric augmentation enabled: {augment_live_photometric}")
    if augment_live_photometric:
        print(
            "Live photometric augmentation mode:",
            augment_live_photometric_mode or LIVE_PHOTOMETRIC_AUGMENTATION_MODE,
        )
    print(f"Selected live images: {len(live_paths)}")
    print(f"Selected spoof images: {len(spoof_paths)}")

    # Process live images
    valid_live = 0
    for index, img_path in enumerate(live_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_dataset_image(
            img,
            detector=detector,
            dataset_type=dataset_type,
            apply_quality_gate=apply_quality_gate,
        )
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(1)
        if return_image_paths:
            sample_paths.append(str(img_path))
        valid_live += 1

        if augment_live_photometric:
            gray_img = img
            if img.ndim == 3:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            augmented_img, augmentation_name = apply_live_photometric_augmentation(
                gray_img,
                augmentation_rng,
                mode=augment_live_photometric_mode,
            )
            if dataset_type == "raw_frame" and augmented_img.ndim == 2:
                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_GRAY2BGR)
            augmented_features = extract_pad_features_from_dataset_image(
                augmented_img,
                detector=detector,
                dataset_type=dataset_type,
                apply_quality_gate=apply_quality_gate,
            )
            if augmented_features is not None:
                features.append(augmented_features)
                labels.append(1)
                if return_image_paths:
                    sample_paths.append(f"{img_path} | augmented:{augmentation_name}")
                augmented_live += 1
                augmentation_mode_counts[augmentation_name] = (
                    augmentation_mode_counts.get(augmentation_name, 0) + 1
                )

        if index % PAD_IMAGE_PROGRESS_STEP == 0:
            print(f"Processed live images: {index}/{len(live_paths)}")

    # Process spoof images
    valid_spoof = 0
    for index, img_path in enumerate(spoof_paths, start=1):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_dataset_image(
            img,
            detector=detector,
            dataset_type=dataset_type,
            apply_quality_gate=apply_quality_gate,
        )
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(0)
        if return_image_paths:
            sample_paths.append(str(img_path))
        valid_spoof += 1

        if index % PAD_IMAGE_PROGRESS_STEP == 0:
            print(f"Processed spoof images: {index}/{len(spoof_paths)}")

    elapsed = time.perf_counter() - start_time
    print(f"Valid live samples extracted: {valid_live}")
    if augment_live_photometric:
        print(f"Valid augmented live samples extracted: {augmented_live}")
        if augmentation_mode_counts:
            print("Augmentation mode counts:", augmentation_mode_counts)
    print(f"Valid spoof samples extracted: {valid_spoof}")
    if valid_live + valid_spoof > 0:
        live_ratio = valid_live / (valid_live + valid_spoof) * 100
        spoof_ratio = valid_spoof / (valid_live + valid_spoof) * 100
        print(f"Final valid live weight: {live_ratio:.2f}%")
        print(f"Final valid spoof weight: {spoof_ratio:.2f}%")
    print(f"PAD split loading completed in {elapsed:.2f}s")

    features_array = np.array(features, dtype=np.float32)
    labels_array = np.array(labels, dtype=np.int32)

    if return_image_paths:
        return features_array, labels_array, sample_paths

    return features_array, labels_array


if __name__ == "__main__":
    features, labels = load_combined_pad_training_dataset()
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
