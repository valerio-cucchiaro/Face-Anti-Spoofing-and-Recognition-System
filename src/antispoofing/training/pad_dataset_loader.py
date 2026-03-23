"""
Module: pad_dataset_loader

Description:
This module provides dataset loading utilities for the presentation
attack detection (PAD) training pipeline.

The loader supports both training and evaluation dataset loading,
including single-dataset and combined-dataset variants. It scans the
live and spoof folders, detects the main face with the shared
MediaPipe-based validation helper, extracts the PAD feature vector
through the anti-spoofing preprocessing pipeline, and assigns the
corresponding label.

The resulting dataset is returned as a pair of NumPy arrays:
one for the extracted feature vectors and one for the class labels.
"""

import cv2
import numpy as np
from pathlib import Path

from src.antispoofing.pad_crop_utils import extract_pad_face_crop
from src.antispoofing.pad_feature_extractor import extract_pad_features
from src.detection.face_detector import load_mediapipe_detector
from src.preprocessing.face_preprocessing import preprocess_face_for_pad
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

# ----- RUNTIME PAD DATASET ----- #
RUNTIME_ROOT = Path(__file__).resolve().parent.parent.parent.parent / "data" / "pad_runtime_dataset"

RUNTIME_TRAIN_LIVE_PATH = RUNTIME_ROOT / "train" / "live"
RUNTIME_TRAIN_SPOOF_PATH = RUNTIME_ROOT / "train" / "spoof"
RUNTIME_TEST_LIVE_PATH = RUNTIME_ROOT / "test" / "live"
RUNTIME_TEST_SPOOF_PATH = RUNTIME_ROOT / "test" / "spoof"

# ----- CONSTANTS ----- #
THRESHOLD_AREA = 8000


def extract_pad_features_from_image(img, detector):
    """
    Extract a PAD feature vector from the main valid face found in the
    input image.

    This function performs the preprocessing pipeline required before
    anti-spoofing feature extraction:

    image
    -> face detection
    -> shared PAD crop extraction
    -> face normalization
    -> PAD feature extraction

    If no valid face is detected, the function returns ``None``.
    """

    # Detect the face inside the image and validate it
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

    face = preprocess_face_for_pad(
        pad_face_crop,
        0,
        0,
        pad_face_crop.shape[1],
        pad_face_crop.shape[0]
    )

    return extract_pad_features(face)


def load_combined_pad_training_dataset():
    """
    Load the combined PAD training dataset (CASIA + Replay-Attack).
    """
    # CASIA dataset
    print("Loading CASIA PAD dataset...")
    X1, y1 = load_pad_split_dataset(TRAIN_LIVE_PATH, TRAIN_SPOOF_PATH)

    # ReplayAttack dataset
    print("Loading ReplayAttack PAD dataset...")
    X2, y2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=True,
    )

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return X, y


def load_replay_pad_training_dataset():
    """
    Load the Replay-Attack PAD training dataset only.
    """
    return load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=True,
    )


def load_runtime_pad_training_dataset():
    """
    Load the custom runtime PAD training dataset only.
    """
    return load_pad_split_dataset(
        RUNTIME_TRAIN_LIVE_PATH,
        RUNTIME_TRAIN_SPOOF_PATH,
        balance=False,
    )


def load_pad_test_dataset():
    """
    Load the default PAD test dataset (CASIA) and extract feature vectors.
    """
    return load_pad_split_dataset(TEST_LIVE_PATH, TEST_SPOOF_PATH)


def load_pad_test_dataset_casia():
    """
    Load the CASIA PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(TEST_LIVE_PATH, TEST_SPOOF_PATH)


def load_pad_test_dataset_replay():
    """
    Load the Replay-Attack PAD test dataset and extract feature vectors.
    """
    return load_pad_split_dataset(
        REPLAY_TEST_LIVE_PATH,
        REPLAY_TEST_SPOOF_PATH,
        balance=False,
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


def load_combined_with_runtime_pad_training_dataset(runtime_repeat=5):
    """
    Load the combined PAD training dataset and oversample the runtime
    training split so it has a stronger influence during training.
    """

    print("Loading CASIA PAD dataset...")
    X1, y1 = load_pad_split_dataset(TRAIN_LIVE_PATH, TRAIN_SPOOF_PATH)

    print("Loading ReplayAttack PAD dataset...")
    X2, y2 = load_pad_split_dataset(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=True,
    )

    print("Loading runtime PAD dataset...")
    X3, y3 = load_runtime_pad_training_dataset()

    if len(y3) > 0 and runtime_repeat > 1:
        X3 = np.concatenate([X3] * runtime_repeat, axis=0)
        y3 = np.concatenate([y3] * runtime_repeat, axis=0)

    X = np.concatenate([X1, X2, X3], axis=0)
    y = np.concatenate([y1, y2, y3], axis=0)

    return X, y


def load_pad_split_dataset(live_path, spoof_path, balance=False):
    """
    Load a custom PAD dataset split from specified directories.

    This function reads live and spoof images from the provided
    directories, extracts PAD feature vectors, and assigns binary labels.

    If balance=True, the spoof class is randomly reduced to match
    the number of live images.
    """

    import random

    features = []
    labels = []
    valid_extensions = {".png", ".jpg", ".jpeg"}

    # Load the detector
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

    # Replay-Attack contains many more spoof frames than live frames.
    # During training we optionally downsample spoof samples to reduce bias.
    if balance:
        n_live = len(live_paths)
        spoof_paths = random.sample(spoof_paths, min(len(spoof_paths), n_live))

    # Process live images
    for img_path in live_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_image(img, detector)
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(1)

    # Process spoof images
    for img_path in spoof_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_image(img, detector)
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


if __name__ == "__main__":
    features, labels = load_combined_pad_training_dataset()
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
