"""
Module: pad_dataset_loader

Description:
This module provides dataset loading utilities for the presentation
attack detection (PAD) training pipeline.

The loader supports both single-dataset and combined-dataset loading.
It scans the live and spoof folders, reads each image, extracts the
PAD feature vector using the anti-spoofing feature extraction pipeline,
and assigns the corresponding label.

The resulting dataset is returned as a pair of NumPy arrays:
one for the extracted feature vectors and one for the class labels.
"""

import cv2
import numpy as np
from pathlib import Path

from src.antispoofing.feature_extractor import extract_pad_features
from src.detection.face_detector import load_face_detector
from src.preprocessing.frame_preprocessing import get_valid_face
from src.preprocessing.face_preprocessing import preprocess_face_for_pad


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

# ----- CONSTANTS ----- #
MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 2000


def extract_pad_features_from_img(img, detector):
    """
    Extract PAD feature vector from a detected face in the image.

    This function performs the preprocessing pipeline required before
    anti-spoofing feature extraction:

    image
    -> face detection
    -> face cropping
    -> face normalization (resize, illumination normalization)
    -> PAD feature extraction

    If no valid face is detected the function returns None.
    """

    # Detect the face inside the image and validate it
    face_box, gray_frame, _ = get_valid_face(
        img,
        detector,
        SCALE_FACTOR,
        MIN_NEIGHBORS,
        MIN_SIZE,
        THRESHOLD_AREA
    )

    if face_box is None:
        return None

    x, y, w, h = face_box

    # Normalize the face region before feature extraction
    face = preprocess_face_for_pad(gray_frame, x, y, w, h)

    return extract_pad_features(face)


def load_pad_dataset():
    """
    Load the combined PAD training dataset (CASIA + Replay-Attack).
    """
    # CASIA dataset
    print("Loading CASIA PAD dataset...")
    X1, y1 = load_pad_dataset_split(TRAIN_LIVE_PATH, TRAIN_SPOOF_PATH)

    # ReplayAttack dataset
    print("Loading ReplayAttack PAD dataset...")
    X2, y2 = load_pad_dataset_split(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=True,
    )

    X = np.concatenate([X1, X2], axis=0)
    y = np.concatenate([y1, y2], axis=0)

    return X, y


def load_pad_dataset_replay():
    """
    Load the Replay-Attack PAD training dataset only.
    """
    return load_pad_dataset_split(
        REPLAY_TRAIN_LIVE_PATH,
        REPLAY_TRAIN_SPOOF_PATH,
        balance=True,
    )


def load_pad_test_dataset():
    """
    Load the default PAD test dataset (CASIA) and extract feature vectors.
    """
    return load_pad_dataset_split(TEST_LIVE_PATH, TEST_SPOOF_PATH)


def load_pad_test_dataset_casia():
    """
    Load the CASIA PAD test dataset and extract feature vectors.
    """
    return load_pad_dataset_split(TEST_LIVE_PATH, TEST_SPOOF_PATH)


def load_pad_test_dataset_replay():
    """
    Load the Replay-Attack PAD test dataset and extract feature vectors.
    """
    return load_pad_dataset_split(
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


def load_pad_dataset_split(live_path, spoof_path, balance=False):
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
    detector = load_face_detector(str(MODEL_PATH))

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

        img_features = extract_pad_features_from_img(img, detector)
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(1)

    # Process spoof images
    for img_path in spoof_paths:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_img(img, detector)
        if img_features is None:
            continue

        features.append(img_features)
        labels.append(0)

    return np.array(features, dtype=np.float32), np.array(labels, dtype=np.int32)


if __name__ == "__main__":
    features, labels = load_pad_dataset()
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)
