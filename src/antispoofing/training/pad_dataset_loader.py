"""
Module: pad_dataset_loader

Description:
This module provides dataset loading utilities for the CASIA face
anti-spoofing dataset used in presentation attack detection (PAD).

The loader supports both training and test splits. It scans the live
and spoof folders, reads each image, extracts the PAD feature vector
using the anti-spoofing feature extraction pipeline, and assigns the
corresponding label.

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


# ----- CONSTANTS ----- #
TRAIN_LIVE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "casia_pad" / "train" / "live"
TRAIN_SPOOF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "casia_pad" / "train" / "spoof"
TEST_LIVE_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "casia_pad" / "test" / "live"
TEST_SPOOF_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "casia_pad" / "test" / "spoof"

MODEL_PATH = Path(__file__).resolve().parent.parent.parent.parent / "models" / "haarcascade_frontalface_default.xml"

SCALE_FACTOR = 1.1
MIN_NEIGHBORS = 6
MIN_SIZE = (30, 30)
THRESHOLD_AREA = 2000


def extract_pad_features_from_img(img, detector):
    '''
    Extract PAD feature vector from a detected face in the image.

    This function performs the preprocessing pipeline required before
    anti-spoofing feature extraction:

    image
    → face detection
    → face cropping
    → face normalization (resize, illumination normalization)
    → PAD feature extraction
    
    If no valid face is detected the function returns None.
    '''

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
    """Load the CASIA PAD training dataset and extract feature vectors."""
    return load_pad_dataset_split(TRAIN_LIVE_PATH, TRAIN_SPOOF_PATH)


def load_pad_test_dataset():
    """Load the CASIA PAD test dataset and extract feature vectors."""
    return load_pad_dataset_split(TEST_LIVE_PATH, TEST_SPOOF_PATH)


def load_pad_dataset_split(live_path, spoof_path):
    """
    Load a custom PAD dataset split from specified directories.

    This function reads live and spoof images from the provided
    directories, extracts PAD feature vectors, and assigns binary labels.

    """

    features = []
    labels = []
    valid_extensions = {".png", ".jpg", ".jpeg"}

    # Load the detector
    detector = load_face_detector(str(MODEL_PATH))

    # process live images
    for img_path in sorted(Path(live_path).iterdir()):
        if img_path.suffix.lower() not in valid_extensions:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        img_features = extract_pad_features_from_img(img, detector)
        if img_features is None:
            continue
        features.append(img_features)
        labels.append(1)

    # process spoof images
    for img_path in sorted(Path(spoof_path).iterdir()):
        if img_path.suffix.lower() not in valid_extensions:
            continue

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

