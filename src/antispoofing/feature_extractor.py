"""
Module: feature_extractor

Description:
This module combines the descriptors used for face anti-spoofing
detection.

It extracts Local Binary Pattern (LBP) features and image quality
features from a face image, then concatenates them into a single
fixed-length feature vector.

The resulting combined descriptor is used as input to the
presentation attack detection (PAD) classifier.
"""

import cv2
import numpy as np
from src.antispoofing.texture_features import extract_lbp_features
from src.antispoofing.quality_features import extract_quality_features
from src.antispoofing.lft_features import extract_lft_features


def extract_pad_features_single_scale(face_image):
    """
    Extract PAD features from a single-scale face image.
    """

    lbp_features = extract_lbp_features(face_image)
    quality_features = extract_quality_features(face_image)
    lft_features = extract_lft_features(face_image)

    combined_features = np.concatenate((
        lbp_features,
        quality_features,
        lft_features
    ))

    return combined_features.astype(np.float32)


def extract_pad_features(face_image):
    """
    Extract the combined multi-scale PAD feature vector from a face image.
    """

    features_original = extract_pad_features_single_scale(face_image)

    # Blur before downsampling to reduce aliasing and make the second scale focus on coarser spoof artefacts.
    blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)
    downsampled_face = cv2.resize(blurred_face, (64, 64))

    features_downsampled = extract_pad_features_single_scale(downsampled_face)

    combined_features = np.concatenate((
        features_original,
        features_downsampled
    ))

    return combined_features.astype(np.float32)
