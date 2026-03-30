"""
Module: pad_feature_extractor

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
from src.antispoofing.pad_texture_features import extract_lbp_features
from src.antispoofing.pad_quality_features import extract_quality_features
from src.antispoofing.pad_lft_features import extract_lft_features
from src.antispoofing.pad_lftp_features import extract_lftp_hist, to_grayscale as lftp_to_grayscale
from src.antispoofing.pad_tgsd_features import extract_tgsd_features


# ----- DEBUG FEATURE CONFIGURATION ----- #
# This switch is meant for temporary PAD debugging and feature ablation
# experiments. It lets us compare different feature subsets without
# manually commenting and uncommenting blocks before each run.
#
# Supported modes:
# - full: full original PAD feature vector
# - lbp_only: keep only LBP from both scales
# - lbp_lft: keep LBP + LFT from both scales
# - no_quality: remove quality features from both scales
# - no_down_quality: keep all features except the downsampled quality block
PAD_DEBUG_FEATURE_MODE = "full"


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
        lft_features,
    ))

    return combined_features.astype(np.float32)


def select_debug_feature_subset(
    features_original,
    features_downsampled,
    tgsd_features,
    lftp_features,
    mode=PAD_DEBUG_FEATURE_MODE,
):
    """
    Select a debug feature subset for controlled PAD ablation tests.

    This helper is intentionally small and centralized so we can switch
    between a few debug modes without touching the rest of the training
    and runtime evaluation pipeline.
    """

    orig_lbp = features_original[0:54]
    orig_quality = features_original[54:60]
    orig_lft = features_original[60:64]

    down_lbp = features_downsampled[0:54]
    down_quality = features_downsampled[54:60]
    down_lft = features_downsampled[60:64]

    orig_tgsd = tgsd_features[0:1]
    down_tgsd = tgsd_features[1:2]

    if mode == "full":
        return np.concatenate((
            features_original,
            orig_tgsd,
            features_downsampled,
            down_tgsd,
            lftp_features,
        ))

    if mode == "lbp_only":
        return np.concatenate((
            orig_lbp,
            down_lbp,
        ))

    if mode == "lbp_lft":
        return np.concatenate((
            orig_lbp,
            orig_lft,
            orig_tgsd,
            down_lbp,
            down_lft,
            down_tgsd,
        ))

    if mode == "no_quality":
        return np.concatenate((
            orig_lbp,
            orig_lft,
            orig_tgsd,
            down_lbp,
            down_lft,
            down_tgsd,
        ))

    if mode == "no_down_quality":
        return np.concatenate((
            orig_lbp,
            orig_quality,
            orig_lft,
            orig_tgsd,
            down_lbp,
            down_lft,
            down_tgsd,
        ))

    raise ValueError(
        "Unsupported PAD_DEBUG_FEATURE_MODE: "
        f"{mode}. Use 'full', 'lbp_only', 'lbp_lft', "
        "'no_quality', or 'no_down_quality'."
    )


def extract_pad_features(face_image):
    """
    Extract the combined multi-scale PAD feature vector from a face image.
    """

    features_original = extract_pad_features_single_scale(face_image)
    tgsd_features = extract_tgsd_features(face_image)

    # Blur before downsampling to reduce aliasing and make the second scale focus on coarser spoof artefacts.
    blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)
    downsampled_face = cv2.resize(blurred_face, (64, 64))

    features_downsampled = extract_pad_features_single_scale(downsampled_face)
    gray_face = lftp_to_grayscale(face_image)
    downsampled_gray = lftp_to_grayscale(downsampled_face)
    lftp_features = np.concatenate((
        extract_lftp_hist(gray_face),
        extract_lftp_hist(downsampled_gray),
    ))

    combined_features = select_debug_feature_subset(
        features_original,
        features_downsampled,
        tgsd_features,
        lftp_features,
    )

    return combined_features.astype(np.float32)
