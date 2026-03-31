"""
Module: pad_feature_extractor

Description:
This module combines the descriptors used for face anti-spoofing
detection.

The baseline descriptor combines Local Binary Pattern (LBP), image
quality, and one selectable LFT representation at two scales.

Optional debug additions can then append TGSD and LFTP blocks on top of
that baseline to support controlled PAD ablation experiments.
"""

import cv2
import numpy as np
from src.antispoofing.pad_texture_features import extract_lbp_features
from src.antispoofing.pad_quality_features import extract_quality_features
from src.antispoofing.pad_lft_features import (
    extract_lft_features,
    extract_lft_ggd_light_features,
)
from src.antispoofing.pad_lftp_features import extract_lftp_hist, to_grayscale as lftp_to_grayscale
from src.antispoofing.pad_tgsd_features import extract_tgsd_features


# ----- DEBUG FEATURE CONFIGURATION ----- #
# This switch is meant for temporary PAD debugging and feature ablation
# experiments. It lets us compare different feature subsets without
# manually commenting and uncommenting blocks before each run.
#
# Supported modes:
# - baseline: multi-scale baseline only
# - full: baseline + TGSD + LFTP
PAD_DEBUG_FEATURE_MODE = "baseline"
PAD_LFT_MODE = "summary"   # summary or ggd_light


def extract_pad_features_single_scale(face_image):
    """
    Extract one scale of the PAD baseline descriptor.

    The per-scale descriptor always includes LBP and quality features.
    The LFT branch is then represented either by the original summary
    statistics or by the lightweight GGD-inspired pair [shape, variance].
    """

    lbp_features = extract_lbp_features(face_image)
    quality_features = extract_quality_features(face_image)
    if PAD_LFT_MODE == "summary":
        lft_features = extract_lft_features(face_image)
    elif PAD_LFT_MODE == "ggd_light":
        lft_features = extract_lft_ggd_light_features(face_image)
    else:
        raise ValueError(
            "Unsupported PAD_LFT_MODE: "
            f"{PAD_LFT_MODE}. Use 'summary' or 'ggd_light'."
        )

    combined_features = np.concatenate((
        lbp_features,
        quality_features,
        lft_features,
    ))

    return combined_features.astype(np.float32)


def select_debug_feature_subset(
    baseline_features,
    tgsd_features,
    lftp_features,
    mode=None,
):
    """
    Select the final PAD feature vector for the current debug mode.

    The baseline branch is always built first. The "full" mode appends
    TGSD and LFTP on top of that same baseline so runtime and offline
    experiments share the same feature path.
    """

    selected_mode = mode or PAD_DEBUG_FEATURE_MODE

    if selected_mode == "baseline":
        return baseline_features

    if selected_mode == "full":
        return np.concatenate((
            baseline_features,
            tgsd_features,
            lftp_features,
        ))

    raise ValueError(
        "Unsupported PAD_DEBUG_FEATURE_MODE: "
        f"{selected_mode}. Use 'baseline' or 'full'."
    )


def extract_pad_features(face_image):
    """
    Extract the complete multi-scale PAD feature vector from a face image.

    The function always builds the two-scale baseline first. Depending on
    ``PAD_DEBUG_FEATURE_MODE``, it can either return that baseline as-is
    or append TGSD and LFTP blocks for the legacy full configuration.
    """

    features_original = extract_pad_features_single_scale(face_image)
    tgsd_features = extract_tgsd_features(face_image)

    # Blur before downsampling so the second scale emphasizes coarser spoof artefacts.
    blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)
    downsampled_face = cv2.resize(blurred_face, (64, 64))

    features_downsampled = extract_pad_features_single_scale(downsampled_face)
    gray_face = lftp_to_grayscale(face_image)
    downsampled_gray = lftp_to_grayscale(downsampled_face)
    lftp_features = np.concatenate((
        extract_lftp_hist(gray_face),
        extract_lftp_hist(downsampled_gray),
    ))
    baseline_features = np.concatenate((
        features_original,
        features_downsampled,
    ))

    combined_features = select_debug_feature_subset(
        baseline_features,
        tgsd_features,
        lftp_features,
    )

    return combined_features.astype(np.float32)
