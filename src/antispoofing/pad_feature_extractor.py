"""
Module: pad_feature_extractor

Description:
This module combines the descriptors used for face anti-spoofing
detection.

The active default descriptor combines the baseline multi-scale stack
(LBP + quality) with the promoted LFT GGD-light representation and the
promoted TGSD refinement.

The debug switch still allows controlled comparisons against the
baseline-only branch for PAD ablation experiments.
"""

import cv2
import numpy as np
import time
from src.antispoofing.pad_texture_features import extract_lbp_features
from src.antispoofing.pad_quality_features import extract_quality_features
from src.antispoofing.pad_lft_features import (
    extract_lft_ggd_light_features,
)
from src.antispoofing.pad_tgsd_features import (
    configure_tgsd,
    extract_tgsd_features,
    get_tgsd_config,
)


# ----- DEBUG FEATURE CONFIGURATION ----- #
# This switch is meant for temporary PAD debugging and feature ablation
# experiments. It lets us compare different feature subsets without
# manually commenting and uncommenting blocks before each run.
#
# Supported modes:
# - baseline: multi-scale baseline only
# - tgsd: baseline + TGSD (promoted default)
#
# The historical LFTP module is intentionally kept in the repository as
# an archived descriptor block, but it is no longer wired into the
# active PAD feature path.
PAD_DEBUG_FEATURE_MODE = "tgsd"
PAD_LFT_MODE = "ggd_light"


def get_pad_feature_modes():
    """
    Return the PAD feature modes currently active in this module.
    """

    active_modes = {
        "pad_debug_feature_mode": PAD_DEBUG_FEATURE_MODE,
        "pad_lft_mode": PAD_LFT_MODE,
    }
    active_modes.update(get_tgsd_config())
    return active_modes


def configure_pad_feature_modes(
    debug_mode=None,
    lft_mode=None,
    tgsd_tau=None,
    tgsd_top_percent=None,
    tgsd_gradient_operator=None,
):
    """
    Update the active PAD feature-extraction modes at runtime.

    This helper lets experiment runners switch between the promoted
    TGSD default and the baseline-only ablation branch, while keeping
    the promoted ``ggd_light`` LFT representation fixed across runs.
    """
    global PAD_DEBUG_FEATURE_MODE, PAD_LFT_MODE

    if debug_mode is not None:
        if debug_mode not in {"baseline", "tgsd"}:
            raise ValueError(
                "Unsupported PAD_DEBUG_FEATURE_MODE: "
                f"{debug_mode}. Use 'baseline' or 'tgsd'."
            )
        PAD_DEBUG_FEATURE_MODE = debug_mode

    if lft_mode is not None:
        if lft_mode != "ggd_light":
            raise ValueError(
                "Unsupported PAD_LFT_MODE: "
                f"{lft_mode}. Use 'ggd_light'."
            )
        PAD_LFT_MODE = lft_mode

    configure_tgsd(
        tau=tgsd_tau,
        top_percent=tgsd_top_percent,
        gradient_operator=tgsd_gradient_operator,
    )


def extract_pad_features_single_scale(face_image):
    """
    Extract one scale of the PAD baseline descriptor.

    The per-scale descriptor always includes LBP, quality features, and
    the promoted lightweight GGD-inspired LFT pair [shape, variance].
    """

    lbp_features = extract_lbp_features(face_image)
    quality_features = extract_quality_features(face_image)
    if PAD_LFT_MODE == "ggd_light":
        lft_features = extract_lft_ggd_light_features(face_image)
    else:
        raise ValueError(
            "Unsupported PAD_LFT_MODE: "
            f"{PAD_LFT_MODE}. Use 'ggd_light'."
        )

    combined_features = np.concatenate((
        lbp_features,
        quality_features,
        lft_features,
    ))

    return combined_features.astype(np.float32)


def extract_pad_features_single_scale_with_timing(face_image, scale_name):
    """
    Extract one scale of the PAD baseline descriptor together with
    per-block timings.
    """

    lbp_start_time = time.perf_counter()
    lbp_features = extract_lbp_features(face_image)
    lbp_elapsed_ms = (time.perf_counter() - lbp_start_time) * 1000.0

    quality_start_time = time.perf_counter()
    quality_features = extract_quality_features(face_image)
    quality_elapsed_ms = (time.perf_counter() - quality_start_time) * 1000.0

    lft_start_time = time.perf_counter()
    if PAD_LFT_MODE == "ggd_light":
        lft_features = extract_lft_ggd_light_features(face_image)
    else:
        raise ValueError(
            "Unsupported PAD_LFT_MODE: "
            f"{PAD_LFT_MODE}. Use 'ggd_light'."
        )
    lft_elapsed_ms = (time.perf_counter() - lft_start_time) * 1000.0

    combined_features = np.concatenate((
        lbp_features,
        quality_features,
        lft_features,
    ))

    timings_ms = {
        f"{scale_name}_lbp": lbp_elapsed_ms,
        f"{scale_name}_quality": quality_elapsed_ms,
        f"{scale_name}_lft": lft_elapsed_ms,
        f"{scale_name}_total": (
            lbp_elapsed_ms + quality_elapsed_ms + lft_elapsed_ms
        ),
    }

    return combined_features.astype(np.float32), timings_ms


def select_debug_feature_subset(
    baseline_features,
    tgsd_features=None,
    mode=None,
):
    """
    Select the final PAD feature vector for the current debug mode.

    The baseline branch is always built first. The "tgsd" mode appends
    only TGSD on top of that same baseline.
    """

    selected_mode = mode or PAD_DEBUG_FEATURE_MODE

    if selected_mode == "baseline":
        return baseline_features

    if selected_mode == "tgsd":
        if tgsd_features is None:
            raise ValueError("TGSD features are required when PAD mode is 'tgsd'.")
        return np.concatenate((
            baseline_features,
            tgsd_features,
        ))

    raise ValueError(
        "Unsupported PAD_DEBUG_FEATURE_MODE: "
        f"{selected_mode}. Use 'baseline' or 'tgsd'."
    )


def extract_pad_features(face_image):
    """
    Extract the complete multi-scale PAD feature vector from a face image.

    The function always builds the two-scale baseline first. Depending on
    ``PAD_DEBUG_FEATURE_MODE``, it can either return that baseline as-is
    or append TGSD.
    """

    selected_mode = PAD_DEBUG_FEATURE_MODE
    features_original = extract_pad_features_single_scale(face_image)

    # Blur before downsampling so the second scale emphasizes coarser spoof artefacts.
    blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)
    downsampled_face = cv2.resize(blurred_face, (64, 64))
    features_downsampled = extract_pad_features_single_scale(downsampled_face)
    baseline_features = np.concatenate((
        features_original,
        features_downsampled,
    ))

    tgsd_features = None
    if selected_mode == "tgsd":
        tgsd_features = extract_tgsd_features(face_image)

    combined_features = select_debug_feature_subset(
        baseline_features,
        tgsd_features,
        mode=selected_mode,
    )

    return combined_features.astype(np.float32)


def extract_pad_features_with_timing(face_image):
    """
    Extract the complete PAD feature vector together with coarse timing
    information for the main descriptor blocks.
    """

    selected_mode = PAD_DEBUG_FEATURE_MODE
    timings_ms = {}

    original_features, original_timings = extract_pad_features_single_scale_with_timing(
        face_image,
        "original",
    )
    timings_ms.update(original_timings)

    blur_downsample_start_time = time.perf_counter()
    blurred_face = cv2.GaussianBlur(face_image, (3, 3), 0)
    downsampled_face = cv2.resize(blurred_face, (64, 64))
    timings_ms["blur_downsample"] = (
        time.perf_counter() - blur_downsample_start_time
    ) * 1000.0

    downsampled_features, downsampled_timings = extract_pad_features_single_scale_with_timing(
        downsampled_face,
        "downsampled",
    )
    timings_ms.update(downsampled_timings)

    baseline_features = np.concatenate((
        original_features,
        downsampled_features,
    ))

    tgsd_features = None
    if selected_mode == "tgsd":
        tgsd_start_time = time.perf_counter()
        tgsd_features = extract_tgsd_features(face_image)
        timings_ms["tgsd"] = (time.perf_counter() - tgsd_start_time) * 1000.0

    select_start_time = time.perf_counter()
    combined_features = select_debug_feature_subset(
        baseline_features,
        tgsd_features,
        mode=selected_mode,
    )
    timings_ms["feature_selection"] = (
        time.perf_counter() - select_start_time
    ) * 1000.0
    timings_ms["feature_total"] = sum(timings_ms.values())

    return combined_features.astype(np.float32), timings_ms
