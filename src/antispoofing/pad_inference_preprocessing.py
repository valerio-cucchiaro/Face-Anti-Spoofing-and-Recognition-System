"""
Module: pad_inference_preprocessing

Description:
This module provides the canonical post-crop PAD preprocessing path.

It receives an already built PAD crop, optionally validates its shared
crop quality, applies the same face preprocessing used during
training, and extracts the PAD feature vector required by the
classifier.
"""

import cv2
import time

from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_pad_crop_quality,
)
from src.preprocessing.face_preprocessing import preprocess_face_for_pad
from src.antispoofing.pad_feature_extractor import (
    extract_pad_features,
    extract_pad_features_with_timing,
)


def convert_pad_crop_to_grayscale(face_crop):
    """
    Convert a PAD crop to grayscale while preserving already grayscale
    inputs unchanged.
    """

    if face_crop is None or face_crop.size == 0:
        return None

    if len(face_crop.shape) == 3:
        return cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    return face_crop


def extract_pad_features_from_crop(
    face_crop,
    apply_quality_gate=False,
    min_variance_laplacian=PAD_MIN_VARIANCE_LAPLACIAN,
):
    """
    Extract PAD features from an already built PAD crop.

    This is the canonical post-crop path shared by runtime inference
    helpers and by training loaders that operate on saved PAD crops.
    When enabled, the shared quality gate rejects only geometrically
    invalid crops before preprocessing; callers that want stricter
    blur rejection should inspect the quality result explicitly.
    """

    gray_face = convert_pad_crop_to_grayscale(face_crop)
    if gray_face is None:
        return None

    if apply_quality_gate:
        quality_result = evaluate_pad_crop_quality(
            gray_face,
            min_variance_laplacian=min_variance_laplacian,
        )
        if not quality_result["is_valid_crop"]:
            return None

    # Reuse the same normalization used during PAD training by treating
    # the whole crop as the face region.
    face = preprocess_face_for_pad(
        gray_face,
        0,
        0,
        gray_face.shape[1],
        gray_face.shape[0]
    )

    if face is None:
        return None

    return extract_pad_features(face)


def extract_pad_features_from_crop_with_timing(
    face_crop,
    apply_quality_gate=False,
    min_variance_laplacian=PAD_MIN_VARIANCE_LAPLACIAN,
):
    """
    Extract PAD features from an already built PAD crop together with
    timing information for preprocessing and feature blocks.

    This helper mirrors the canonical post-crop path and exposes the
    same optional quality gate used by offline loaders.
    """

    timings_ms = {}

    grayscale_start_time = time.perf_counter()
    gray_face = convert_pad_crop_to_grayscale(face_crop)
    timings_ms["grayscale"] = (time.perf_counter() - grayscale_start_time) * 1000.0
    if gray_face is None:
        return None, timings_ms

    if apply_quality_gate:
        quality_gate_start_time = time.perf_counter()
        quality_result = evaluate_pad_crop_quality(
            gray_face,
            min_variance_laplacian=min_variance_laplacian,
        )
        timings_ms["quality_gate"] = (
            time.perf_counter() - quality_gate_start_time
        ) * 1000.0
        if not quality_result["is_valid_crop"]:
            return None, timings_ms

    preprocess_start_time = time.perf_counter()
    face = preprocess_face_for_pad(
        gray_face,
        0,
        0,
        gray_face.shape[1],
        gray_face.shape[0]
    )
    timings_ms["pad_preprocess"] = (
        time.perf_counter() - preprocess_start_time
    ) * 1000.0

    if face is None:
        return None, timings_ms

    features, feature_timings = extract_pad_features_with_timing(face)
    timings_ms.update(feature_timings)
    return features, timings_ms
