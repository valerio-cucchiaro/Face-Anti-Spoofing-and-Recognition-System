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

from src.antispoofing.pad_frame_quality import (
    PAD_MIN_VARIANCE_LAPLACIAN,
    evaluate_pad_crop_quality,
)
from src.preprocessing.face_preprocessing import preprocess_face_for_pad
from src.antispoofing.pad_feature_extractor import extract_pad_features


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
