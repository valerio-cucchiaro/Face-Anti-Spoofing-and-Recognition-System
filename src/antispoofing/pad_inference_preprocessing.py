"""
Module: pad_inference_preprocessing

Description:
This module provides runtime helper utilities for PAD inference.

It receives an already cropped face region from the main runtime
pipeline, applies the same face preprocessing used during training,
and extracts the PAD feature vector required by the classifier.
"""

import cv2

from src.preprocessing.face_preprocessing import preprocess_face_for_pad
from src.antispoofing.pad_feature_extractor import extract_pad_features


def extract_pad_features_from_crop(face_crop):
    """
    Extract PAD features from an already cropped runtime face image.
    """

    # Reject empty runtime crops before any preprocessing step.
    if face_crop is None or face_crop.size == 0:
        return None

    # Convert runtime crops to grayscale to match the PAD training pipeline.
    if len(face_crop.shape) == 3:
        gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray_face = face_crop

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
