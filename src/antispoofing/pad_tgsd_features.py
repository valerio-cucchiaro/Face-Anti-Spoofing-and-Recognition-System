"""
Module: pad_tgsd_features

Description:
This module implements a TGSD-style descriptor for face anti-spoofing.

The descriptor is based on a top gradient similarity (TGS) map built
from the gradient magnitude of the original face image and the
gradient magnitude of a lightly blurred version of the same image.

The final TGSD value is computed as the standard deviation of the top
5 percent highest values from the TGS map.
"""

import cv2
import numpy as np


# ----- CONSTANTS ----- #
TGSD_GAUSSIAN_KERNEL_SIZE = (3, 3)
TGSD_EPSILON = 1e-6


def to_grayscale(image):
    """Convert a color image to grayscale."""

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_gradient_magnitude(gray):
    """
    Compute the Sobel gradient magnitude of a grayscale image.
    """

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    return np.sqrt(gx ** 2 + gy ** 2)


def compute_tgs_map(gray):
    """
    Compute the top gradient similarity (TGS) map of a grayscale image.
    """

    blur = cv2.GaussianBlur(gray, TGSD_GAUSSIAN_KERNEL_SIZE, 0)

    g_orig = compute_gradient_magnitude(gray)
    g_blur = compute_gradient_magnitude(blur)

    return (
        (2.0 * g_orig * g_blur + TGSD_EPSILON)
        / (g_orig ** 2 + g_blur ** 2 + TGSD_EPSILON)
    )


def extract_tgsd_feature(face_image):
    """
    Extract a single TGSD feature from a face image.

    The feature is the standard deviation of the top 5 percent highest
    values from the TGS map.
    """

    gray_face = to_grayscale(face_image)
    tgs = compute_tgs_map(gray_face)
    values = tgs.flatten()

    k = max(1, int(0.05 * len(values)))
    top_values = np.partition(values, -k)[-k:]
    tgsd_value = np.std(top_values)

    return float(tgsd_value)


def extract_tgsd_features(face_image):
    """
    Extract TGSD features at two scales from a face image.
    """

    gray_face = to_grayscale(face_image)
    downsampled_face = cv2.resize(gray_face, (0, 0), fx=0.5, fy=0.5)

    feature_original = extract_tgsd_feature(gray_face)
    feature_downsampled = extract_tgsd_feature(downsampled_face)

    return np.array([feature_original, feature_downsampled], dtype=np.float32)
