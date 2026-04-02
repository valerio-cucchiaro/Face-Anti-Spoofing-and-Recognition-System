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
# Current promoted TGSD setting after the controlled comparisons:
# tau=170, gradient_operator='sobel', top_percent=0.05.
# Other combinations remain available here as archived experimental
# branches, but are not the current recommended runtime choice.
TGSD_TAU = 170.0
TGSD_TOP_PERCENT = 0.05
TGSD_GRADIENT_OPERATOR = "sobel"


def get_tgsd_config():
    """
    Return the TGSD configuration currently active in this module.
    """

    return {
        "tgsd_tau": float(TGSD_TAU),
        "tgsd_top_percent": float(TGSD_TOP_PERCENT),
        "tgsd_gradient_operator": TGSD_GRADIENT_OPERATOR,
    }


def configure_tgsd(
    tau=None,
    top_percent=None,
    gradient_operator=None,
):
    """
    Update the TGSD descriptor configuration at runtime.
    """

    global TGSD_TAU, TGSD_TOP_PERCENT, TGSD_GRADIENT_OPERATOR

    if tau is not None:
        if float(tau) <= 0.0:
            raise ValueError("TGSD tau must be > 0.")
        TGSD_TAU = float(tau)

    if top_percent is not None:
        if not (0.0 < float(top_percent) <= 1.0):
            raise ValueError("TGSD top_percent must be in the interval (0, 1].")
        TGSD_TOP_PERCENT = float(top_percent)

    if gradient_operator is not None:
        if gradient_operator != "sobel":
            raise ValueError(
                "Unsupported TGSD gradient operator: "
                f"{gradient_operator}. Use 'sobel'."
            )
        TGSD_GRADIENT_OPERATOR = gradient_operator


def to_grayscale(image):
    """Convert a color image to grayscale."""

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_gradient_magnitude(gray):
    """
    Compute the gradient magnitude of a grayscale image.
    """

    if TGSD_GRADIENT_OPERATOR == "sobel":
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    else:
        raise ValueError(
            "Unsupported TGSD gradient operator: "
            f"{TGSD_GRADIENT_OPERATOR}. Use 'sobel'."
        )

    return np.sqrt(gx ** 2 + gy ** 2)


def compute_tgs_map(gray):
    """
    Compute the top gradient similarity (TGS) map of a grayscale image.
    """

    blur = cv2.GaussianBlur(gray, TGSD_GAUSSIAN_KERNEL_SIZE, 0)

    g_orig = compute_gradient_magnitude(gray)
    g_blur = compute_gradient_magnitude(blur)

    return (
        (2.0 * g_orig * g_blur + TGSD_TAU)
        / (g_orig ** 2 + g_blur ** 2 + TGSD_TAU + TGSD_EPSILON)
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

    k = max(1, int(TGSD_TOP_PERCENT * len(values)))
    top_values = np.partition(values, -k)[-k:]
    tgsd_value = np.std(top_values)

    return float(tgsd_value)


def extract_tgsd_features(face_image):
    """
    Extract TGSD features at two scales from a face image.
    """

    gray_face = to_grayscale(face_image)
    blurred_gray = cv2.GaussianBlur(gray_face, TGSD_GAUSSIAN_KERNEL_SIZE, 0)
    downsampled_face = cv2.resize(blurred_gray, (0, 0), fx=0.5, fy=0.5)

    feature_original = extract_tgsd_feature(gray_face)
    feature_downsampled = extract_tgsd_feature(downsampled_face)

    return np.array([feature_original, feature_downsampled], dtype=np.float32)
