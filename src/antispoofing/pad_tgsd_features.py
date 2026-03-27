"""
Module: pad_tgsd_features

Description:
This module implements a lightweight TGSD-style descriptor for face
anti-spoofing.

The current implementation exposes a single scalar feature computed
from the difference between the gradient magnitude of the original face
image and the gradient magnitude of a lightly blurred version of the
same image.

This file is intentionally isolated from the existing PAD feature
blocks so TGSD can be tested as a separate feature family before being
integrated into the main PAD feature extractor.
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


def compute_mean_gradient_magnitude(gray_face):
    """
    Compute the average Sobel gradient magnitude of a grayscale face.
    """

    sobel_x = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(sobel_x, sobel_y)

    return float(np.mean(gradient_magnitude))


def extract_tgsd_feature(face_image):
    """
    Extract a single TGSD-style feature from a face image.

    The descriptor is the normalized difference between the mean
    gradient magnitude of the original image and the same quantity
    measured after a light Gaussian blur. Larger values indicate that
    fine-detail gradients are more strongly attenuated by smoothing.
    """

    gray_face = to_grayscale(face_image).astype(np.uint8)

    original_gradient_mean = compute_mean_gradient_magnitude(gray_face)

    blurred_face = cv2.GaussianBlur(gray_face, TGSD_GAUSSIAN_KERNEL_SIZE, 0)
    blurred_gradient_mean = compute_mean_gradient_magnitude(blurred_face)

    tgsd_value = (
        abs(original_gradient_mean - blurred_gradient_mean)
        / (original_gradient_mean + TGSD_EPSILON)
    )

    return np.array([tgsd_value], dtype=np.float32)
