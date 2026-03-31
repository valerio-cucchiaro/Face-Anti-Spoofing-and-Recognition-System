"""
Module: pad_lft_features

Description:
This module implements Local Feature Transformation (LFT) features
used for face anti-spoofing.

LFT describes the local luminance deviation of each pixel with respect
to its neighborhood by normalizing the difference between the pixel
intensity and the local mean with the local standard deviation.

The resulting LFT map highlights local texture and luminance artifacts
that are often introduced by spoofing attacks such as printed photos
or display screens.
"""

import cv2
import numpy as np


# ----- CONSTANTS ----- #
GAUSSIAN_KERNEL_SIZE = (5, 5)
EPSILON = 1e-6


def compute_lft_map(gray_face):
    """
    Compute the Local Feature Transformation (LFT) map from a grayscale face image.

    The LFT map measures the normalized deviation of each pixel from its
    local neighborhood statistics using the formula:

        LFT(x,y) = (I(x,y) - μ(x,y)) / (σ(x,y) + ε)

    where μ and σ are the local mean and standard deviation estimated
    with Gaussian smoothing.

    """

    gray_face = gray_face.astype(np.float32)

    # Estimate the local mean using Gaussian smoothing.
    mu = cv2.GaussianBlur(gray_face, GAUSSIAN_KERNEL_SIZE, 0)

    # Compute the difference between each pixel and its local mean.
    diff = gray_face - mu

    # Estimate the local variance from squared deviations.
    var = cv2.GaussianBlur(diff ** 2, GAUSSIAN_KERNEL_SIZE, 0)

    # Compute the local standard deviation.
    sigma = np.sqrt(var)

    # Normalize the local deviation using the local standard deviation.
    lft_map = diff / (sigma + EPSILON)

    return lft_map.astype(np.float32)


def extract_lft_features(face_image):
    """
    Extract statistical descriptors from the LFT map of a face image.

    The function computes the LFT map and then extracts global statistics
    that describe its distribution. These statistics capture asymmetry
    and heavy-tail behavior often associated with spoofing artifacts.

    """

    # Compute LFT representation of the face
    lft_map = compute_lft_map(face_image)

    # Flatten the map for statistical computation
    lft_map = lft_map.ravel()

    # Compute first-order statistics
    mean_value = np.mean(lft_map)
    std_value = np.std(lft_map)

    # Center values before computing higher-order moments
    centered_map = lft_map - mean_value

    # Prevent unstable statistics when variance is near zero
    if std_value < EPSILON:
        skewness_value = 0.0
        kurtosis_value = 0.0
    else:
        # Third standardized moment
        skewness_value = np.mean((centered_map / std_value) ** 3)

        # Fourth standardized moment
        kurtosis_value = np.mean((centered_map / std_value) ** 4)

    return np.array(
        [mean_value, std_value, skewness_value, kurtosis_value],
        dtype=np.float32
    )


def extract_lft_ggd_light_features(face_image):
    """
    Extract a lightweight GGD-inspired descriptor from the LFT map.

    The descriptor keeps only two stable statistics per scale:
    a shape proxy and the variance of the LFT distribution.
    """

    lft_map = compute_lft_map(face_image)
    x = lft_map.ravel().astype(np.float32)

    variance = np.var(x)
    mean_abs = np.mean(np.abs(x))
    mean_sq = np.mean(x ** 2)
    shape = mean_abs / (np.sqrt(mean_sq) + EPSILON)

    return np.array([shape, variance], dtype=np.float32)
