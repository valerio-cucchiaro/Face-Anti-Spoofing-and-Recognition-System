"""
Module: pad_lftp_features

Description:
This module implements a simple Local Frequency-based Texture Pattern
(LFTP) descriptor for face anti-spoofing experiments.

The first version is intentionally lightweight: it computes a local
8-bit code from the normalized FFT magnitude of each 3x3 patch, builds
an LFTP code map, and summarizes it with a normalized 256-bin histogram.
"""

import cv2
import numpy as np


# ----- CONSTANTS ----- #
LFTP_PATCH_SIZE = 3
LFTP_HIST_BINS = 256


def to_grayscale(image):
    """Convert a BGR image to grayscale when needed."""

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_lftp_code(patch):
    """
    Compute the 8-bit LFTP code of a 3x3 patch.

    The code is built by comparing the normalized FFT-magnitude center
    value with its eight neighbors in clockwise order.
    """

    if patch.shape != (LFTP_PATCH_SIZE, LFTP_PATCH_SIZE):
        raise ValueError("compute_lftp_code expects a 3x3 patch")

    fft = np.fft.fft2(patch.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    magnitude_sum = np.sum(magnitude)
    if magnitude_sum > 0.0:
        magnitude /= magnitude_sum

    center = magnitude[1, 1]
    neighbors = (
        magnitude[0, 0],
        magnitude[0, 1],
        magnitude[0, 2],
        magnitude[1, 2],
        magnitude[2, 2],
        magnitude[2, 1],
        magnitude[2, 0],
        magnitude[1, 0],
    )

    code = 0
    for index, value in enumerate(neighbors):
        if value >= center:
            code |= (1 << index)

    return code


def compute_lftp_map(gray):
    """
    Compute the per-pixel LFTP code map of a grayscale image.

    Border pixels are left at zero because they do not have a valid 3x3
    neighborhood.
    """

    height, width = gray.shape
    lftp_map = np.zeros((height, width), dtype=np.uint8)

    for row in range(1, height - 1):
        for col in range(1, width - 1):
            patch = gray[row - 1:row + 2, col - 1:col + 2]
            lftp_map[row, col] = compute_lftp_code(patch)

    return lftp_map


def extract_lftp_hist(gray):
    """
    Extract a normalized 256-bin histogram from the LFTP code map.
    """

    if gray.size == 0:
        return np.zeros(LFTP_HIST_BINS, dtype=np.float32)

    lftp_map = compute_lftp_map(gray)
    hist, _ = np.histogram(
        lftp_map.ravel(),
        bins=LFTP_HIST_BINS,
        range=(0, LFTP_HIST_BINS),
    )

    hist = hist.astype(np.float32)
    hist_sum = np.sum(hist)
    if hist_sum > 0.0:
        hist /= hist_sum

    return hist
