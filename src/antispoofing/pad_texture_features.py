"""
Module: pad_texture_features

Description:
This module implements texture-based feature extraction for the
anti-spoofing (presentation attack detection) component of the system.

It provides utilities to compute Local Binary Pattern (LBP) texture
descriptors from a face image. The module converts the input face to
grayscale, computes the LBP representation, and generates a normalized
histogram used as the feature vector for anti-spoofing classification.

The extracted LBP histogram captures micro-texture patterns that help
distinguish real human skin from spoof artifacts such as printed photos
or replayed images.
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern as lbp


def to_grayscale(image):
    '''Convert a color image to grayscale.'''

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_lbp_image(gray_face, radius):
    '''Compute the Local Binary Pattern (LBP) image from a grayscale face image.'''

    n_points = 8 * radius
    lbp_image = lbp(
        gray_face, 
        P = n_points, 
        R = radius, 
        method='uniform'
    )

    return lbp_image

def compute_lbp_histogram(lbp_image, n_points):
    '''Compute the histogram of LBP values from the LBP image.'''

    n_bins = n_points + 2 # Number of bins for uniform LBP patterns
    histogram, _ = np.histogram(lbp_image.ravel(), bins=n_bins, range=(0, n_bins))
    histogram = histogram.astype("float")
    histogram = histogram / (histogram.sum() + 1e-7)  # Normalize the histogram

    return histogram

def extract_lbp_features(face_image):
    '''
    Extract LBP texture features from a face image.

    The function computes Local Binary Pattern descriptors at multiple
    spatial scales and concatenates their histograms into a single
    feature vector used for spoof detection.
    '''

    # Ensure the image is grayscale because LBP works on intensity values
    gray_face = to_grayscale(face_image)

    # Use multiple radii to capture texture at different spatial scales
    radii = [1, 2, 3]

    # This list will accumulate all histograms from different radii
    all_features = []

    for radius in radii:

        # Number of sampling points around each pixel
        n_points = 8 * radius

        # Compute the LBP image, each pixel is replaced with a code describing the local texture
        lbp_image = compute_lbp_image(gray_face, radius)

        # Convert the LBP image into a normalized histogram
        lbp_histogram = compute_lbp_histogram(lbp_image, n_points)

        # Append histogram values to the final feature vector
        all_features.extend(lbp_histogram)

    # Convert the list to a NumPy vector used by the classifier
    return np.array(all_features, dtype=np.float32)
