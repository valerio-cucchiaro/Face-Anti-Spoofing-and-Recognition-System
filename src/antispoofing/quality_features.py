"""
Module: quality_features

Description:
This module implements the extraction of image quality features used
for face anti-spoofing detection.

The implemented features describe the visual characteristics of a face
image that can help distinguish a real face from spoofing attacks such
as printed photos or replayed screens.

The module computes statistical descriptors from three image
representations:

- the grayscale face image
- the Laplacian response (high-frequency texture and sharpness)
- the Sobel gradient magnitude (edge strength)

For each representation, the mean and standard deviation are computed
and concatenated into a fixed-length feature vector. These descriptors
capture differences in brightness distribution, sharpness, and edge
structure that may appear in spoofed samples.

The resulting feature vector is used as input to the presentation
attack detection (PAD) classifier.
"""

import cv2
import numpy as np

def to_grayscale(image):
    '''Convert a color image to grayscale.'''

    if len(image.shape) == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def compute_basic_statistics(image):
    '''Compute basic statistical descriptors from an image or map.'''

    # Convert image to float for stable numerical operations
    image = image.astype(np.float32)

    # Compute average intensity
    mean_value = np.mean(image)

    # Compute intensity standard deviation
    std_value = np.std(image)

    # Return the statistics as a feature vector
    return np.array([
        mean_value,
        std_value,
    ], dtype=np.float32)


def extract_quality_features(face_image):
    '''Extract quality features from a given face image.'''

    # Convert input face to grayscale
    gray_face = to_grayscale(face_image).astype(np.uint8)

    # Compute Laplacian to capture high-frequency texture and sharpness
    laplacian = cv2.Laplacian(gray_face, cv2.CV_32F)

    # Compute horizontal and vertical image gradient
    sobel_x = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)

    # Compute gradient magnitude combining both directions
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)

    # Compute global brightness statistics
    gray_stats = compute_basic_statistics(gray_face)

    # Compute statistics describing image sharpness
    laplacian_stats = compute_basic_statistics(laplacian)

    # Compute statistics describing edge strength
    sobel_stats = compute_basic_statistics(sobel_magnitude)

    # Concatenate all statistics into a single feature vector
    return np.concatenate([
        gray_stats,
        laplacian_stats,
        sobel_stats
    ]).astype(np.float32)