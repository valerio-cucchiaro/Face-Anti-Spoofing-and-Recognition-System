"""
Module: texture_analyzer

Description:
This module implements texture-based analysis used for the anti-spoofing
component of the system.

The goal is to distinguish between a real human face and a potential
presentation attack (e.g., printed photo or screen replay) by analyzing
visual texture characteristics of the detected face.

Two complementary metrics are computed:

* image sharpness using the variance of the Laplacian operator
* edge density using the Canny edge detector

These signals provide a simple heuristic to identify flat or artificial
surfaces that are typical of spoofing attempts.
"""

import cv2
import numpy as np


# ----- CONSTANTS ----- #
SHARPNESS_THRESHOLD = 40
EDGE_THRESHOLD = 0.03

def compute_sharpness(face):
    """
    Compute the sharpness of a face image using the variance of the Laplacian.

    Higher values indicate richer texture and stronger local intensity
    variations, which are typical of real facial skin.
    """

    laplacian = cv2.Laplacian(face, cv2.CV_64F)
    sharpness = laplacian.var()

    return sharpness

def compute_edge_density(face):
    """
    Compute the edge density of the face image using Canny edge detection.

    The metric represents the ratio of edge pixels to total pixels in the
    image and helps identify the amount of local detail in the face region.
    """

    edges = cv2.Canny(face, 100, 200)
    edge_pixels = np.sum(edges != 0)
    total_pixels = face.shape[0] * face.shape[1]
    edge_density = edge_pixels / total_pixels

    return edge_density

def analyze_texture(face):
    """
    Analyze the texture of a face image to detect potential spoof attempts.

    The function evaluates sharpness and edge density and compares them
    with predefined thresholds to determine whether the texture is
    consistent with a real face.
    """

    sharpness = compute_sharpness(face)
    edge_density = compute_edge_density(face)

    if sharpness > SHARPNESS_THRESHOLD and edge_density > EDGE_THRESHOLD:
        texture_ok = True
    else:
        texture_ok = False

    return texture_ok, sharpness, edge_density

